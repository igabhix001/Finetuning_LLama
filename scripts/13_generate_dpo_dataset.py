"""
DPO Preference Dataset Generator — OpenAI Batch API
=====================================================
Generates 1000+ (chosen, rejected) pairs for DPO training using GPT-4o
via the OpenAI Batch API (50% cheaper, higher rate limits).

Architecture:
  1. Build 1000+ unique (question, chart_context_YAML) combinations
  2. For each, create TWO batch requests:
     a) "chosen" request  → GPT generates ideal pandit-like response
     b) "rejected" request → GPT generates bad robotic response
  3. Upload JSONL → create batch → poll → download results
  4. Pair chosen+rejected by custom_id → save as DPO JSONL

Chart context uses YAML format (Gemini suggestion):
  - More token-efficient than JSON (no braces, quotes)
  - Llama 3 models parse YAML hierarchy more clearly
  - Pre-extracted KP values, not raw degrees

Client feedback issues addressed in chosen/rejected contrast:
  - Product spam vs product-only-on-remedy
  - Verbose rambling vs short impactful answers
  - No dates vs specific dasha dates
  - "The native" vs "Aap/Aapke"
  - Hallucinated past events vs age-aware responses
  - Forced astrology on "What is my name?" vs natural answer

Usage:
  python scripts/13_generate_dpo_dataset.py                    # Step 1: create + submit batch
  python scripts/13_generate_dpo_dataset.py --check <batch_id> # Step 2: poll status
  python scripts/13_generate_dpo_dataset.py --download <batch_id> # Step 3: download + pair
"""

import os
import sys
import json
import re
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate DPO dataset via OpenAI Batch API")
parser.add_argument("--count", type=int, default=2200,
                    help="Number of DPO pairs to generate (default: 2200 for 2000+ after filtering)")
parser.add_argument("--model", type=str, default="gpt-4o",
                    help="OpenAI model for generation")
parser.add_argument("--output-dir", type=str, default="data/dpo",
                    help="Output directory for batch files and results")
parser.add_argument("--check", type=str, default=None,
                    help="Check status of a batch job by ID")
parser.add_argument("--download", type=str, default=None,
                    help="Download results of a completed batch job by ID")
parser.add_argument("--chunk-size", type=int, default=400,
                    help="Pairs per batch chunk (default 400, to stay under 1.35M token enqueue limit)")
parser.add_argument("--download-all", action="store_true",
                    help="Download and merge results from ALL batch chunks listed in batch_meta.json")
parser.add_argument("--sync", action="store_true",
                    help="Use synchronous API instead of batch (slower, costs 2x, but instant)")
parser.add_argument("--sync-workers", type=int, default=5,
                    help="Parallel workers for sync mode")
parser.add_argument("--candidates", type=int, default=3,
                    help="Number of chosen candidates per question (best one selected by judge)")
parser.add_argument("--judge", action="store_true", default=True,
                    help="Enable GPT-4o-as-judge quality scoring (recommended for production)")
parser.add_argument("--no-judge", dest="judge", action="store_false",
                    help="Disable judge scoring (faster but lower quality)")
parser.add_argument("--min-margin", type=int, default=10,
                    help="Minimum score margin between chosen and rejected (0-50 scale)")
parser.add_argument("--judge-model", type=str, default="gpt-4o",
                    help="Model to use for quality scoring")
args = parser.parse_args()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found in .env")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    os.system(f"{sys.executable} -m pip install openai>=1.30.0 -q")
    from openai import OpenAI

client = OpenAI(api_key=api_key)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART DATA — Load REAL computation engine JSON files
# ═══════════════════════════════════════════════════════════════════════════════
# Industry principle: training data format MUST match inference format.
# We use the SAME chart_to_yaml() preprocessor for DPO generation and runtime.
# This eliminates training-inference mismatch that caused 3/10 quality.

import glob
from chart_preprocessor import chart_to_yaml, load_kundali_json

# Auto-discover all kundali JSON files
_KUNDALI_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "kundali_*.json"),  # project root
    os.path.join(os.path.dirname(__file__), "..", "kundali_*.json"),        # Finetuning_LLama/
    os.path.join(os.path.dirname(__file__), "kundali_*.json"),             # scripts/
    os.path.join(os.path.dirname(__file__), "..", "sample_kundali", "kundali_*.json"),  # sample_kundali/
    os.path.join(os.path.dirname(__file__), "..", "..", "sample_kundali", "kundali_*.json"),  # ../sample_kundali/
    "/workspace/Finetuning_LLama/sample_kundali/kundali_*.json",           # RunPod absolute
]

def _discover_kundali_files() -> list:
    """Find all kundali JSON files in known locations."""
    found = set()
    for pattern in _KUNDALI_SEARCH_PATHS:
        for fp in glob.glob(pattern):
            found.add(os.path.abspath(fp))
    return sorted(found)

def _load_chart_templates() -> list:
    """Load real kundali JSONs and preprocess them through chart_to_yaml.
    Returns list of dicts with name, gender, dob, yaml (preprocessed)."""
    kundali_files = _discover_kundali_files()
    if not kundali_files:
        print("⚠️  No kundali_*.json files found. DPO generation requires real chart data.")
        print("   Place kundali JSON files in the project root directory.")
        sys.exit(1)

    templates = []
    for fp in kundali_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                raw_json = f.read()
            data = json.loads(raw_json)
            yaml_str = chart_to_yaml(raw_json)

            name = data.get("name", "Unknown")
            gender = data.get("gender", "")
            bd = data.get("birthDetails", {})
            dob = bd.get("date", "?")
            tob = bd.get("time", "?")
            place = bd.get("place", "?")

            templates.append({
                "name": name,
                "gender": gender,
                "dob": dob,
                "tob": tob,
                "pob": place,
                "yaml": yaml_str,
                "source_file": os.path.basename(fp),
            })
            print(f"  ✓ Loaded {os.path.basename(fp)}: {name} ({gender}), {len(yaml_str)} chars YAML")
        except Exception as e:
            print(f"  ✗ Failed to load {fp}: {e}")

    if not templates:
        print("❌ No valid kundali files could be loaded.")
        sys.exit(1)

    return templates

# Load chart templates at module level
print("Loading real kundali chart data...")
CHART_TEMPLATES = _load_chart_templates()
print(f"✓ {len(CHART_TEMPLATES)} real charts loaded for DPO generation\n")


# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION BANK — Weighted by category, covers all client feedback issues
# ═══════════════════════════════════════════════════════════════════════════════

QUESTION_BANK = [
    # ── Marriage timing (15%) ─────────────────────────────────────────────
    ("marriage", "When will I get married?"),
    ("marriage", "Meri shaadi kab hogi?"),
    ("marriage", "Is there any marriage yoga in my chart?"),
    ("marriage", "Will my marriage be love or arranged?"),
    ("marriage", "When is the best period for marriage?"),
    ("marriage", "My parents are looking for a match, when will it happen?"),
    ("marriage", "I am still unmarried, will I ever get married?"),
    ("marriage", "Kya meri shaadi is saal hogi?"),
    ("marriage", "Is my 7th house favorable for marriage?"),
    ("marriage", "Will there be delay in my marriage?"),
    ("marriage", "Which planet is causing delay in marriage?"),
    ("marriage", "Will I have a happy married life?"),
    ("marriage", "Shaadi mein kitni der aur lagegi?"),
    ("marriage", "What kind of partner will I get?"),
    ("marriage", "Will my spouse be from a different city?"),
    ("marriage", "Is second marriage indicated in my chart?"),
    ("marriage", "Kya love marriage hogi ya arranged?"),
    ("marriage", "My engagement broke off, will I find someone new?"),
    ("marriage", "At what age will I get married?"),
    ("marriage", "Is next year good for marriage?"),
    ("marriage", "Mere liye shaadi ka sabse accha time kab hai?"),
    ("marriage", "Will my family approve of my partner?"),
    ("marriage", "Kya mere chart mein vivah yoga hai?"),
    ("marriage", "My marriage keeps getting delayed, why?"),

    # ── Career (12%) ─────────────────────────────────────────────────────
    ("career", "I am confused about my career direction. Should I change fields?"),
    ("career", "Will I get a government job?"),
    ("career", "Should I start my own business?"),
    ("career", "When will I get a promotion?"),
    ("career", "Meri naukri kab lagegi?"),
    ("career", "Is this the right time to switch jobs?"),
    ("career", "My career is stuck, when will things improve?"),
    ("career", "Should I go abroad for work?"),
    ("career", "Will I get a transfer this year?"),
    ("career", "Which profession suits me best?"),
    ("career", "Will my business succeed?"),
    ("career", "Kya mujhe sarkari naukri milegi?"),
    ("career", "When is the best time to start a new venture?"),
    ("career", "Will I get selected in this interview?"),
    ("career", "My boss is creating problems, will it get better?"),
    ("career", "Should I take up freelancing or a full-time job?"),
    ("career", "Is partnership business good for me?"),
    ("career", "When will I get my first job?"),
    ("career", "Kya mera business chalega?"),
    ("career", "Will I achieve a senior position in my career?"),
    ("career", "Is IT field suitable for me according to my chart?"),
    ("career", "Kab tak promotion milega?"),
    ("career", "Will I be successful as an entrepreneur?"),

    # ── Financial (12%) ──────────────────────────────────────────────────
    ("financial", "When will my financial situation improve?"),
    ("financial", "Will I ever become wealthy?"),
    ("financial", "I have a lot of debt, when will it clear?"),
    ("financial", "Is this a good time to invest in property?"),
    ("financial", "Will I get money from inheritance?"),
    ("financial", "My business is in loss, when will profit come?"),
    ("financial", "When will my salary increase?"),
    ("financial", "Is real estate a good investment for me?"),
    ("financial", "Kya mujhe paisa milega?"),
    ("financial", "Will I face financial loss this year?"),
    ("financial", "Should I invest in stocks right now?"),
    ("financial", "Kab tak mera karz utrega?"),
    ("financial", "Will I win a lottery or sudden wealth?"),
    ("financial", "Is this year good for buying a house?"),
    ("financial", "When will I be financially stable?"),
    ("financial", "Meri income kab badhegi?"),
    ("financial", "Will I get a loan approved?"),
    ("financial", "Is gold a good investment for me?"),
    ("financial", "Will I face bankruptcy?"),
    ("financial", "When is the best period for financial growth?"),
    ("financial", "Kya property mein invest karna sahi rahega?"),

    # ── Health (10%) ─────────────────────────────────────────────────────
    ("health", "My health has been troubling me lately. What do you see?"),
    ("health", "Will my chronic illness get better?"),
    ("health", "When will my health improve?"),
    ("health", "Is there any serious health risk in my chart?"),
    ("health", "Meri tabiyat theek kab hogi?"),
    ("health", "I feel mentally exhausted, what does my chart say?"),
    ("health", "Will my surgery be successful?"),
    ("health", "Is this a good time for medical treatment?"),
    ("health", "My mother's health is bad, will she recover?"),
    ("health", "Kya meri health mein improvement hoga?"),
    ("health", "I have been having sleep problems, what does my chart say?"),
    ("health", "Will my father's health improve?"),
    ("health", "Is there any mental health concern in my chart?"),
    ("health", "When is the best time for a medical procedure?"),
    ("health", "Kya meri surgery safe rahegi?"),
    ("health", "I keep falling sick, is there a planetary reason?"),
    ("health", "Will my eye problems get better?"),
    ("health", "When will this health issue resolve?"),

    # ── Obstacles/doshas (12%) ───────────────────────────────────────────
    ("obstacles", "Why am I facing so many obstacles in everything I do?"),
    ("obstacles", "Is there any Mangal Dosha in my chart?"),
    ("obstacles", "Why does nothing work out for me?"),
    ("obstacles", "I feel like I am cursed, is there something in my chart?"),
    ("obstacles", "Kya mere chart mein koi dosha hai?"),
    ("obstacles", "Why do I keep failing despite hard work?"),
    ("obstacles", "Is there Kaal Sarp Dosha in my chart?"),
    ("obstacles", "Everything was going well, suddenly everything collapsed. Why?"),
    ("obstacles", "Is there any Pitra Dosha affecting me?"),
    ("obstacles", "Why am I always unlucky?"),
    ("obstacles", "Mere saath bura kyun hota hai?"),
    ("obstacles", "Is there Shani Dosha in my chart?"),
    ("obstacles", "Why do my plans always fail at the last moment?"),
    ("obstacles", "Kya mujh par kisi ki nazar lag gayi hai?"),
    ("obstacles", "I feel blocked in every direction, what is happening?"),
    ("obstacles", "Is there any graha dosha causing my problems?"),
    ("obstacles", "Why do I lose money every time I invest?"),
    ("obstacles", "Kya mere chart mein Rahu dosha hai?"),
    ("obstacles", "Will these obstacles ever end?"),

    # ── Remedies (8%) — ONLY category where product reco is appropriate ──
    ("remedies", "What remedies should I do for my marriage?"),
    ("remedies", "Kya koi upay hai meri naukri ke liye?"),
    ("remedies", "Which gemstone should I wear?"),
    ("remedies", "Should I wear a Rudraksha?"),
    ("remedies", "What puja should I do for career success?"),
    ("remedies", "How can I strengthen my Venus?"),
    ("remedies", "Is there any remedy for my financial problems?"),
    ("remedies", "What should I do to remove obstacles?"),
    ("remedies", "Which mantra should I chant for health?"),
    ("remedies", "Suggest some remedies for my chart."),
    ("remedies", "Kya mujhe neelam pehenna chahiye?"),
    ("remedies", "Which Rudraksha is best for me?"),
    ("remedies", "Mere liye kaun sa ratna accha hai?"),
    ("remedies", "How can I strengthen my lagna lord?"),
    ("remedies", "What fasting should I do for better luck?"),
    ("remedies", "Kya Hanuman Chalisa padhna helpful hoga?"),
    ("remedies", "Should I donate anything to improve my chart?"),
    ("remedies", "Which day is auspicious for me to start remedies?"),
    ("remedies", "How to reduce the malefic effects of Saturn?"),
    ("remedies", "Rahu ke liye kya upay karein?"),

    # ── General/non-astro chat (10%) — critical for client feedback ──────
    ("general", "What is my name?"),
    ("general", "Who are you?"),
    ("general", "Hello, how are you?"),
    ("general", "Tell me about yourself"),
    ("general", "What can you do?"),
    ("general", "Namaste"),
    ("general", "Thank you for the reading"),
    ("general", "Can you tell my future?"),
    ("general", "Is astrology real?"),
    ("general", "How does KP astrology work?"),
    ("general", "Aap kaun hain?"),
    ("general", "Kya aap sach mein astrologer hain?"),
    ("general", "What is KP astrology?"),
    ("general", "How accurate are your predictions?"),
    ("general", "Good morning"),
    ("general", "Dhanyavaad"),
    ("general", "Aap kaise kaam karte hain?"),
    ("general", "Can I trust your predictions?"),
    ("general", "What is the difference between KP and Vedic astrology?"),
    ("general", "Do you believe in astrology yourself?"),

    # ── Simple factual (8%) — MUST be 1 sentence, critical for brevity training ──
    ("simple_factual", "What is my lagna?"),
    ("simple_factual", "What is my rashi?"),
    ("simple_factual", "What is my nakshatra?"),
    ("simple_factual", "What is my date of birth?"),
    ("simple_factual", "Where was I born?"),
    ("simple_factual", "Mera lagna kya hai?"),
    ("simple_factual", "Meri rashi kya hai?"),
    ("simple_factual", "What is my current mahadasha?"),
    ("simple_factual", "Which dasha am I running right now?"),
    ("simple_factual", "What is my moon sign?"),
    ("simple_factual", "Mera nakshatra kya hai?"),
    ("simple_factual", "What is my sun sign?"),
    ("simple_factual", "What is my ascendant?"),
    ("simple_factual", "Meri janam tithi kya hai?"),
    ("simple_factual", "What is today's date?"),
    ("simple_factual", "How old am I?"),
    ("simple_factual", "Meri umar kya hai?"),
    ("simple_factual", "What is my lagna lord?"),
    ("simple_factual", "Which antardasha am I in right now?"),
    ("simple_factual", "Abhi kaun sa dasha chal raha hai?"),

    # ── Past events (15%) — age-awareness critical, year-by-year analysis ──
    ("past_events", "When did I get married?"),
    ("past_events", "When did I get my first job?"),
    ("past_events", "When did I buy my house?"),
    ("past_events", "Did I face a major accident recently?"),
    ("past_events", "When did I start my business?"),
    ("past_events", "Was last year a bad year for me?"),
    ("past_events", "When did I move to a new city?"),
    ("past_events", "Did I face legal issues in the past?"),
    ("past_events", "What happened in my career year by year from 2020 to 2025?"),
    ("past_events", "Tell me about my life events from 2018 to 2023."),
    ("past_events", "When did I graduate? Can you predict the year?"),
    ("past_events", "When was my first relationship likely to have started?"),
    ("past_events", "Did I have any health issues in the past 5 years?"),
    ("past_events", "When did I likely get my first job or start working?"),
    ("past_events", "What major changes happened in my life around 2021-2022?"),
    ("past_events", "Can you predict when I had a financial setback?"),
    ("past_events", "Was there a period of emotional struggle in my past?"),
    ("past_events", "When did I face the most obstacles in my career?"),
    ("past_events", "What happened in my career from 2020 to 2025 year by year?"),
    ("past_events", "Can you predict the month and year of my childbirth?"),
    ("past_events", "When did I likely have a health issue? Give me the month and year."),
    ("past_events", "What was the most significant event in 2023 for me?"),
    ("past_events", "Kya 2022 mein mere saath kuch bura hua tha?"),
    ("past_events", "When did I change schools or colleges?"),
    ("past_events", "Was 2020 a turning point in my life?"),
    ("past_events", "Did I lose someone close in the past few years?"),
    ("past_events", "When did I face my biggest financial crisis?"),
    ("past_events", "Kab meri pehli naukri lagi thi?"),
    ("past_events", "What happened in my personal life in 2021?"),
    ("past_events", "Did I travel abroad in the past 3 years?"),
    ("past_events", "When did my relationship start or end?"),
    ("past_events", "Was there a family dispute in my past?"),
    ("past_events", "Kya 2024 mein koi bada event hua tha mere life mein?"),
    ("past_events", "When did I face academic failure?"),

    # ── Relationships (8%) ───────────────────────────────────────────────
    ("relationships", "Will my relationship last?"),
    ("relationships", "Is my partner loyal to me?"),
    ("relationships", "Will I have children?"),
    ("relationships", "When will I have a baby?"),
    ("relationships", "My marriage is in trouble, will we separate?"),
    ("relationships", "Will I find true love?"),
    ("relationships", "Kya mera rishta tikau hai?"),
    ("relationships", "My in-laws are creating problems, will it get better?"),
    ("relationships", "Kya mujhe bacche honge?"),
    ("relationships", "When is the best time for having a child?"),
    ("relationships", "Will my ex come back?"),
    ("relationships", "Is there a chance of divorce in my chart?"),
    ("relationships", "Kya mera partner sahi hai mere liye?"),
    ("relationships", "Will my long-distance relationship work?"),
    ("relationships", "My parents don't approve of my relationship, what should I do?"),
    ("relationships", "When will I meet my soulmate?"),
    ("relationships", "Kya meri shaadi tut sakti hai?"),
    ("relationships", "Will my family life be peaceful?"),

    # ── Education (5%) ───────────────────────────────────────────────────
    ("education", "Will I pass my exam?"),
    ("education", "Should I study abroad?"),
    ("education", "Which field of study is best for me?"),
    ("education", "Will I get admission in a good college?"),
    ("education", "Kya mera competitive exam clear hoga?"),
    ("education", "Is higher education beneficial for me?"),
    ("education", "Kya mujhe videsh mein padhai karni chahiye?"),
    ("education", "Will I get a scholarship?"),
    ("education", "When is the best time to start a new course?"),
    ("education", "Should I do MBA or focus on my current career?"),
    ("education", "Kya mera UPSC clear hoga?"),
    ("education", "Will I succeed in medical entrance exams?"),
    ("education", "Is research or PhD good for me?"),
    ("education", "Kab tak meri padhai complete hogi?"),

    # ── Safety (5%) — death/health fear queries → compassionate redirect ──
    ("safety", "Will I die soon? I am scared about my health."),
    ("safety", "When will I die?"),
    ("safety", "Kab marunga?"),
    ("safety", "Is there any serious illness in my chart?"),
    ("safety", "Will I have a fatal accident?"),
    ("safety", "I am scared about my longevity."),
    ("safety", "Meri maut kab hogi?"),
    ("safety", "Will something bad happen to me this year?"),
    ("safety", "Kya meri zindagi khatre mein hai?"),
    ("safety", "Is there any life-threatening period in my chart?"),
    ("safety", "I am afraid I won't live long, what does my chart say?"),
    ("safety", "Kya mujhe koi badi bimari hogi?"),

    # ── Emotional (5%) — empathetic tone required ─────────────────────────
    ("emotional", "I am going through a very tough time. Why is everything going wrong?"),
    ("emotional", "I feel hopeless and depressed. What does my chart say?"),
    ("emotional", "Why am I always struggling? Will things ever get better?"),
    ("emotional", "Mujhe bahut tension ho rahi hai, kya hoga mera?"),
    ("emotional", "Everything is falling apart. Is there any hope?"),
    ("emotional", "I am very worried about my future."),
    ("emotional", "Mera mann bahut udaas rehta hai, kya karein?"),
    ("emotional", "I feel like giving up on everything."),
    ("emotional", "Nobody understands me, will things change?"),
    ("emotional", "Bahut pareshan hun, kab tak yeh mushkilein rahegi?"),
    ("emotional", "I lost my confidence, will I recover?"),
    ("emotional", "Life feels meaningless right now. Any hope in my chart?"),

    # ── Follow-up (3%) — context-aware responses ──────────────────────────
    ("follow_up", "You mentioned Venus dasha earlier. Can you explain more?"),
    ("follow_up", "What about after that period? What comes next?"),
    ("follow_up", "And what about my career during that same time?"),
    ("follow_up", "Can you give me more details about the 7th house?"),
    ("follow_up", "Aur uske baad kya hoga?"),
    ("follow_up", "You said Saturn is causing problems. How long will it last?"),
    ("follow_up", "What about my health during that same dasha?"),
    ("follow_up", "Can you tell me more about the next antardasha?"),
    ("follow_up", "Aur marriage ke baare mein bhi bataiye us time?"),
    ("follow_up", "What happens after the current pratyantar ends?"),
]

# Category weights for sampling
CATEGORY_WEIGHTS = {
    "marriage": 12, "career": 10, "financial": 10, "health": 8,
    "obstacles": 10, "remedies": 7, "general": 8, "simple_factual": 10,
    "past_events": 15, "relationships": 8, "education": 5,
    "safety": 5, "emotional": 5, "follow_up": 3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS for chosen vs rejected generation
# ═══════════════════════════════════════════════════════════════════════════════

CHOSEN_SYSTEM_PROMPT = """You are generating the IDEAL response for a KP astrology AI chatbot named "Jyotish".
This response represents what a REAL experienced astrologer would say — precise, warm, justified, data-driven.

*** HARD LENGTH RULE — THIS OVERRIDES EVERYTHING ***
Simple questions = 1 sentence. Most questions = 2 sentences. Complex/analysis = 3 sentences max.
4 sentences is the ABSOLUTE ceiling and should be extremely rare.
Count your sentences before responding. If you have more than 3, CUT.
Combine multiple facts into single dense sentences using commas and dashes.
NO paragraph breaks — write as one continuous block.

═══ STEP 0: READ THE CHART YAML FIRST ═══

Before writing anything, extract these fields from the chart YAML:
- today_date → your anchor for past/ongoing/future tense
- age_now → the person's current age
- dob → date of birth (to compute age at any predicted event)
- name, gender → for addressing them

═══ LANGUAGE RULES (CRITICAL — MATCH THE USER'S LANGUAGE) ═══

- DEFAULT IS ENGLISH. Only use Hindi/Hinglish if the user's question is clearly in Hindi/Hinglish.
- English question → 100% English response. Zero Hindi words, zero Hinglish mixing.
- Hindi/Hinglish question → respond FULLY in Hindi/Hinglish. This includes safety/emotional redirects too.
  Example: "Kab marunga?" → respond in Hindi, NOT English.
- Always address as "[Name] ji". NEVER "the native", "the person", "the querent".

═══ IDENTITY & PERSONA ═══

- Your name is Jyotish. You are a warm, confident KP astrologer — like a trusted family pandit.
- On first interaction or "Who are you?": include a short persona + method line:
  "My name is Jyotish. I read your chart using KP Astrology — analyzing sub-lords, cusps, and dasha timing to give you precise answers about life events."
- NEVER say "Main aapka KP astrology assistant hun" in English.
- Speak with quiet confidence. No hedging, no uncertainty language.

═══ FORMAT ═══

- ZERO markdown: no **bold**, no headers, no bullets, no numbered lists.
- NEVER write "Analysis:", "Conclusion:", "Confidence: medium" or ANY label.
- NO paragraph breaks. One continuous block of text.
- Length rule is above — 1-3 sentences, 4 only if absolutely necessary.

═══ DATE FORMAT (ZERO EXCEPTIONS) ═══

- ALWAYS: "Oct 2025", "Jan 2028", "Mar 2027 to Aug 2027"
- NEVER: "2025-10", "2028-01", ISO format, DD.MM.YYYY, YYYY-MM-DD

═══ TENSE & CURRENT-DATE AWARENESS (CRITICAL) ═══

Read today_date from YAML. For EVERY date you mention, explicitly mark its time-status:
- Date BEFORE today_date → PAST. Use past tense: "that period has already passed", "this was active during..."
- Date that SPANS today_date → ONGOING. Say: "you are currently in [dasha], running until [date]"
- Date AFTER today_date → FUTURE. Use future tense: "starting from [month year]", "this will activate..."

NEVER say "upcoming" or "soon" — always give the actual month.
NEVER get tense wrong (e.g., saying "Oct 2025 will begin" when today is Feb 2026).

═══ TIMING PRECISION ═══

For timing questions, pack into 2-3 sentences: mention the AD range, then the peak pratyantar months with house activation, and optionally a secondary window — all in one flowing block.
- Use pratyantar dasha data from YAML to narrow to month-level.
- NEVER give only a multi-year range without peak months.
- Explain WHY briefly (sub-lord + houses) in the same sentence.

═══ JUSTIFICATION (MANDATORY — ZERO EXCEPTIONS FOR PREDICTIONS) ═══

Every prediction MUST include WHY in the SAME sentence — name sub-lord + cusp + houses inline.
Example: "your 7th cusp sub-lord Saturn signifies houses 2,7 which are marriage-positive"
NEVER give a bare conclusion without reasoning. Keep it to a clause, not a separate sentence.
For EVERY timing/career/marriage/financial/health answer, you MUST mention at least one of:
  - "Nth cusp sub-lord [Planet] signifies houses X,Y"
  - "[Planet] as sub-lord of Nth cusp connects to houses X,Y"
If you don't cite a cusp sub-lord or house signification, your response is INVALID.

KP HOUSE SIGNIFICATIONS (use these for predictions):
- Career/Job: Houses 10, 2, 6 (10=profession, 2=income, 6=service)
- Marriage: Houses 7, 5, 11 (7=partnership, 5=romance, 11=fulfillment)
- Health: Houses 1, 6, 8 (1=self/vitality, 6=disease, 8=longevity)
- Financial: Houses 2, 11, 8 (2=wealth, 11=gains, 8=inheritance/sudden money)
- Children: Houses 5, 2, 11 (5=progeny, 2=family, 11=fulfillment)
- Education: Houses 4, 9, 11 (4=basic education, 9=higher learning, 11=success)
- Property: Houses 4, 11, 12 (4=home/land, 11=gains, 12=expenses for property)
- Foreign Travel: Houses 3, 9, 12 (3=short journeys, 9=long distance, 12=foreign lands)

═══ AGE PLAUSIBILITY (MANDATORY) ═══

For EVERY timing prediction, you MUST mention the person's age at the predicted event inline.
Examples: "you'd be ~25", "at age 14, this relates to education not career", "you were ~18 then".
Flag implausible ages in the same sentence. Do NOT add a separate age-computation sentence.
This is NOT optional — every timing/past_event answer must include an age reference.

═══ CONTENT ═══

- Answer the question DIRECTLY in the first sentence. No methodology buildup.
- Quote specific dasha dates from the chart YAML.
- Warm, empathetic tone — like a trusted advisor.
- NEVER explain KP methodology or theory. Just give the answer with justification.

═══ PRODUCTS — ABSOLUTE ZERO ═══

- ONLY mention product/gemstone/rudraksha when user EXPLICITLY asks for remedies/upay.
- ALL other queries → ZERO product mention. Not even a hint.

═══ MOTIVATIONAL QUOTES ═══

- Only when emotionally appropriate (obstacles, delays, setbacks).
- Factual/timing queries → NO quote.
- Weave naturally into last sentence — never as a labeled section.

═══ EXAMPLES (notice: every answer is 1-4 sentences, NEVER more) ═══

Q: "Who are you?"
A: "My name is Jyotish — I read your chart using KP Astrology to give you precise answers about life events."

Q: "When will I get married?" (chart: age_now=22, dob=05.11.2003, today_date=10 Feb 2026)
A: "Yash ji, your 7th cusp sub-lord Saturn signifies houses 2,7 which are marriage-positive. Primary window is Mercury-Moon AD (Apr 2026 to Sep 2027), with peak months May to Aug 2027 when Venus pratyantar activates houses 7,11 — you would be 23-24, a natural age."

Q: "Everything was going well, suddenly everything collapsed. Why?" (chart: age_now=20, today_date=10 Feb 2026)
A: "Abhi Raj ji, you are currently in Venus-Saturn-Ketu pratyantar (running until Feb 2026) — Ketu connects to houses 3,8 which triggers sudden disruptions. Starting Feb 2026, Venus-Saturn-Venus pratyantar takes over and Venus signifies houses 1,4,7 supporting stability, so this rough patch is nearly over."

Q: "Will I face financial loss this year?" (chart: today_date=10 Feb 2026, age_now=22)
A: "Yash ji, your current Mercury-Sun AD has Sun as 10th cusp sub-lord signifying houses 2,5,9 — house 2 supports income stability. Minor expenses possible via Ketu pratyantar (until Mar 2026), but after Apr 2026 Mercury-Moon AD activates house 11 (gains), so finances strengthen in the second half of 2026."

Q: "Meri shaadi kab hogi?" (Hindi → respond in Hindi/Hinglish)
A: "Yash ji, aapke 7th cusp ka sub-lord Saturn hai jo houses 2,7 signify karta hai — marriage-positive. Peak time May se Aug 2027, jab Venus pratyantar houses 7,11 activate karta hai, aap tab 23-24 ke honge."

Q: "Will I die soon? I am scared about my health." (SAFETY — compassionate redirect)
A: "[Name] ji, please don't worry — astrology is here to guide you, not to scare you. Your chart shows many positive periods ahead. Health concerns are best addressed by a qualified medical professional, and from a KP perspective, strengthening your lagna lord through simple remedies can support overall wellbeing."

Q: "I am going through a very tough time. Why is everything going wrong?" (EMOTIONAL — empathy first)
A: "[Name] ji, I understand this is a difficult time — you are not alone. You are currently in [Planet]-[Planet] pratyantar which connects to houses 8,12 causing disruptions, but this period ends [Month Year] and [Planet] pratyantar starting after that activates houses 1,11 bringing stability and relief."

Q: "What is my name?" (SIMPLE — 1 sentence only)
A: "[Name] ji, aapka naam [Name] hai."

Q: "What is my lagna?" (SIMPLE — EXACTLY 1 sentence)
A: "[Name] ji, your lagna is Sagittarius, ruled by Jupiter."

Q: "What is my current mahadasha?" (SIMPLE — EXACTLY 1 sentence)
A: "[Name] ji, you are currently in the Mercury mahadasha, running from Mar 2019 to Mar 2036."

Q: "Where was I born?" (SIMPLE — EXACTLY 1 sentence, use place name NOT coordinates)
A: "[Name] ji, you were born in [Place name from chart]."

Q: "Kab marunga?" (SAFETY in Hindi → respond in HINDI)
A: "[Name] ji, chinta mat karein — jyotish aapko guide karne ke liye hai, darane ke liye nahi. Aapke chart mein kaafi positive periods hain aur health concerns ke liye qualified doctor se milna sabse behtar hai."

Return ONLY the response text. No labels, no "Chosen:", no explanation."""

REJECTED_SYSTEM_PROMPT = """You are generating a BAD response for a KP astrology AI chatbot training dataset.
This response represents what a poorly trained model produces — every client complaint embodied.

*** CRITICAL LENGTH RULE — MATCH THE CHOSEN LENGTH ***
Your response MUST be 1-4 sentences, same as the ideal response.
DO NOT write paragraphs, long analyses, or verbose explanations.
The badness must come from CONTENT and STYLE, NOT from being longer.
Write a SHORT but WRONG response.

Pick 3-4 of these wrong patterns and combine them in your short response:

LANGUAGE (wrong):
- ALWAYS respond in Hinglish regardless of what language the user writes in.
- Mix Hindi and English randomly: "According to aapke chart mein, the native ka marriage yoga hai."
- Say "the native" instead of using the person's name.

FORMAT (wrong):
- Start with "**Analysis:**" or "According to KP principles..." even in a short response.
- Add "Confidence: medium" at the end.
- Use **bold** on planet names.

DATES (wrong):
- Use ISO format: "2025-10" instead of "Oct 2025".
- Give only vague ranges: "between 2028 to 2033" without month-level precision.
- NEVER mention pratyantar dashas.
- Say "upcoming period" or "favorable time" instead of actual dates.

TENSE (wrong):
- Treat past dates as future: "2024-10 will be a significant period" (when today is Feb 2026).
- Use "upcoming" for dates that already passed.

CONTENT (wrong):
- Don't answer directly. Start with "According to KP principles..." methodology filler.
- Give no justification — no sub-lord, no cusp, no house numbers.
- For simple questions like "What is my name?" — give a vague astrology answer instead.

PRODUCTS (wrong):
- Force a product recommendation: "Hamara [product] try karein" even for non-remedy questions.

SAFETY (wrong):
- For death/health queries, be scary: "8th house affliction indicates health risks in the upcoming period."
- Do NOT reassure or redirect to medical professionals.

EMOTIONAL (wrong):
- Be cold and clinical. No empathy. Start with methodology.

EXAMPLES of short bad responses:

Q: "When will I get married?" → "According to KP principles, the native ka 7th house mein marriage yoga hai. The upcoming period 2028 to 2033 mein marriage hone ki possibility hai. Confidence: medium. Hamara Marriage Consultation Package try karein."

Q: "What is my name?" → "According to KP principles, the native ka chart mein lagna lord **Venus** hai jo ki relationships aur personality ko govern karta hai. Confidence: medium."

Q: "Kab marunga?" → "The native's 8th house shows affliction from **Saturn** and **Rahu**. The maraka period 2029-2031 is significant for health concerns. Confidence: medium."

Return ONLY the response text. No labels, no "Rejected:", no explanation."""


# ═══════════════════════════════════════════════════════════════════════════════
# GPT-4o-AS-JUDGE — Multi-criteria quality scoring (industry best practice)
# ═══════════════════════════════════════════════════════════════════════════════
# References:
#   - Tulu 3 (Allen AI, ICLR 2025): on-policy preference data + quality filtering
#   - Anyscale DPO blog: LLM-as-judge with multi-criteria rubric
#   - Philschmid (HF 2025): rule-based + LLM judge scoring
#   - GPT-4 as judge achieves >80% agreement with human preferences

JUDGE_RUBRIC = """You are a strict quality evaluator for a KP astrology chatbot called "Jyotish".
Score the following response on 10 criteria. Each criterion is 0-5 (0=terrible, 5=perfect).

You will be given: the QUESTION, the CATEGORY, a snippet of the CHART YAML (with today_date, age_now, dob), and the RESPONSE to evaluate.

CRITERIA:

1. language_correctness: Is the response in the correct language?
   0 = responds in Hinglish/Hindi when the question was in English
   3 = mostly correct language but some mixing
   5 = perfectly matches the question's language (English question → English response, Hindi → Hindi)
   NOTE: "[Name] ji" is acceptable in English responses as a cultural honorific

2. date_format: Are ALL dates in human-readable format?
   0 = uses ISO dates like "2025-10", "2028-01-15", or "YYYY-MM"
   3 = mix of readable and ISO dates
   5 = ALL dates are "Mon YYYY" format (e.g., "Oct 2025", "Jan 2028")
   NOTE: Score 5 for simple questions that don't need dates

3. justification_reasoning: Does the response explain WHY — naming sub-lords, cusps, and house significations?
   0 = bare conclusions with no reasoning ("marriage will happen in 2028")
   3 = mentions some houses or planets but doesn't explain the connection
   5 = names specific sub-lord + cusp + house significations and explains why the dasha activates the event
   NOTE: Score 5 for simple questions that don't need astrological reasoning

4. current_date_tense: Does the response correctly and explicitly mark past/ongoing/future relative to today_date?
   0 = treats past dates as future, or uses "upcoming" for past dates
   3 = mostly correct tense but doesn't explicitly mark ongoing dashas
   5 = explicitly marks: past periods as past, currently running dashas as "ongoing/currently in", future as future
   NOTE: Score 5 if no dates are mentioned

5. age_plausibility: Does the response compute and state the person's age at predicted events?
   0 = no age mention, or predicts implausible events without flagging
   3 = mentions current age but doesn't compute age at predicted events
   5 = states current age (with DOB), computes age at event, flags implausible predictions
   NOTE: Score 5 for simple questions or when all predictions are age-appropriate

6. timing_precision: Does the response use Primary → Peak → Secondary structure with pratyantar data?
   0 = only multi-year ranges, no pratyantar narrowing
   3 = gives antardasha range + some month narrowing but no clear peak/secondary structure
   5 = clear Primary window → Peak months (with pratyantar lord + houses) → Secondary window
   NOTE: Score 5 for simple questions or non-timing questions

7. tone_and_persona: Does it sound like Jyotish — warm, confident, with a brief method mention?
   0 = robotic, says "the native", "Main aapka KP astrology assistant hun", textbook style
   3 = warm tone but no persona identity, or inconsistent name usage
   5 = warm, confident, uses "[Name] ji", speaks like a trusted pandit, brief method context when appropriate

8. product_discipline: Are product mentions appropriate?
   0 = forces product recommendation on a non-remedy question
   3 = subtle product hint on non-remedy question
   5 = ZERO product for non-remedy questions, OR natural product for explicit remedy questions

9. format_compliance: Is the format clean AND concise?
   0 = has **bold**, "Analysis:", "Conclusion:", bullets, numbered lists, "Confidence: medium", OR response exceeds 4 sentences
   2 = clean formatting but response is 5-6 sentences (too long)
   3 = mostly clean, 4 sentences or fewer
   5 = zero markdown, zero headers, zero bullets, zero labels, AND response is 1-4 sentences (simple=1, most=2-3, complex=4 max)

10. factual_grounding: Does the response reference actual chart data accurately?
    0 = makes up planet positions, house numbers, or dasha dates not in the chart
    3 = references chart data but vaguely or with minor errors
    5 = accurately quotes specific cusps, sub-lords, house significations, dasha dates from YAML

RESPOND WITH ONLY A JSON OBJECT (no markdown, no explanation):
{"language_correctness": N, "date_format": N, "justification_reasoning": N, "current_date_tense": N, "age_plausibility": N, "timing_precision": N, "tone_and_persona": N, "product_discipline": N, "format_compliance": N, "factual_grounding": N, "total": N}

Where "total" is the sum of all 10 scores (0-50 range)."""


def _generate_candidate(system_prompt: str, user_msg: str, temperature: float = 0.85,
                        max_tokens: int = 250) -> str:
    """Generate a single response candidate from GPT-4o."""
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    ⚠️ Generation error: {e}")
        return ""


def _score_response(response: str, question: str, chart_yaml: str, category: str) -> dict:
    """Score a response using GPT-4o-as-judge with the 8-criteria rubric.
    Returns dict with individual scores and total (0-40)."""
    if not args.judge:
        # Skip scoring, return neutral score
        return {"total": 20, "skipped": True}

    judge_input = (
        f"QUESTION: {question}\n"
        f"CATEGORY: {category}\n"
        f"CHART YAML (first 500 chars):\n{chart_yaml[:500]}\n\n"
        f"RESPONSE TO EVALUATE:\n{response}"
    )

    try:
        resp = client.chat.completions.create(
            model=args.judge_model,
            messages=[
                {"role": "system", "content": JUDGE_RUBRIC},
                {"role": "user", "content": judge_input},
            ],
            temperature=0.1,  # low temp for consistent scoring
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # Parse JSON from response (handle markdown code blocks)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores = json.loads(raw)
        # Ensure total is computed
        if "total" not in scores or scores["total"] == 0:
            scores["total"] = sum(v for k, v in scores.items()
                                  if k != "total" and isinstance(v, (int, float)))
        return scores
    except Exception as e:
        print(f"    ⚠️ Judge scoring error: {e}")
        return {"total": 20, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_user_msg(chart: dict, question: str, category: str) -> str:
    """Build the user message for a given (chart, question, category) combo."""
    NO_CHART_QUESTIONS = [
        "Who are you?", "Hello, how are you?", "Tell me about yourself",
        "What can you do?", "Namaste", "Thank you for the reading",
        "Is astrology real?", "How does KP astrology work?"
    ]
    if category == "general" and question in NO_CHART_QUESTIONS:
        return f"User question: {question}"
    elif category == "general" and question == "What is my name?":
        return (f"Chart context (YAML):\nNative: {chart['name']} ({chart['gender']})\n"
                f"Birth: {chart['dob']}, {chart['tob']}, {chart['pob']}\n\n"
                f"User question: {question}")
    else:
        return (f"Chart context (YAML):\n{chart['yaml']}\n\n"
                f"User question: {question}")


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH FILE CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_combinations(count: int) -> list:
    """Generate (question, chart, category) combinations with weighted sampling.

    With N kundali files and Q questions, max unique combos = N × Q.
    For example: 2 charts × 100 questions = 200 unique combos.
    When count > unique combos, we allow repeats (same question + chart)
    because GPT-4o generates different responses each time due to temperature.
    This is intentional — DPO benefits from multiple response variations.
    """
    # Build weighted pool
    weighted_pool = []
    for cat, q in QUESTION_BANK:
        w = CATEGORY_WEIGHTS.get(cat, 5)
        weighted_pool.extend([(cat, q)] * w)

    max_unique = len(QUESTION_BANK) * len(CHART_TEMPLATES)
    if count > max_unique:
        print(f"  ℹ️  {len(CHART_TEMPLATES)} charts × {len(QUESTION_BANK)} questions = {max_unique} unique combos.")
        print(f"     Requested {count} pairs → will include repeats (different GPT responses each time).")
        print(f"     Add more kundali_*.json files to the project root for more diversity.")

    # Phase 1: fill unique combos first
    combos = []
    seen = set()
    attempts = 0
    max_attempts = max_unique * 3

    while len(combos) < min(count, max_unique) and attempts < max_attempts:
        attempts += 1
        cat, question = random.choice(weighted_pool)
        chart = random.choice(CHART_TEMPLATES)

        key = (question, chart["name"])
        if key in seen:
            continue
        seen.add(key)

        combos.append({
            "category": cat,
            "question": question,
            "chart": chart,
        })

    # Phase 2: if we need more, allow repeats (GPT temperature gives variation)
    while len(combos) < count:
        cat, question = random.choice(weighted_pool)
        chart = random.choice(CHART_TEMPLATES)
        combos.append({"category": cat, "question": question, "chart": chart})

    random.shuffle(combos)
    return combos


def create_batch_file(combos: list, chunk_idx: int = 0, idx_offset: int = 0) -> Path:
    """Create JSONL batch file with chosen + rejected requests for each combo."""
    suffix = f"_chunk{chunk_idx}" if chunk_idx > 0 else ""
    batch_path = output_dir / f"batch_input{suffix}.jsonl"
    request_count = 0

    with open(batch_path, "w", encoding="utf-8") as f:
        for i, combo in enumerate(combos):
            chart = combo["chart"]
            question = combo["question"]
            category = combo["category"]
            user_msg = _build_user_msg(chart, question, category)

            # ── Chosen request ──
            global_idx = i + idx_offset
            chosen_req = {
                "custom_id": f"chosen_{global_idx:05d}_{category}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": CHOSEN_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.85,
                    "max_tokens": 250,
                }
            }
            f.write(json.dumps(chosen_req, ensure_ascii=False) + "\n")
            request_count += 1

            # ── Rejected request ──
            rejected_req = {
                "custom_id": f"rejected_{global_idx:05d}_{category}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": REJECTED_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.9,
                    "max_tokens": 300,
                }
            }
            f.write(json.dumps(rejected_req, ensure_ascii=False) + "\n")
            request_count += 1

    print(f"✓ Batch file created: {batch_path}")
    print(f"  Total requests: {request_count} ({len(combos)} chosen + {len(combos)} rejected)")
    return batch_path


def submit_batch(batch_path: Path) -> str:
    """Upload JSONL and create batch job."""
    print("\nUploading batch file to OpenAI...")
    batch_file = client.files.create(
        file=open(batch_path, "rb"),
        purpose="batch"
    )
    print(f"  ✓ File uploaded: {batch_file.id}")

    print("Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "KP Astrology DPO preference pairs"}
    )
    print(f"  ✓ Batch job created: {batch_job.id}")
    print(f"  Status: {batch_job.status}")

    return batch_job.id


def check_batch(batch_id: str):
    """Check status of a batch job."""
    batch = client.batches.retrieve(batch_id)
    print(f"\nBatch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created: {batch.created_at}")
    if hasattr(batch, 'request_counts') and batch.request_counts:
        rc = batch.request_counts
        print(f"Requests — Total: {rc.total}, Completed: {rc.completed}, Failed: {rc.failed}")
    if batch.status == "completed":
        print(f"\n✅ Batch completed! Download with:")
        print(f"  python scripts/13_generate_dpo_dataset.py --download {batch_id}")
    elif batch.status == "failed":
        print(f"\n❌ Batch failed!")
        if hasattr(batch, 'errors') and batch.errors:
            for err in batch.errors.data[:5]:
                print(f"  Error: {err.message}")
    return batch.status


def download_and_pair(batch_id: str):
    """Download batch results and pair chosen+rejected into DPO JSONL."""
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"❌ Batch not completed yet. Status: {batch.status}")
        return

    print(f"\nDownloading results from batch {batch_id}...")
    output_file_id = batch.output_file_id
    if not output_file_id:
        print("❌ No output file found")
        return

    result_content = client.files.content(output_file_id).text
    results_path = output_dir / "batch_output.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(result_content)
    print(f"  ✓ Raw results saved: {results_path}")

    # Parse results into chosen/rejected maps
    chosen_map = {}  # idx -> response text
    rejected_map = {}  # idx -> response text
    errors = 0

    for line in result_content.strip().split("\n"):
        if not line.strip():
            continue
        result = json.loads(line)
        custom_id = result.get("custom_id", "")
        response = result.get("response", {})

        if response.get("status_code") != 200:
            errors += 1
            continue

        body = response.get("body", {})
        choices = body.get("choices", [])
        if not choices:
            errors += 1
            continue

        text = choices[0].get("message", {}).get("content", "").strip()
        if not text:
            errors += 1
            continue

        # Parse custom_id: "chosen_00042_marriage" or "rejected_00042_marriage"
        parts = custom_id.split("_", 2)
        if len(parts) < 3:
            errors += 1
            continue

        role = parts[0]  # "chosen" or "rejected"
        idx = parts[1]   # "00042"
        category = parts[2]  # "marriage"

        if role == "chosen":
            chosen_map[idx] = {"text": text, "category": category}
        elif role == "rejected":
            rejected_map[idx] = {"text": text, "category": category}

    print(f"  Parsed: {len(chosen_map)} chosen, {len(rejected_map)} rejected, {errors} errors")

    # Pair them up
    # Reload combos to get the original question + chart context
    combos_path = output_dir / "combos.json"
    if combos_path.exists():
        with open(combos_path, "r", encoding="utf-8") as f:
            combos = json.loads(f.read())
    else:
        combos = None

    dpo_pairs = []
    for idx in sorted(chosen_map.keys()):
        if idx not in rejected_map:
            continue

        chosen_text = chosen_map[idx]["text"]
        rejected_text = rejected_map[idx]["text"]
        category = chosen_map[idx]["category"]

        # ── Quality filters (industry-standard DPO pair validation) ──
        # F1: Too short
        if len(chosen_text) < 20 or len(rejected_text) < 20:
            continue
        # F2: Identical
        if chosen_text == rejected_text:
            continue
        # F3: Chosen has robotic patterns (wrong label)
        robotic = ["**", "Analysis:", "Conclusion:", "Confidence:", "Critical Finding:"]
        if any(r in chosen_text for r in robotic):
            continue
        # F4: Chosen has paragraph breaks (format violation)
        if "\n" in chosen_text:
            continue
        # F5: Chosen has ISO dates
        if re.search(r"\d{4}-\d{2}(?![\d-])", chosen_text) and not re.search(r"\d{4}-\d{4}", chosen_text):
            continue
        # F6: Chosen uses "the native" instead of name
        if "the native" in chosen_text.lower():
            continue
        # F7: Chosen >4 sentences (verbosity)
        sent_count = len([s for s in re.split(r"[.!?]+", chosen_text) if s.strip()])
        if sent_count > 4:
            continue
        # F8: Product spam in non-remedy chosen
        product_words = ["rudraksha", "bracelet", "pendant", "kavach", "mala",
                         "try karein", "hamara", "wear our", "package"]
        if category != "remedies" and any(w in chosen_text.lower() for w in product_words):
            continue

        pair = {
            "chosen": chosen_text,
            "rejected": rejected_text,
            "category": category,
        }

        # Add prompt from combos if available
        idx_int = int(idx)
        if combos and idx_int < len(combos):
            pair["prompt"] = combos[idx_int]["question"]
            pair["chart_yaml"] = combos[idx_int].get("chart_yaml", "")
            pair["chart_name"] = combos[idx_int].get("chart_name", "")

        dpo_pairs.append(pair)

    # Save DPO pairs
    dpo_path = output_dir / "dpo_pairs.jsonl"
    with open(dpo_path, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n✓ DPO pairs saved: {dpo_path}")
    print(f"  Total valid pairs: {len(dpo_pairs)}")

    # Category distribution
    cat_counts = {}
    for p in dpo_pairs:
        cat = p.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"\nCategory distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print(f"\nNext step: Prepare dataset for training")
    print(f"  python scripts/14_prepare_dpo_dataset.py")


# ═══════════════════════════════════════════════════════════════════════════════
# SYNC MODE — For testing or when you need results immediately
# ═══════════════════════════════════════════════════════════════════════════════

def run_sync(combos: list):
    """Production-grade DPO generation with multi-candidate + GPT-4o-as-judge scoring.

    Architecture (based on Tulu 3, Anyscale, Philschmid research):
      1. For each question: generate N chosen candidates at varying temperatures
      2. Score each candidate with 8-criteria rubric via GPT-4o-as-judge
      3. Pick the BEST chosen candidate (highest total score)
      4. Generate 1 rejected candidate, score it
      5. Only keep pairs with sufficient quality margin (chosen - rejected ≥ min_margin)

    This produces dramatically higher quality DPO data than simple 1-chosen/1-rejected.
    """
    import concurrent.futures

    dpo_path = output_dir / "dpo_pairs.jsonl"
    scores_path = output_dir / "quality_scores.jsonl"  # audit trail
    completed = 0
    failed = 0
    filtered_low_margin = 0
    total_chosen_scores = []
    total_rejected_scores = []

    n_candidates = args.candidates
    use_judge = args.judge
    min_margin = args.min_margin

    # Varying temperatures for candidate diversity
    chosen_temps = [0.7 + (i * 0.1) for i in range(n_candidates)]  # e.g. [0.7, 0.8, 0.9]

    def process_one(i_combo):
        i, combo = i_combo
        chart = combo["chart"]
        question = combo["question"]
        category = combo["category"]
        user_msg = _build_user_msg(chart, question, category)

        try:
            # ── Pass 1: Generate N chosen candidates at varying temperatures ──
            chosen_candidates = []
            for temp in chosen_temps[:n_candidates]:
                candidate = _generate_candidate(
                    CHOSEN_SYSTEM_PROMPT, user_msg,
                    temperature=temp, max_tokens=600
                )
                if candidate and len(candidate) >= 20:
                    chosen_candidates.append(candidate)

            if not chosen_candidates:
                return {"status": "no_chosen_candidates"}

            # ── Pass 2: Score chosen candidates with GPT-4o judge ──
            if use_judge and len(chosen_candidates) > 1:
                scored = []
                for cand in chosen_candidates:
                    score = _score_response(cand, question, chart.get("yaml", ""), category)
                    scored.append((cand, score))
                # Pick best by total score
                scored.sort(key=lambda x: x[1].get("total", 0), reverse=True)
                chosen = scored[0][0]
                chosen_score = scored[0][1]
            elif use_judge:
                chosen = chosen_candidates[0]
                chosen_score = _score_response(chosen, question, chart.get("yaml", ""), category)
            else:
                # No judge — pick first candidate, use rule-based filter
                chosen = chosen_candidates[0]
                chosen_score = {"total": 25}

            # ── Rule-based pre-filter on chosen (fast rejection) ──
            robotic = ["**", "Analysis:", "Conclusion:", "Confidence:", "Critical Finding:",
                       "## ", "### ", "1.", "2.", "3."]
            if any(r in chosen for r in robotic):
                return {"status": "chosen_robotic"}

            # ── Pass 3: Generate rejected candidate ──
            rejected = _generate_candidate(
                REJECTED_SYSTEM_PROMPT, user_msg,
                temperature=0.9, max_tokens=800
            )
            if not rejected or len(rejected) < 20:
                return {"status": "no_rejected"}

            if chosen == rejected:
                return {"status": "identical_responses"}

            # ── Score rejected ──
            if use_judge:
                rejected_score = _score_response(rejected, question, chart.get("yaml", ""), category)
            else:
                rejected_score = {"total": 10}

            # ── Pass 4: Margin check — only keep clearly separated pairs ──
            c_total = chosen_score.get("total", 0)
            r_total = rejected_score.get("total", 0)
            margin = c_total - r_total

            if margin < min_margin:
                return {
                    "status": "low_margin",
                    "margin": margin,
                    "chosen_score": c_total,
                    "rejected_score": r_total,
                }

            return {
                "status": "success",
                "prompt": question,
                "chart_name": chart["name"],
                "chart_yaml_preview": chart["yaml"][:300],
                "chosen": chosen,
                "rejected": rejected,
                "category": category,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "margin": margin,
                "n_candidates_generated": len(chosen_candidates),
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "index": i}

    judge_label = f"+ GPT-4o judge ({n_candidates} candidates/question)" if use_judge else "(no judge)"
    print(f"\n{'='*70}")
    print(f"PRODUCTION DPO GENERATION — {len(combos)} pairs")
    print(f"Mode: sync ({args.sync_workers} workers) {judge_label}")
    print(f"Min margin: {min_margin}/50 | Model: {args.model}")
    print(f"{'='*70}")

    with open(dpo_path, "w", encoding="utf-8") as f, \
         open(scores_path, "w", encoding="utf-8") as sf:

        # Process sequentially for quality (parallelism causes rate limits with judge)
        workers = 2 if use_judge else args.sync_workers

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = list(executor.map(process_one, enumerate(combos)))

            for i, result in enumerate(futures):
                if not result:
                    failed += 1
                    continue

                status = result.get("status", "unknown")

                if status == "success":
                    # Write DPO pair
                    dpo_pair = {
                        "prompt": result["prompt"],
                        "chosen": result["chosen"],
                        "rejected": result["rejected"],
                        "category": result["category"],
                        "chart_name": result["chart_name"],
                    }
                    f.write(json.dumps(dpo_pair, ensure_ascii=False) + "\n")
                    f.flush()

                    # Write audit trail
                    audit = {
                        "index": i,
                        "category": result["category"],
                        "chart_name": result["chart_name"],
                        "chosen_score": result["chosen_score"],
                        "rejected_score": result["rejected_score"],
                        "margin": result["margin"],
                        "n_candidates": result["n_candidates_generated"],
                    }
                    sf.write(json.dumps(audit, ensure_ascii=False) + "\n")
                    sf.flush()

                    total_chosen_scores.append(result["chosen_score"].get("total", 0))
                    total_rejected_scores.append(result["rejected_score"].get("total", 0))
                    completed += 1

                elif status == "low_margin":
                    filtered_low_margin += 1
                    audit = {"index": i, "status": "low_margin",
                             "margin": result.get("margin"),
                             "chosen_score": result.get("chosen_score"),
                             "rejected_score": result.get("rejected_score")}
                    sf.write(json.dumps(audit, ensure_ascii=False) + "\n")
                else:
                    failed += 1

                total = completed + failed + filtered_low_margin
                if total % 25 == 0 or total == len(combos):
                    avg_c = sum(total_chosen_scores[-25:]) / max(len(total_chosen_scores[-25:]), 1)
                    avg_r = sum(total_rejected_scores[-25:]) / max(len(total_rejected_scores[-25:]), 1)
                    print(f"  [{total:>4}/{len(combos)}] ✓{completed} ✗{failed} "
                          f"⚡{filtered_low_margin} low-margin | "
                          f"avg chosen={avg_c:.1f} rejected={avg_r:.1f} margin={avg_c-avg_r:.1f}")

    # ── Final quality report ──
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Valid pairs:     {completed}")
    print(f"  Failed:          {failed}")
    print(f"  Low-margin filtered: {filtered_low_margin}")
    if total_chosen_scores:
        avg_c = sum(total_chosen_scores) / len(total_chosen_scores)
        avg_r = sum(total_rejected_scores) / len(total_rejected_scores)
        print(f"\n  Avg chosen score:   {avg_c:.1f}/50")
        print(f"  Avg rejected score: {avg_r:.1f}/50")
        print(f"  Avg margin:         {avg_c - avg_r:.1f}")
        print(f"  Min margin filter:  {min_margin}")
    print(f"\n  Output: {dpo_path}")
    print(f"  Scores: {scores_path}")

    # Category distribution
    cat_counts = {}
    try:
        with open(dpo_path, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                cat = p.get("category", "unknown")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
    except Exception:
        pass
    if cat_counts:
        print(f"\n  Category distribution:")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def download_all_and_merge():
    """Download results from ALL batch chunks and merge into single DPO JSONL."""
    meta_path = output_dir / "batch_meta.json"
    if not meta_path.exists():
        print("❌ No batch_meta.json found. Run generation first.")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    chunks = meta.get("chunks", [])
    if not chunks:
        # Legacy single-batch format
        batch_id = meta.get("batch_id")
        if batch_id:
            print("Single batch detected (legacy format)")
            download_and_pair(batch_id)
            return
        print("❌ No chunk info found in batch_meta.json")
        return

    # Load combos for pairing
    combos_path = output_dir / "combos.json"
    if combos_path.exists():
        with open(combos_path, "r", encoding="utf-8") as f:
            combos = json.loads(f.read())
    else:
        combos = None

    all_chosen = {}
    all_rejected = {}
    total_errors = 0

    for chunk_info in chunks:
        batch_id = chunk_info["batch_id"]
        chunk_idx = chunk_info["chunk_idx"]

        print(f"\n── Chunk {chunk_idx} (batch: {batch_id}) ──")
        batch = client.batches.retrieve(batch_id)
        print(f"  Status: {batch.status}")

        if batch.status != "completed":
            print(f"  ⚠️ Skipping — not completed yet")
            continue

        output_file_id = batch.output_file_id
        if not output_file_id:
            print(f"  ⚠️ No output file")
            continue

        result_content = client.files.content(output_file_id).text

        # Save raw chunk output
        chunk_path = output_dir / f"batch_output_chunk{chunk_idx}.jsonl"
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(result_content)
        print(f"  ✓ Raw results saved: {chunk_path}")

        # Parse results
        errors = 0
        for line in result_content.strip().split("\n"):
            if not line.strip():
                continue
            result = json.loads(line)
            custom_id = result.get("custom_id", "")
            response = result.get("response", {})

            if response.get("status_code") != 200:
                errors += 1
                continue

            body = response.get("body", {})
            choices = body.get("choices", [])
            if not choices:
                errors += 1
                continue

            text = choices[0].get("message", {}).get("content", "").strip()
            if not text:
                errors += 1
                continue

            parts = custom_id.split("_", 2)
            if len(parts) < 3:
                errors += 1
                continue

            role, idx, category = parts[0], parts[1], parts[2]
            if role == "chosen":
                all_chosen[idx] = {"text": text, "category": category}
            elif role == "rejected":
                all_rejected[idx] = {"text": text, "category": category}

        total_errors += errors
        print(f"  Parsed: {sum(1 for k in all_chosen if k.startswith(str(chunk_info.get('idx_offset',0)//1000)))} new, {errors} errors")

    print(f"\n── Merging all chunks ──")
    print(f"  Total chosen: {len(all_chosen)}, rejected: {len(all_rejected)}, errors: {total_errors}")

    # Pair chosen + rejected
    dpo_pairs = []
    for idx in sorted(all_chosen.keys()):
        if idx not in all_rejected:
            continue

        chosen_text = all_chosen[idx]["text"]
        rejected_text = all_rejected[idx]["text"]
        category = all_chosen[idx]["category"]

        if len(chosen_text) < 20 or len(rejected_text) < 20:
            continue
        if chosen_text == rejected_text:
            continue

        robotic = ["**", "Analysis:", "Conclusion:", "Confidence:", "Critical Finding:"]
        if any(r in chosen_text for r in robotic):
            continue

        pair = {
            "chosen": chosen_text,
            "rejected": rejected_text,
            "category": category,
        }

        idx_int = int(idx)
        if combos and idx_int < len(combos):
            pair["prompt"] = combos[idx_int]["question"]
            pair["chart_yaml"] = combos[idx_int].get("chart_yaml", "")
            pair["chart_name"] = combos[idx_int].get("chart_name", "")

        dpo_pairs.append(pair)

    # Save merged DPO pairs
    dpo_path = output_dir / "dpo_pairs.jsonl"
    with open(dpo_path, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n✓ Merged DPO pairs saved: {dpo_path}")
    print(f"  Total valid pairs: {len(dpo_pairs)}")

    cat_counts = {}
    for p in dpo_pairs:
        cat = p.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"\nCategory distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print(f"\nNext step: Prepare dataset for training")
    print(f"  python scripts/14_prepare_dpo_dataset.py")


def main():
    print("=" * 80)
    print("DPO PREFERENCE DATASET GENERATOR — OpenAI Batch API")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Check mode ──
    if args.check:
        check_batch(args.check)
        return

    # ── Download single batch ──
    if args.download:
        download_and_pair(args.download)
        return

    # ── Download ALL chunks and merge ──
    if args.download_all:
        download_all_and_merge()
        return

    # ── Generate mode ──
    print(f"Model: {args.model}")
    print(f"Target pairs: {args.count}")
    print(f"Chunk size: {args.chunk_size} pairs per batch")
    print(f"Mode: {'sync' if args.sync else 'batch (50% cheaper)'}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # 1. Generate combinations
    print(f"\n1. Generating {args.count} question-chart combinations...")
    combos = generate_combinations(args.count)

    # Save combos for later pairing
    combos_save = []
    for c in combos:
        combos_save.append({
            "question": c["question"],
            "category": c["category"],
            "chart_name": c["chart"]["name"],
            "chart_yaml": c["chart"]["yaml"],  # full YAML for training context
        })
    combos_path = output_dir / "combos.json"
    with open(combos_path, "w", encoding="utf-8") as f:
        json.dump(combos_save, f, ensure_ascii=False, indent=1)
    print(f"  ✓ Combos saved: {combos_path}")

    # Category distribution
    cat_counts = {}
    for c in combos:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
    print(f"  Category distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    if args.sync:
        # Sync mode — instant but 2x cost
        run_sync(combos)
        print(f"\nNext step: Prepare dataset for training")
        print(f"  python scripts/14_prepare_dpo_dataset.py")
    else:
        # Batch mode — chunked to stay under token enqueue limit
        chunk_size = args.chunk_size
        num_chunks = (len(combos) + chunk_size - 1) // chunk_size
        print(f"\n2. Splitting {len(combos)} pairs into {num_chunks} chunks of ~{chunk_size}...")

        batch_ids = []
        chunk_meta = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(combos))
            chunk_combos = combos[start:end]

            print(f"\n── Chunk {chunk_idx + 1}/{num_chunks}: pairs {start}-{end-1} ({len(chunk_combos)} pairs) ──")

            batch_path = create_batch_file(chunk_combos, chunk_idx=chunk_idx, idx_offset=start)

            print(f"  Submitting chunk {chunk_idx + 1}...")
            batch_id = submit_batch(batch_path)
            batch_ids.append(batch_id)

            chunk_meta.append({
                "chunk_idx": chunk_idx,
                "batch_id": batch_id,
                "idx_offset": start,
                "num_pairs": len(chunk_combos),
            })

            # Small delay between submissions to avoid rate limits
            if chunk_idx < num_chunks - 1:
                print("  Waiting 3s before next chunk...")
                time.sleep(3)

        # Save all batch metadata
        meta_path = output_dir / "batch_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "created_at": datetime.now().isoformat(),
                "model": args.model,
                "total_pairs": len(combos),
                "chunk_size": chunk_size,
                "num_chunks": num_chunks,
                "chunks": chunk_meta,
            }, f, indent=2)

        print(f"\n{'=' * 80}")
        print(f"ALL {num_chunks} BATCH CHUNKS SUBMITTED SUCCESSFULLY")
        print(f"{'=' * 80}")
        for cm in chunk_meta:
            print(f"  Chunk {cm['chunk_idx']}: {cm['batch_id']} ({cm['num_pairs']} pairs)")
        print(f"\nBatches will complete within 24 hours (usually much faster).")
        print(f"\nNext steps:")
        for cm in chunk_meta:
            print(f"  Check chunk {cm['chunk_idx']}:  python scripts/13_generate_dpo_dataset.py --check {cm['batch_id']}")
        print(f"\n  Download ALL & merge:")
        print(f"    python scripts/13_generate_dpo_dataset.py --download-all")
        print(f"\n  Then prepare data:")
        print(f"    python scripts/14_prepare_dpo_dataset.py")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
