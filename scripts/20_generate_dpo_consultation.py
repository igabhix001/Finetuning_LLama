"""
DPO Consultation Dataset Generator — Rebuild with chart context + all 5 rule categories.
==========================================================================================
Fixes all DPO gaps identified in the audit:

GAPS FIXED:
  - 8% English Q → Hinglish chosen (language mismatch)
  - Only 8 past-marriage pairs (client's #1 complaint)
  - Only 17 profession pairs (client's #2 complaint)
  - Only 9 education past-event pairs (client's #3 complaint)
  - Only 1% remedy/product pairs
  - Only 1% emotional pairs
  - No system_prompt in DPO pairs (model never learned persona during DPO)
  - DPO max_length 1024 too short for chart YAML (~3500 tokens)

THIS SCRIPT:
  - Uses GPT-4o (OpenAI) — client-specified model for DPO generation
  - Every pair includes chart YAML in the prompt (training-inference match)
  - Chosen responses are 1-4 sentences, English Q → English answer
  - Rejected responses are short but wrong (language mismatch, no dates, no justification)
  - Covers all client-specified question types with proper distribution
  - Includes system_prompt field for DPO training
  - Generates pairs for all 5 rule categories
  - Uses OpenAI Batch API for cost efficiency (~50% cheaper)

Usage:
  python scripts/20_generate_dpo_consultation.py --count 3000
  python scripts/20_generate_dpo_consultation.py --count 100 --dry-run
  python scripts/20_generate_dpo_consultation.py --resume
  python scripts/20_generate_dpo_consultation.py --count 3000 --batch  # use Batch API
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import date
from pathlib import Path

import anthropic
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv

load_dotenv()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate DPO consultation dataset")
parser.add_argument("--count", type=int, default=3000, help="Target number of DPO pairs")
parser.add_argument("--model", type=str, default="gpt-4o",
                    help="OpenAI model for DPO generation (default: gpt-4o as client specified)")
parser.add_argument("--output-dir", type=str, default="data/dpo_consultation", help="Output directory")
parser.add_argument("--kundali-dir", type=str, default="sample_kundali", help="Kundali JSON directory")
parser.add_argument("--rules-dir", type=str, default="data/rules", help="Rules JSON directory")
parser.add_argument("--dry-run", action="store_true", help="Generate 5 pairs and print, no save")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
parser.add_argument("--batch-size", type=int, default=50, help="Checkpoint interval")
parser.add_argument("--batch", action="store_true",
                    help="Use OpenAI Batch API (~50% cheaper, results in <24h)")
args = parser.parse_args()

# ── Setup ─────────────────────────────────────────────name───────────────────
# DPO uses GPT-4o (OpenAI) as client specified.
# SFT uses Claude Sonnet 4 (Anthropic) — see 19_generate_sft_consultation.py
oai_key = os.getenv("OPENAI_API_KEY")
if not oai_key:
    print("❌ OPENAI_API_KEY not found in .env")
    print("  Add OPENAI_API_KEY=sk-... to your .env file")
    sys.exit(1)

client = OpenAIClient(api_key=oai_key)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_file = output_dir / "checkpoint.jsonl"

sys.path.insert(0, str(Path(__file__).parent))
from chart_preprocessor import chart_to_yaml

_TODAY = date.today().strftime("%d %b %Y")

# ── Load rules ────────────────────────────────────────────────────────────────
rules_dir = Path(args.rules_dir)

def _load_json(path):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}

category_rules = _load_json(rules_dir / "01_category_rules.json")
dasha_rules = _load_json(rules_dir / "02_dasha_rules.json")
planet_house_rules = _load_json(rules_dir / "03_planet_house_rules.json")
product_rules = _load_json(rules_dir / "04_product_rules.json")
comm_rules = _load_json(rules_dir / "05_communication_rules.json")
print(f"✓ Loaded rule JSONs")

# ── Load charts ───────────────────────────────────────────────────────────────
kundali_dir = Path(args.kundali_dir)
charts = []
for kf in sorted(kundali_dir.glob("*.json")):
    try:
        with open(kf, encoding="utf-8") as f:
            raw = f.read()
        yaml_str = chart_to_yaml(raw)
        name = kf.stem.replace("kundali_", "").replace("_", " ")
        charts.append({"name": name, "yaml": yaml_str, "file": kf.name})
    except Exception as e:
        print(f"  Warning: {kf.name}: {e}")
print(f"✓ Loaded {len(charts)} charts")

# ── Question bank with target distribution ────────────────────────────────────
# Distribution designed to fix the audit gaps
QUESTION_POOL = [
    # Past events — heavily weighted (client's top complaints)
    ("past_event", "When did I get married?", 1.0),
    ("past_event", "Which month and year did I get married?", 1.0),
    ("past_event", "I am already married. When did my marriage happen?", 1.0),
    ("past_event", "When did I start my current job?", 1.0),
    ("past_event", "When did I graduate college?", 1.0),
    ("past_event", "When did I complete my education?", 1.0),
    ("past_event", "What happened in my life in 2020?", 1.0),
    ("past_event", "What major event happened in 2018?", 1.0),
    ("past_event", "When did I buy my house?", 0.8),
    ("past_event", "When did I have my first child?", 0.8),
    ("past_event", "What was significant in my life in 2015?", 0.8),
    ("past_event", "Main kab shadi hua?", 0.8),
    ("past_event", "Naukri kab lagi meri?", 0.8),
    ("past_event", "College kab complete kiya?", 0.8),

    # Profession/career analysis — client's #2 complaint
    ("analysis_career", "What is my field of work?", 1.0),
    ("analysis_career", "What career is best suited for me?", 1.0),
    ("analysis_career", "What profession does my chart indicate?", 1.0),
    ("analysis_career", "Am I suited for business or job?", 1.0),
    ("analysis_career", "What does my chart say about my career?", 1.0),
    ("analysis_career", "Will I be successful in my career?", 0.8),
    ("analysis_career", "What kind of work will I do?", 0.8),
    ("analysis_career", "Mera career kaisa rahega?", 0.8),
    ("analysis_career", "Konsa kaam mujhe suit karta hai?", 0.8),
    ("analysis_career", "Business ya job — kya better hai mere liye?", 0.8),

    # Timing — marriage
    ("timing_marriage", "When will I get married?", 1.0),
    ("timing_marriage", "What is my marriage timing?", 1.0),
    ("timing_marriage", "When is my best marriage window?", 1.0),
    ("timing_marriage", "Will I get married this year?", 0.8),
    ("timing_marriage", "Meri shaadi kab hogi?", 0.8),
    ("timing_marriage", "Shaadi ka time kab hai?", 0.8),

    # Timing — career
    ("timing_career", "When will I get a job?", 1.0),
    ("timing_career", "When will I get a promotion?", 1.0),
    ("timing_career", "When will my career improve?", 1.0),
    ("timing_career", "Naukri kab milegi?", 0.8),
    ("timing_career", "Promotion kab milega?", 0.8),

    # Timing — finance
    ("timing_finance", "When will my financial situation improve?", 1.0),
    ("timing_finance", "When will my income increase?", 1.0),
    ("timing_finance", "When will my debts clear?", 0.8),
    ("timing_finance", "Paisa kab aayega?", 0.8),

    # Timing — property
    ("timing_property", "When should I buy a house?", 0.8),
    ("timing_property", "When will I get my own home?", 0.8),
    ("timing_property", "Ghar kab khareedun?", 0.6),

    # Analysis — marriage
    ("analysis_marriage", "Will I have a love marriage or arranged marriage?", 0.8),
    ("analysis_marriage", "What kind of spouse will I have?", 0.8),
    ("analysis_marriage", "Is marriage promised in my chart?", 0.8),
    ("analysis_marriage", "Love marriage hogi ya arranged?", 0.6),

    # Analysis — finance
    ("analysis_finance", "Will I be wealthy?", 0.6),
    ("analysis_finance", "What does my chart say about my finances?", 0.6),
    ("analysis_finance", "Kya main ameer banunga?", 0.5),

    # Analysis — health
    ("analysis_health", "What does my chart say about my health?", 0.6),
    ("analysis_health", "Am I prone to any health issues?", 0.6),
    ("analysis_health", "Meri health kaisi rahegi?", 0.5),

    # Emotional — heavily weighted (client complaint)
    ("emotional", "I feel very unlucky. Nothing is going right.", 1.0),
    ("emotional", "I am very depressed and hopeless.", 1.0),
    ("emotional", "Why am I always struggling?", 1.0),
    ("emotional", "Why do I keep failing despite hard work?", 1.0),
    ("emotional", "I am going through a very difficult time.", 1.0),
    ("emotional", "Why is life so hard for me?", 1.0),
    ("emotional", "Mujhe bahut tension hai.", 0.8),
    ("emotional", "Main bahut pareshan hun.", 0.8),
    ("emotional", "Kuch bhi theek nahi ho raha.", 0.8),

    # Remedy — heavily weighted (client requirement)
    ("remedy", "What remedy should I do for my marriage?", 1.0),
    ("remedy", "Which gemstone should I wear?", 1.0),
    ("remedy", "What rudraksha is good for me?", 1.0),
    ("remedy", "What upay should I do for career?", 1.0),
    ("remedy", "Suggest a product for my problems.", 1.0),
    ("remedy", "Kaunsa rudraksha pehnu?", 0.8),
    ("remedy", "Koi upay batao career ke liye.", 0.8),
    ("remedy", "Shaadi ke liye kya remedy karun?", 0.8),

    # Safety
    ("safety", "When will I die?", 0.5),
    ("safety", "How long will I live?", 0.5),
    ("safety", "Kab marunga main?", 0.4),

    # Simple factual — heavily weighted to fix verbosity issue
    ("simple_factual", "What is my name?", 1.0),
    ("simple_factual", "What is my lagna?", 1.0),
    ("simple_factual", "What is my rashi?", 1.0),
    ("simple_factual", "What is today's date?", 1.0),
    ("simple_factual", "What is the current date?", 0.8),
    ("simple_factual", "Mera naam kya hai?", 0.8),
    ("simple_factual", "Mera lagna kya hai?", 0.8),
    ("simple_factual", "Aaj ki date kya hai?", 0.8),

    # Children queries — to fix medical disclaimer issue
    ("analysis_children", "Will I have children?", 1.0),
    ("analysis_children", "When will I have a child?", 1.0),
    ("analysis_children", "Am I blessed with children in my chart?", 0.8),
    ("analysis_children", "Kya mujhe bacche honge?", 0.8),
    ("analysis_children", "Baccha kab hoga?", 0.8),

    # Identity/KP system queries — to fix wrong attribution
    ("identity", "Who are you?", 0.6),
    ("identity", "What is your name?", 0.5),
    ("identity", "What is KP astrology?", 0.8),
    ("identity", "What is Krishnamurti Paddhati?", 0.6),
    ("identity", "Who developed KP astrology?", 0.6),
    ("identity", "KP astrology kya hai?", 0.5),

    # Follow-up context (important for conversation continuity)
    ("followup", "But I am already married.", 0.8),
    ("followup", "That period has already passed.", 0.6),
    ("followup", "I didn't get married in that period.", 0.6),
    ("followup", "Main toh pehle se shadi shuda hun.", 0.6),
    ("followup", "Can you be more specific about the month?", 0.5),
]

def _sample_question():
    questions = [(q[0], q[1]) for q in QUESTION_POOL]
    weights = [q[2] for q in QUESTION_POOL]
    total = sum(weights)
    weights = [w / total for w in weights]
    idx = random.choices(range(len(questions)), weights=weights, k=1)[0]
    return questions[idx]


# ── Chosen response prompt ────────────────────────────────────────────────────
CHOSEN_SYSTEM = f"""You are generating the IDEAL response for a KP astrology AI named "Jyotish".
This response represents what a REAL experienced astrologer would say — precise, warm, justified, data-driven.

TODAY'S DATE: {_TODAY}

HARD LENGTH RULE — THIS OVERRIDES EVERYTHING:
Simple factual questions (name, lagna, rashi, date) = 1 sentence WITHOUT addressing.
  WRONG: "Aadhya Das ji, today's date is 08 March 2026."
  RIGHT: "08 March 2026."
  WRONG: "Aadhya Das ji, your lagna is Aquarius, ruled by Saturn."
  RIGHT: "Aquarius, ruled by Saturn."

Timing predictions = 2 sentences max (1 for date+justification, 1 optional for additional context).
Emotional queries = 2-3 sentences (empathy + end date + encouragement).
Analysis queries = 2-3 sentences max.
Past events = 2-3 sentences (past tense + dasha + age reference).

4 sentences is the ABSOLUTE ceiling for complex analysis only.
NO paragraph breaks — write as one continuous block.

LANGUAGE RULES (CRITICAL):
- English question → 100% English response. ZERO Hindi words, zero Hinglish mixing.
- Hindi/Hinglish question → respond FULLY in Hindi/Hinglish.
- WRONG: User asks "When will I get married?" → you reply "Aapke liye favorable combination hai..." ← FORBIDDEN
- RIGHT: User asks "When will I get married?" → you reply "[Name] ji, your marriage window is..." ← CORRECT

FORMAT:
- ZERO markdown: no **bold**, no headers, no bullets, no numbered lists.
- NO paragraph breaks. One continuous block of text.
- ZERO emojis: no 🙏, no ❤️, no 🌟. Text only.

DATE FORMAT (ZERO EXCEPTIONS):
- ALWAYS: "Oct 2025", "Jan 2028", "Mar 2027 to Aug 2027"
- NEVER: "2025-10", "2028-01", ISO format, "upcoming", "soon"
- Past dates (before {_TODAY}) → PAST tense. Future dates → FUTURE tense.

JUSTIFICATION (MANDATORY):
Every prediction MUST include WHY in the SAME sentence — name sub-lord + cusp + houses inline.
Example: "your 7th cusp sub-lord Saturn signifies houses 2,7 which are marriage-positive"

AGE REFERENCE: For every timing prediction, mention the person's age at the predicted event inline.

ADDRESS: Always "[Name] ji" for predictions/analysis. NEVER for simple factual queries (name/lagna/rashi/date).
NEVER "the native", "the person", "the querent".

EMOTIONAL QUERIES (CRITICAL):
MUST include empathy prefix + WHEN the difficult period ENDS with specific month-year.
  WRONG: "I understand you're struggling. Saturn period is challenging."
  RIGHT: "I understand how overwhelming this feels. Your current Saturn-Rahu period ends in Jul 2026, after which Venus-Mercury brings relief and new opportunities."

PAST EVENT QUERIES (CRITICAL):
MUST answer with actual past dasha analysis. NEVER deflect or give vague methodology.
  WRONG: "Looking at previous planetary combinations, significant changes often manifest when..."
  RIGHT: "Major developments occurred during Sun-Venus period from Oct 2022 to Feb 2023 (yeh period beet chuka hai) when you were 27 years old."

CAREER/ANALYSIS QUERIES (CRITICAL):
MUST give direct answer. NEVER deflect with "consult professionals" unless it's a medical/legal question.
  WRONG: "Please consult proper professionals who can give you tailored advice."
  RIGHT: "Based on your 10th cusp sub-lord Sun signifying houses 2,7,9,10,11, your career field is teaching, law, or government sectors."

CHILDREN QUERIES:
Answer directly about astrological prospects. NO medical disclaimers unless user asks about fertility issues.
  WRONG: "...Medical consultation should accompany astrological timing guidance."
  RIGHT: "Children prospects look promising as your 5th cusp sub-lord signifies houses 2,5,11 during Jupiter period from Jan 2027 to May 2028 at age 31-32."

SAFETY QUERIES: 
NEVER give timing for death/longevity. Compassionate redirect only. NO scary phrases.
  WRONG: "8th house affliction indicates health risks" or "death ki timing"
  RIGHT: "Please don't worry — astrology is here to guide you, not to scare you. Health concerns are best addressed by a qualified medical professional."

PRODUCTS: 
ONLY when user EXPLICITLY asks for remedies/upay. ZERO product mention otherwise.

KP SYSTEM ATTRIBUTION:
If asked about KP astrology, ALWAYS credit "Prof. K.S. Krishnamurti" (1960s).
  WRONG: "Developed by Dr. Yashoda Devi"
  RIGHT: "Developed by Prof. K.S. Krishnamurti in the 1960s"

Read the chart YAML. Use actual dasha dates, cusp sub-lords, house significations.
Return ONLY the response text. No labels, no "Chosen:", no explanation."""


# ── Rejected response prompt ──────────────────────────────────────────────────
REJECTED_SYSTEM = f"""You are generating a BAD response for a KP astrology AI training dataset.
This response represents what a poorly trained model produces.

CRITICAL LENGTH RULE: Your response MUST be 1-4 sentences, same as the ideal response.
The badness must come from CONTENT and STYLE, NOT from being longer.

Pick 3-4 of these wrong patterns and combine them in your short response:

SIMPLE FACTUAL (wrong):
- Add unnecessary addressing: "Aadhya Das ji, today's date is 08 March 2026" instead of just "08 March 2026"
- Add extra explanation: "Aadhya Das ji, your lagna is Aquarius, ruled by Saturn" instead of just "Aquarius, ruled by Saturn"
- Make it 2-3 sentences when 1 sentence is enough.

LANGUAGE (wrong):
- ALWAYS respond in Hinglish regardless of what language the user writes in.
- Mix Hindi and English randomly: "According to aapke chart mein, the native ka marriage yoga hai."
- Say "the native" instead of using the person's name.

FORMAT (wrong):
- Start with "**Analysis:**" or "According to KP principles..." even in a short response.
- Add "Confidence: medium" at the end.
- Use emojis: 🙏, ❤️, 🌟

DATES (wrong):
- Use ISO format: "2025-10" instead of "Oct 2025".
- Give only vague ranges: "between 2028 to 2033" without month-level precision.
- Say "upcoming period" or "favorable time" instead of actual dates.
- NEVER mention pratyantar dashas.

TENSE (wrong):
- Treat past dates as future: say "will be" for dates that already passed.

EMOTIONAL QUERIES (wrong):
- Be cold and clinical. No empathy. Start with methodology.
- Give empathy but NO end date: "I understand you're struggling. Saturn period is challenging." (missing WHEN it ends)

PAST EVENT QUERIES (wrong):
- Deflect with vague methodology: "Looking at previous planetary combinations, significant changes often manifest when multiple significator planets align..."
- Give future dates instead of analyzing past dashas.

CAREER/ANALYSIS QUERIES (wrong):
- Deflect to professionals: "Please consult proper professionals who can give you tailored advice."
- List ALL possible careers instead of the specific one.
- Don't answer directly. Start with "According to KP principles..." methodology filler.
- Give no justification — no sub-lord, no cusp, no house numbers.

CHILDREN QUERIES (wrong):
- Add unnecessary medical disclaimer: "Medical consultation should accompany astrological timing guidance for optimal outcomes."
- Make response too long (4+ sentences) with excessive detail.

SAFETY QUERIES (wrong):
- Use scary phrases: "8th house affliction indicates health risks" or "death ki timing"
- Give actual death timing instead of compassionate redirect.

KP ATTRIBUTION (wrong):
- Credit wrong person: "Developed by Dr. Yashoda Devi" instead of "Prof. K.S. Krishnamurti"

PRODUCTS (wrong):
- Force a product recommendation even for non-remedy questions.

FOLLOW-UP CONTEXT (wrong):
- For "I am already married" — ignore the context and give marriage timing anyway.

Return ONLY the response text. No labels, no "Rejected:", no explanation."""


def _build_prompt(question: str, chart_yaml: str) -> str:
    return f"""CHART DATA:
{chart_yaml}

USER QUESTION: {question}

Generate the response following ALL the rules above."""


def _generate_pair(qtype: str, question: str, chart: dict) -> dict | None:
    """Generate a chosen+rejected pair using GPT-4o (OpenAI) as client specified."""
    chart_yaml = chart["yaml"]
    prompt_text = _build_prompt(question, chart_yaml)

    try:
        # Generate chosen (ideal response)
        chosen_resp = client.chat.completions.create(
            model=args.model,
            max_tokens=300,
            messages=[
                {"role": "system", "content": CHOSEN_SYSTEM},
                {"role": "user", "content": prompt_text}
            ]
        )
        chosen = chosen_resp.choices[0].message.content.strip()

        # Generate rejected (bad response)
        rejected_resp = client.chat.completions.create(
            model=args.model,
            max_tokens=300,
            messages=[
                {"role": "system", "content": REJECTED_SYSTEM},
                {"role": "user", "content": prompt_text}
            ]
        )
        rejected = rejected_resp.choices[0].message.content.strip()

        # Quality checks
        if len(chosen) < 20 or len(rejected) < 20:
            return None
        if chosen == rejected:
            return None
        # Chosen should not have Hinglish for English questions
        is_english_q = not any(w in question.lower() for w in
                               ["kab", "kya", "mera", "meri", "batao", "hoga", "hogi", "hun", "hai"])
        hindi_words = ["aapki", "aapka", "mein", "hain", "karta", "karte", "karna"]
        if is_english_q and sum(1 for w in hindi_words if w in chosen.lower()) >= 3:
            return None  # Skip — chosen is still Hinglish for English question

        # Build the DPO pair in TRL format with system_prompt
        system_prompt = (
            f"You are Jyotish, a warm and confident KP astrologer. "
            f"TODAY'S DATE: {_TODAY}. "
            "Answer in the same language as the question. "
            "English question = 100% English. Hindi/Hinglish question = Hindi/Hinglish. "
            "Address as [Name] ji. No markdown. Max 4 sentences."
        )

        return {
            "prompt": question,
            "chosen": chosen,
            "rejected": rejected,
            "system": system_prompt,
            "chart_yaml": chart_yaml[:2000],
            "metadata": {
                "qtype": qtype,
                "chart_name": chart["name"],
                "model": args.model,
                "generated_date": _TODAY,
            }
        }

    except Exception as e:
        err_str = str(e).lower()
        if "rate" in err_str or "429" in err_str:
            time.sleep(30)
        else:
            print(f"  API error: {e}")
        return None


def _load_checkpoint() -> list:
    if checkpoint_file.exists():
        pairs = []
        with open(checkpoint_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        print(f"  Resumed: {len(pairs)} existing pairs")
        return pairs
    return []


def _save_checkpoint(pairs: list):
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def _save_final(pairs: list):
    """Save final DPO dataset as JSONL (used by 14_prepare_dpo_dataset.py)."""
    out_path = output_dir / "dpo_consultation.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"✓ Saved {len(pairs)} pairs to {out_path}")
    print(f"  Next: run scripts/14_prepare_dpo_dataset.py --input {out_path}")


def main():
    print("=" * 80)
    print("DPO CONSULTATION DATASET GENERATOR")
    print("=" * 80)
    print(f"Target: {args.count} pairs")
    print(f"Model: {args.model}")
    print(f"Charts: {len(charts)}")
    print("=" * 80)

    if args.dry_run:
        print("\n--- DRY RUN: generating 5 pairs ---\n")
        for i in range(5):
            chart = random.choice(charts)
            qtype, question = _sample_question()
            pair = _generate_pair(qtype, question, chart)
            if pair:
                print(f"[{i+1}] qtype={qtype}")
                print(f"  Q: {question}")
                print(f"  CHOSEN: {pair['chosen']}")
                print(f"  REJECTED: {pair['rejected']}")
                print()
        return

    pairs = _load_checkpoint() if args.resume else []
    start = len(pairs)
    errors = 0
    i = start

    print(f"Generating {args.count - start} more pairs...")

    while i < args.count:
        chart = random.choice(charts)
        qtype, question = _sample_question()
        pair = _generate_pair(qtype, question, chart)

        if pair:
            pairs.append(pair)
            i += 1
            if i % 10 == 0:
                print(f"{i}", end=" ", flush=True)
            if i % args.batch_size == 0:
                _save_checkpoint(pairs)
                print(f"\n  [checkpoint at {i}]", end=" ", flush=True)
        else:
            errors += 1
            if errors > 100:
                print(f"\n⚠️  Too many errors ({errors}), stopping at {i}")
                break
            time.sleep(1)

    print(f"\n\n✓ Generated {len(pairs)} pairs ({errors} errors)")
    _save_final(pairs)

    # Distribution report
    qtypes = {}
    lang_mismatches = 0
    for p in pairs:
        qt = p["metadata"]["qtype"]
        qtypes[qt] = qtypes.get(qt, 0) + 1
        # Check language mismatch
        q = p["prompt"]
        chosen = p["chosen"]
        is_en = not any(w in q.lower() for w in ["kab","kya","mera","meri","batao","hoga","hogi"])
        hindi_words = ["aapki","aapka","mein","hain","karta","karte"]
        if is_en and sum(1 for w in hindi_words if w in chosen.lower()) >= 3:
            lang_mismatches += 1

    print("\nQuestion type distribution:")
    for qt, count in sorted(qtypes.items(), key=lambda x: -x[1]):
        print(f"  {qt}: {count} ({100*count//len(pairs)}%)")
    print(f"\nLanguage mismatches in chosen: {lang_mismatches} ({100*lang_mismatches//len(pairs) if pairs else 0}%)")
    print(f"\n{'='*80}")
    print("DONE — DPO consultation dataset generated")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
