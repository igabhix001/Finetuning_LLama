"""
SFT Consultation Dataset Generator — Rebuild from scratch in consultation format.
===================================================================================
Fixes all gaps identified in the audit:

GAP FIXED: SFT was KP textbook Q&A (no chart, no "ji", bullets, too long, Hinglish)
THIS SCRIPT: Generates chart-grounded consultation pairs where:
  - Every example includes a real chart YAML in the system prompt
  - Responses are 1-4 sentences (consultation style, not encyclopedia)
  - Responses address the user as "[Name] ji"
  - Responses cite cusp sub-lord + house numbers
  - Language matches the question (English Q → English answer)
  - No bullets, no markdown, no "the native"
  - Covers all 7 query types: factual, timing, past_event, analysis,
    emotional, safety, remedy

Uses Anthropic Claude Sonnet 4 (claude-sonnet-4-20250514) — client-specified model for SFT.
Uses Anthropic Batch API for cost efficiency (~50% cheaper than real-time API).
Uses the SAME chart_to_yaml() preprocessor as inference (training-inference match).

Usage:
  python scripts/19_generate_sft_consultation.py --count 15000
  python scripts/19_generate_sft_consultation.py --count 500 --dry-run
  python scripts/19_generate_sft_consultation.py --resume  # continue from checkpoint
  python scripts/19_generate_sft_consultation.py --count 15000 --batch  # use Batch API (cheaper)
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
from dotenv import load_dotenv
from datasets import Dataset

load_dotenv()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate SFT consultation dataset")
parser.add_argument("--count", type=int, default=15000, help="Target number of examples")
parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                    help="Anthropic model (default: claude-sonnet-4-20250514 as client specified)")
parser.add_argument("--output-dir", type=str, default="data/sft_consultation", help="Output directory")
parser.add_argument("--kundali-dir", type=str, default="sample_kundali", help="Directory with kundali JSON files")
parser.add_argument("--rules-dir", type=str, default="data/rules", help="Directory with rule JSONs")
parser.add_argument("--dry-run", action="store_true", help="Generate 10 examples and print, no save")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
parser.add_argument("--batch-size", type=int, default=50, help="Save checkpoint every N examples")
parser.add_argument("--workers", type=int, default=5, help="Parallel API workers")
parser.add_argument("--batch", action="store_true",
                    help="Use Anthropic Batch API (~50% cheaper, results available in <24h)")
parser.add_argument("--batch-create", action="store_true",
                    help="Create and submit batch (use with --batch)")
parser.add_argument("--batch-check", type=str,
                    help="Check batch status by ID")
parser.add_argument("--batch-download", type=str,
                    help="Download batch results by ID and merge with existing")
parser.add_argument("--chunk-size-batch", type=int, default=5000,
                    help="Requests per batch chunk (max 100k, recommended 5k-10k)")
args = parser.parse_args()

# ── Setup ─────────────────────────────────────────────────────────────────────
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ANTHROPIC_API_KEY not found in .env")
    sys.exit(1)

client = anthropic.Anthropic(api_key=api_key)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_file = output_dir / "checkpoint.jsonl"

# ── Load kundali files ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from chart_preprocessor import chart_to_yaml

kundali_dir = Path(args.kundali_dir)
kundali_files = sorted(kundali_dir.glob("*.json"))
if not kundali_files:
    print(f"❌ No kundali JSON files found in {kundali_dir}")
    sys.exit(1)
print(f"✓ Found {len(kundali_files)} kundali files")

# ── Load rules ────────────────────────────────────────────────────────────────
rules_dir = Path(args.rules_dir)
category_rules, dasha_rules, planet_house_rules, product_rules, comm_rules = {}, {}, {}, {}, {}

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
print(f"✓ Loaded rule JSONs from {rules_dir}")

# ── Question bank — covers all 7 query types ─────────────────────────────────
QUESTIONS = {
    "simple_factual": [
        "What is my name?",
        "What is my lagna?",
        "What is my rashi?",
        "What is my nakshatra?",
        "How old am I?",
        "What is my date of birth?",
        "What is my ascendant sign?",
        "Who am I?",
        "What sign am I?",
        "Mera naam kya hai?",
        "Meri rashi kya hai?",
        "Mera lagna kya hai?",
        "Main kitne saal ka hun?",
    ],
    "timing_marriage": [
        "When will I get married?",
        "What is my marriage timing?",
        "When is my best marriage window?",
        "Will I get married this year?",
        "When will I find my life partner?",
        "What year will I get married?",
        "Meri shaadi kab hogi?",
        "Shaadi ka time kab hai?",
        "Kab milegi mujhe life partner?",
        "Is year shaadi hogi kya?",
    ],
    "timing_career": [
        "When will I get a job?",
        "When will I get a promotion?",
        "When will my career improve?",
        "When will I start my own business?",
        "When will I get a government job?",
        "What is the best time for my career?",
        "Naukri kab milegi?",
        "Promotion kab milega?",
        "Career kab improve hoga?",
        "Business kab start karun?",
    ],
    "timing_finance": [
        "When will my financial situation improve?",
        "When will I get rich?",
        "When will my income increase?",
        "When will my debts clear?",
        "When will I get a windfall?",
        "When will I have financial stability?",
        "Paisa kab aayega?",
        "Financial situation kab improve hogi?",
        "Kab milega paisa?",
        "Debt kab clear hoga?",
    ],
    "timing_property": [
        "When should I buy a house?",
        "When will I get my own home?",
        "When is the best time to buy property?",
        "Will I buy a house this year?",
        "Ghar kab khareedun?",
        "Property kab milegi?",
    ],
    "timing_children": [
        "When will I have children?",
        "When will I become a parent?",
        "Will I have children?",
        "When will I have my first child?",
        "Bacche kab honge?",
        "Kab banega main parent?",
    ],
    "timing_foreign": [
        "When will I travel abroad?",
        "Will I settle in a foreign country?",
        "When will I go to a foreign country?",
        "Videsh kab jaunga?",
        "Foreign settlement hogi kya?",
    ],
    "timing_health": [
        "When will my health improve?",
        "When will I recover from illness?",
        "Health kab theek hogi?",
    ],
    "past_event": [
        "When did I get married?",
        "Which month and year did I get married?",
        "When did I start my current job?",
        "When did I graduate college?",
        "When did I move to my current city?",
        "What happened to me in 2020?",
        "What was significant in my life in 2018?",
        "When did I have my first child?",
        "When did I buy my house?",
        "What major event happened in my life in 2015?",
        "Main kab shadi hua?",
        "Meri naukri kab lagi?",
        "College kab complete kiya?",
    ],
    "analysis_career": [
        "What is my field of work?",
        "What career is best for me?",
        "Am I suited for business or job?",
        "What does my chart say about my career?",
        "Will I be successful in my career?",
        "What profession does my chart indicate?",
        "Mera career kaisa rahega?",
        "Konsa kaam mujhe suit karta hai?",
        "Business ya job — kya better hai mere liye?",
    ],
    "analysis_marriage": [
        "Will I have a love marriage or arranged marriage?",
        "What kind of spouse will I have?",
        "Is marriage promised in my chart?",
        "Will my marriage be happy?",
        "What does my chart say about my marriage?",
        "Love marriage hogi ya arranged?",
        "Mera spouse kaisa hoga?",
        "Shaadi khush hogi kya?",
    ],
    "analysis_finance": [
        "Will I be wealthy?",
        "What does my chart say about my finances?",
        "Am I destined to be rich?",
        "Will I have financial stability?",
        "Kya main ameer banunga?",
        "Mera financial future kaisa hai?",
    ],
    "analysis_health": [
        "What does my chart say about my health?",
        "Am I prone to any health issues?",
        "What health challenges might I face?",
        "Meri health kaisi rahegi?",
        "Koi health problem toh nahi hogi?",
    ],
    "analysis_general": [
        "What are the strengths in my chart?",
        "What are the challenges in my chart?",
        "What does my current dasha indicate?",
        "What is the overall prediction for my life?",
        "Mera chart kya kehta hai?",
        "Current dasha mein kya hoga?",
        "Mere chart ki strengths kya hain?",
    ],
    "emotional": [
        "I feel very unlucky. Nothing is going right.",
        "I am very depressed and hopeless.",
        "Why am I always struggling?",
        "I feel like giving up. Nothing works for me.",
        "Why do I keep failing despite hard work?",
        "I am going through a very difficult time.",
        "Why is life so hard for me?",
        "I feel like I have no future.",
        "Mujhe bahut tension hai.",
        "Main bahut pareshan hun.",
        "Kuch bhi theek nahi ho raha.",
        "Main haar gaya hun.",
        "Zindagi bahut mushkil lag rahi hai.",
    ],
    "safety": [
        "When will I die?",
        "How long will I live?",
        "Will I survive this illness?",
        "Am I going to die soon?",
        "Kab marunga main?",
        "Kitna jeeyunga?",
        "Kya main theek ho jaunga?",
    ],
    "remedy": [
        "What remedy should I do for my marriage?",
        "Which gemstone should I wear?",
        "What rudraksha is good for me?",
        "What upay should I do for career?",
        "Suggest a product for my problems.",
        "What should I wear to improve my luck?",
        "Kaunsa rudraksha pehnu?",
        "Koi upay batao career ke liye.",
        "Shaadi ke liye kya remedy karun?",
        "Kaunsa gemstone mujhe suit karega?",
    ],
    "dasha_reading": [
        "What dasha am I running now?",
        "Which mahadasha am I currently in?",
        "What is my current antardasha?",
        "What pratyantar dasha am I in right now?",
        "When does my current mahadasha end?",
        "When did my current mahadasha start?",
        "What dasha was I in during 2020?",
        "Which dasha was active in March 2018?",
        "What antardasha was running in 2022?",
        "Tell me about my current dasha period.",
        "Abhi kaun sa dasha chal raha hai?",
        "Current mahadasha kya hai?",
        "Mere current dasha ki details batao.",
        "2020 mein kaun sa dasha tha?",
        "Current pratyantar kab khatam hoga?",
    ],
    "age_aware_timing": [
        "At what age will I get married?",
        "How old will I be when I get my first job?",
        "At what age am I likely to become a parent?",
        "How old will I be when I buy my first house?",
        "At what age will I achieve career success?",
        "When will I get married? Tell me my age at that time.",
        "What age will I be when I settle abroad?",
        "At what age did I likely graduate college?",
        "How old was I when I got married?",
        "What age will I be when my finances improve significantly?",
        "Kitni umar mein shaadi hogi?",
        "Main kitne saal ka hone par job milegi?",
        "Kis age mein parent banunga?",
    ],
    "followup_context": [
        "But I am already married.",
        "I didn't get married in that period.",
        "That period has already passed.",
        "Can you be more specific?",
        "What about the next period?",
        "And after that?",
        "Main toh pehle se shadi shuda hun.",
        "Woh period toh beet gaya.",
        "Aur specific batao.",
    ],
}

# Flatten to weighted list (more timing/analysis, fewer safety)
QUESTION_WEIGHTS = {
    "simple_factual": 0.05,
    "timing_marriage": 0.10,
    "timing_career": 0.10,
    "timing_finance": 0.08,
    "timing_property": 0.05,
    "timing_children": 0.04,
    "timing_foreign": 0.03,
    "timing_health": 0.03,
    "past_event": 0.08,  # Reduced slightly to make room
    "analysis_career": 0.08,
    "analysis_marriage": 0.06,
    "analysis_finance": 0.04,
    "analysis_health": 0.04,
    "analysis_general": 0.04,
    "emotional": 0.04,
    "safety": 0.02,
    "remedy": 0.03,
    "dasha_reading": 0.06,  # NEW: Dasha parsing training
    "age_aware_timing": 0.05,  # NEW: Age-aware predictions
    "followup_context": 0.00,  # not standalone
}

def _sample_question():
    """Sample a question type and question based on weights."""
    types = list(QUESTION_WEIGHTS.keys())
    weights = [QUESTION_WEIGHTS[t] for t in types]
    # Normalize
    total = sum(weights)
    weights = [w / total for w in weights]
    qtype = random.choices(types, weights=weights, k=1)[0]
    question = random.choice(QUESTIONS[qtype])
    return qtype, question


# ── System prompt for generation ──────────────────────────────────────────────
_TODAY = date.today().strftime("%d %b %Y")

GENERATION_SYSTEM_PROMPT = f"""You are generating training examples for a KP astrology AI named "Jyotish".

TODAY'S DATE: {_TODAY}

CRITICAL RULES — EVERY RESPONSE MUST FOLLOW ALL OF THESE:

1. LENGTH: 
   - Simple factual (name/lagna/rashi/age) = 1 sentence ONLY
   - Timing questions = 2-3 sentences max
   - Analysis questions = 3-4 sentences max  
   - Emotional = 2-3 sentences (empathy first, then astrological perspective)
   - Safety (death/longevity) = 2 sentences (compassionate redirect only)
   - Remedy = 2-3 sentences
   - ABSOLUTE MAXIMUM: 4 sentences. NEVER exceed.

2. FORMAT:
   - NO markdown, NO **bold**, NO headers, NO bullets, NO numbered lists
   - Plain prose only. One continuous block of text.
   - NO paragraph breaks.

3. ADDRESS:
   - ALWAYS address as "[Name] ji" using the name from the chart YAML
   - NEVER say "the native", "the person", "the querent"

4. LANGUAGE (CRITICAL - MATCH THE QUESTION'S LANGUAGE EXACTLY):
   - DETECTION: If question contains ANY Hindi/Hinglish words (kab, kya, mera, hai, hogi, milega, hoga, kaise, kyun, ji, aap, mujhe, etc.) → respond in Hindi/Hinglish
   - English question → 100% English response. ZERO Hindi words. Not even "ji" in the middle of sentences.
   - Hindi/Hinglish question → FULL Hindi/Hinglish response. Use Hindi sentence structure.
   - NEVER write pure English when the question has Hindi words.
   
   Examples:
   - Q: "When will I get married?" → A: "Yash ji, your marriage will come in..." (English)
   - Q: "Promotion kab milega?" → A: "Yash ji, aapki promotion Apr 2027 mein aayegi jab..." (Hinglish)
   - Q: "Meri shaadi kab hogi?" → A: "Yash ji, aapki shaadi Jul 2026 mein hogi jab..." (Hinglish)

5. JUSTIFICATION (mandatory for predictions):
   - Every prediction must cite: cusp sub-lord + house numbers
   - Example: "your 7th cusp sub-lord Saturn signifies houses 2,7 which are marriage-positive"
   - NEVER give a bare conclusion without reasoning.

6. DATES:
   - ALWAYS use: "Oct 2025", "Jan 2028", "Mar 2027 to Aug 2027"
   - NEVER use: "2025-10", "upcoming period", "soon", "favorable time"
   - Past dates (before {_TODAY}) → past tense
   - Future dates → future tense
   - NEVER say "upcoming" for a date that has already passed.

7. AGE REFERENCE:
   - For timing predictions, mention the person's age at the predicted event inline.

8. CONTENT:
   - Answer DIRECTLY in the first sentence. No methodology buildup.
   - NEVER explain KP theory. Just give the answer with justification.
   - Products: ONLY when user explicitly asks for remedy/upay/gemstone.

9. FORBIDDEN PHRASES (never use):
   - "According to KP principles", "Based on planetary positions"
   - "Let me analyze", "I can analyze", "requires analysis"
   - "Confidence: medium/high", "KP Analysis for"
   - "The Pratyantar Lord's influence adds depth"

10. SAFETY QUERIES (death/longevity):
    - NEVER give timing. ALWAYS redirect compassionately to medical professionals.
    - Match the language of the question.

Read the chart YAML carefully. Use the actual dasha dates, cusp sub-lords, and house significations from the chart.
Return ONLY the response text. No labels, no "Response:", no explanation."""


def _build_user_prompt(qtype: str, question: str, chart_yaml: str, rules_context: str) -> str:
    """Build the user prompt for Claude to generate a consultation response."""
    return f"""CHART DATA:
{chart_yaml}

RELEVANT KP RULES:
{rules_context}

USER QUESTION ({qtype}): {question}

Generate the ideal Jyotish response following ALL the rules above."""


def _get_rules_context(qtype: str) -> str:
    """Get COMPREHENSIVE rules context based on question type.
    
    CRITICAL FIX: Previously only used 3 rules max per category.
    NOW: Uses 10-15 relevant rules + planet-house combos + sub-lord interpretations + dasha rules.
    Chart-specific extraction happens during Claude generation, not here.
    """
    lines = []
    
    # === 1. CATEGORY RULES (10-15 rules instead of 3) ===
    category_map = {
        "marriage": "marriage",
        "career": "career",
        "analysis_career": "career",
        "finance": "finance",
        "financial": "finance",
        "health": "health",
        "property": "property",
        "education": "education",
        "children": "children"
    }
    
    for key, cat in category_map.items():
        if key in qtype:
            rules = category_rules.get("rules", [])
            relevant_rules = [r for r in rules if r.get("category") == cat][:10]  # Increased from 3 to 10
            if relevant_rules:
                lines.append(f"\n## Category Rules ({cat.title()}):")
                for r in relevant_rules:
                    lines.append(f"- {r['rule_text']}")
            break
    
    # === 2. PLANET-HOUSE COMBINATIONS (NEW - was never used before) ===
    if planet_house_rules.get("combinations"):
        # Get relevant houses for the query type
        relevant_houses = _get_relevant_houses(qtype)
        combos = planet_house_rules.get("combinations", [])
        
        # Find planet-house combos for relevant houses
        relevant_combos = [c for c in combos if c.get("house") in relevant_houses][:8]
        if relevant_combos:
            lines.append(f"\n## Planet-House Combinations:")
            for c in relevant_combos:
                lines.append(f"- {c['planet']} in {c['house']}th house: {c['interpretation_english'][:120]}")
    
    # === 3. SUB-LORD INTERPRETATIONS (NEW - was never used before) ===
    if ("timing" in qtype or "analysis" in qtype) and planet_house_rules.get("sublord_interpretations"):
        relevant_cusps = _get_relevant_cusps(qtype)
        sublords = planet_house_rules.get("sublord_interpretations", [])
        
        # Find sub-lord rules for relevant cusps
        relevant_sublords = [s for s in sublords if s.get("cusp") in relevant_cusps][:6]
        if relevant_sublords:
            lines.append(f"\n## Cusp Sub-Lord Interpretations:")
            for s in relevant_sublords:
                lines.append(f"- Cusp {s['cusp']} with {s['sublord']}: {s['interpretation'][:100]}")
    
    # === 4. DASHA RULES (NEW - was never used before) ===
    if "timing" in qtype or "past_event" in qtype:
        dasha_rule_list = dasha_rules.get("dasha_rules", [])[:5]
        if dasha_rule_list:
            lines.append(f"\n## Dasha Period Rules:")
            for r in dasha_rule_list:
                lines.append(f"- {r['rule_text'][:150]}")
    
    # === 5. PRODUCT RECOMMENDATIONS (only for remedy queries) ===
    if "remedy" in qtype:
        prod_rule_list = product_rules.get("recommendation_rules", [])[:3]
        if prod_rule_list:
            lines.append(f"\n## Product Recommendations:")
            for r in prod_rule_list:
                lines.append(f"- Weak {r['weak_planet']}: {r['recommendation_text_english'][:120]}")
    
    # === 6. CONVERSATION TEMPLATES ===
    comm_templates = comm_rules.get("conversation_templates", [])
    context_key = qtype.split("_")[0]
    relevant_templates = [t for t in comm_templates if context_key in t.get("context", "")][:2]
    if relevant_templates:
        lines.append(f"\n## Response Style:")
        for t in relevant_templates:
            lines.append(f"- {t.get('english', '')[:150]}")
    
    # Fallback if no rules found
    if not lines:
        return "Use KP house significations: Marriage=2,7,11; Career=2,6,10; Finance=2,11,8; Health=1,6,8"
    
    return "\n".join(lines)


def _get_relevant_houses(qtype: str) -> list:
    """Get relevant house numbers based on query type."""
    house_map = {
        "marriage": [1, 2, 5, 7, 11],
        "career": [1, 2, 6, 10, 11],
        "finance": [2, 6, 8, 11],
        "health": [1, 6, 8, 12],
        "property": [4, 10, 11],
        "education": [3, 4, 5, 9],
        "children": [2, 5, 11]
    }
    
    for key, houses in house_map.items():
        if key in qtype:
            return houses
    return [1, 2, 10, 11]  # Default


def _get_relevant_cusps(qtype: str) -> list:
    """Get relevant cusp numbers based on query type."""
    cusp_map = {
        "marriage": [7, 2, 11],
        "career": [10, 2, 6],
        "finance": [2, 11, 8],
        "health": [1, 6, 8],
        "property": [4, 11],
        "education": [4, 9],
        "children": [5, 2, 11]
    }
    
    for key, cusps in cusp_map.items():
        if key in qtype:
            return cusps
    return [1, 10, 11]  # Default


def _generate_example(qtype: str, question: str, chart_yaml: str, chart_name: str) -> dict | None:
    """Generate a single SFT example via Claude API."""
    rules_context = _get_rules_context(qtype)
    user_prompt = _build_user_prompt(qtype, question, chart_yaml, rules_context)

    try:
        response = client.messages.create(
            model=args.model,
            max_tokens=300,
            system=GENERATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )
        answer = response.content[0].text.strip()

        # Basic quality checks
        if len(answer) < 20:
            return None
        if len(answer.split()) > 120:
            return None
        if "**" in answer or "##" in answer:
            return None
        if "the native" in answer.lower():
            return None

        # Build the instruction in the same format as inference
        # System prompt + chart YAML + question (matches 09_chat_ui.py format)
        instruction = f"[CHART]\n{chart_yaml}\n\n[QUESTION]\n{question}"

        return {
            "instruction": instruction,
            "input": "",
            "output": answer,
            "metadata": {
                "qtype": qtype,
                "chart_name": chart_name,
                "question": question,
                "model": args.model,
                "generated_date": _TODAY,
            }
        }
    except anthropic.RateLimitError:
        time.sleep(30)
        return None
    except anthropic.APIError as e:
        print(f"  API error: {e}")
        return None


def _load_checkpoint() -> list:
    """Load existing checkpoint if resuming."""
    if checkpoint_file.exists():
        examples = []
        with open(checkpoint_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        print(f"  Resumed from checkpoint: {len(examples)} existing examples")
        return examples
    return []


def _save_checkpoint(examples: list):
    """Save examples to checkpoint file."""
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _save_final_dataset(examples: list):
    """Save final dataset in HuggingFace Arrow format (same as existing sft_train)."""
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)

    ds = Dataset.from_list(examples)
    ds.save_to_disk(str(final_dir))
    print(f"✓ Saved {len(examples)} examples to {final_dir}")

    # Also save as JSONL for inspection
    jsonl_path = output_dir / "sft_consultation.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"✓ Also saved as JSONL: {jsonl_path}")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH API FUNCTIONS (50% cheaper than sync API)
# ══════════════════════════════════════════════════════════════════════════════

def create_batch_requests(charts, target_count, existing_count):
    """Create batch request list for Anthropic Batch API."""
    remaining = target_count - existing_count
    print(f"\nCreating {remaining} batch requests...")
    
    requests = []
    metadata_map = {}
    
    for i in range(remaining):
        chart = random.choice(charts)
        qtype, question = _sample_question()
        rules_context = _get_rules_context(qtype)
        user_prompt = _build_user_prompt(qtype, question, chart["yaml"], rules_context)
        
        custom_id = f"sft_{existing_count + i + 1:05d}"
        
        request = {
            "custom_id": custom_id,
            "params": {
                "model": args.model,
                "max_tokens": 1024,
                "system": GENERATION_SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7
            }
        }
        requests.append(request)
        
        # Save metadata for this request
        metadata_map[custom_id] = {
            "qtype": qtype,
            "question": question,
            "chart_name": chart["name"],
            "chart_yaml": chart["yaml"]
        }
        
        if i == 0 or (i + 1) % 1000 == 0:
            print(f"  Created {i + 1}/{remaining} requests...")
    
    # Save metadata mapping
    metadata_file = output_dir / "batch_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_map, f, indent=2)
    
    print(f"✓ Saved metadata for {len(metadata_map)} requests")
    
    return requests


def submit_batch(requests, chunk_idx=0):
    """Submit batch to Anthropic API."""
    batch_file = output_dir / f"batch_requests_chunk{chunk_idx}.jsonl"
    
    # Write requests to JSONL
    with open(batch_file, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    
    print(f"\n✓ Saved {len(requests)} requests to {batch_file}")
    print(f"Submitting batch chunk {chunk_idx}...")
    
    # Create batch via API
    batch = client.messages.batches.create(
        requests=requests
    )
    
    print(f"✓ Batch submitted: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  Request counts: processing={batch.request_counts.processing}")
    
    # Save batch metadata
    meta_file = output_dir / "batch_meta.json"
    meta = {}
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
    
    meta[f"chunk_{chunk_idx}"] = {
        "batch_id": batch.id,
        "request_count": len(requests),
        "submitted_at": _TODAY,
        "status": batch.processing_status
    }
    
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    print(f"✓ Batch metadata saved to {meta_file}")
    
    return batch.id


def check_batch_status(batch_id):
    """Check batch status."""
    batch = client.messages.batches.retrieve(batch_id)
    
    print(f"\n{'='*80}")
    print(f"Batch: {batch_id}")
    print(f"{'='*80}")
    print(f"Status: {batch.processing_status}")
    print(f"Request counts:")
    print(f"  Processing: {batch.request_counts.processing}")
    print(f"  Succeeded: {batch.request_counts.succeeded}")
    print(f"  Errored: {batch.request_counts.errored}")
    print(f"  Canceled: {batch.request_counts.canceled}")
    print(f"  Expired: {batch.request_counts.expired}")
    
    if batch.processing_status == "ended":
        print(f"\n✓ Batch complete!")
        print(f"  Use: python scripts/19_generate_sft_consultation.py --batch-download {batch_id}")
    elif batch.processing_status == "in_progress":
        print(f"\n⏳ Still processing... check again later")
    
    return batch


def download_batch_results(batch_id):
    """Download batch results and merge with existing checkpoint."""
    print(f"\n{'='*80}")
    print(f"Downloading batch results: {batch_id}")
    print(f"{'='*80}")
    
    batch = client.messages.batches.retrieve(batch_id)
    
    if batch.processing_status != "ended":
        print(f"❌ Batch not complete (status: {batch.processing_status})")
        return
    
    # Download results using the iterator
    results_file = output_dir / f"batch_results_{batch_id}.jsonl"
    count = 0
    
    print(f"Downloading results...")
    for result in client.messages.batches.results(batch_id):
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.model_dump()) + "\n")
        count += 1
        if count % 100 == 0:
            print(f"  Downloaded {count} results...")
    
    print(f"✓ Downloaded {count} results to {results_file}")
    
    # Load metadata
    metadata_file = output_dir / "batch_metadata.json"
    metadata_map = {}
    if metadata_file.exists():
        with open(metadata_file, encoding="utf-8") as f:
            metadata_map = json.load(f)
    
    # Parse results into SFT format
    new_examples = []
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            result = json.loads(line)
            
            if result["result"]["type"] == "succeeded":
                msg = result["result"]["message"]
                response_text = msg["content"][0]["text"].strip()
                custom_id = result["custom_id"]
                
                # Get metadata
                meta = metadata_map.get(custom_id, {})
                chart_yaml = meta.get("chart_yaml", "")
                question_text = meta.get("question", "")
                
                # Quality checks (same as sync)
                if len(response_text) < 20:
                    continue
                if len(response_text.split()) > 120:
                    continue
                if "**" in response_text or "##" in response_text:
                    continue
                if "the native" in response_text.lower():
                    continue
                
                instruction = f"[CHART]\n{chart_yaml}\n\n[QUESTION]\n{question_text}"
                
                example = {
                    "instruction": instruction,
                    "input": "",
                    "output": response_text,
                    "metadata": {
                        "qtype": meta.get("qtype", "unknown"),
                        "chart_name": meta.get("chart_name", "unknown"),
                        "question": question_text,
                        "model": args.model,
                        "generated_date": _TODAY,
                        "batch_id": batch_id,
                        "custom_id": custom_id
                    }
                }
                new_examples.append(example)
    
    print(f"✓ Parsed {len(new_examples)} valid examples (filtered {count - len(new_examples)} low-quality)")
    
    # Load existing checkpoint
    existing = _load_checkpoint()
    print(f"✓ Loaded {len(existing)} existing examples")
    
    # Merge
    all_examples = existing + new_examples
    print(f"✓ Total: {len(all_examples)} examples")
    
    # Save merged checkpoint
    _save_checkpoint(all_examples)
    print(f"✓ Saved merged checkpoint")
    
    # Save final dataset
    _save_final_dataset(all_examples)
    
    print(f"\n{'='*80}")
    print(f"✓ COMPLETE: {len(all_examples)} total examples ready")
    print(f"{'='*80}")


# ── Main generation loop ──────────────────────────────────────────────────────
def main():
    # Handle batch operations first
    if args.batch_check:
        check_batch_status(args.batch_check)
        return
    
    if args.batch_download:
        download_batch_results(args.batch_download)
        return
    
    print("=" * 80)
    print("SFT CONSULTATION DATASET GENERATOR")
    print("=" * 80)
    print(f"Target: {args.count} examples")
    print(f"Model: {args.model}")
    print(f"Kundali files: {len(kundali_files)}")
    print(f"Dry run: {args.dry_run}")
    if args.batch_create:
        print(f"Mode: BATCH API (50% cheaper)")
    print("=" * 80)

    # Load chart YAMLs using chart_preprocessor.py
    charts = []
    for kf in kundali_files:
        try:
            with open(kf, encoding="utf-8") as f:
                raw = f.read()
            yaml_str = chart_to_yaml(raw)  # Converts JSON to compact YAML (~1500 chars)
            name = kf.stem.replace("kundali_", "").replace("_", " ")
            charts.append({"name": name, "yaml": yaml_str, "file": kf.name})
        except Exception as e:
            print(f"  Warning: could not load {kf.name}: {e}")
    print(f"✓ Loaded {len(charts)} chart YAMLs")

    if args.dry_run:
        print("\n--- DRY RUN: generating 10 examples ---\n")
        for i in range(10):
            chart = random.choice(charts)
            qtype, question = _sample_question()
            ex = _generate_example(qtype, question, chart["yaml"], chart["name"])
            if ex:
                print(f"[{i+1}] qtype={qtype}")
                print(f"  Q: {question}")
                print(f"  A: {ex['output']}")
                print()
        return

    # BATCH API MODE
    if args.batch_create:
        existing = _load_checkpoint()
        print(f"\n✓ Found {len(existing)} existing examples")
        
        remaining = args.count - len(existing)
        if remaining <= 0:
            print(f"✓ Already have {len(existing)} examples (target: {args.count})")
            return
        
        print(f"Need to generate {remaining} more examples via Batch API")
        
        # Split into chunks
        chunk_size = args.chunk_size_batch
        chunks = []
        for i in range(0, remaining, chunk_size):
            chunk_count = min(chunk_size, remaining - i)
            chunks.append(chunk_count)
        
        print(f"Will create {len(chunks)} batch chunks: {chunks}")
        
        batch_ids = []
        for idx, chunk_count in enumerate(chunks):
            print(f"\n{'='*80}")
            print(f"CHUNK {idx + 1}/{len(chunks)}: {chunk_count} requests")
            print(f"{'='*80}")
            
            requests = create_batch_requests(charts, len(existing) + sum(chunks[:idx+1]), 
                                            len(existing) + sum(chunks[:idx]))
            batch_id = submit_batch(requests, chunk_idx=idx)
            batch_ids.append(batch_id)
            
            time.sleep(2)  # Rate limit safety
        
        print(f"\n{'='*80}")
        print(f"ALL BATCHES SUBMITTED")
        print(f"{'='*80}")
        for idx, bid in enumerate(batch_ids):
            print(f"Chunk {idx}: {bid}")
        
        print(f"\nNext steps:")
        print(f"1. Wait 1-24 hours for batches to complete (usually <1 hour)")
        print(f"2. Check status: python scripts/19_generate_sft_consultation.py --batch-check <batch_id>")
        print(f"3. Download: python scripts/19_generate_sft_consultation.py --batch-download <batch_id>")
        print(f"\nCost savings: 50% cheaper than sync API (${remaining * 0.003 * 0.5:.2f} vs ${remaining * 0.003:.2f})")
        return

    # SYNC API MODE (original code)
    # Resume or start fresh
    examples = _load_checkpoint() if args.resume else []
    start_count = len(examples)
    target = args.count

    print(f"\nGenerating {target - start_count} more examples (have {start_count})...")
    print("Progress: ", end="", flush=True)

    errors = 0
    i = start_count

    while i < target:
        chart = random.choice(charts)
        qtype, question = _sample_question()

        ex = _generate_example(qtype, question, chart["yaml"], chart["name"])
        if ex:
            examples.append(ex)
            i += 1
            if i % 10 == 0:
                print(f"{i}", end=" ", flush=True)
            if i % args.batch_size == 0:
                _save_checkpoint(examples)
                print(f"\n  [checkpoint saved at {i}]", end=" ", flush=True)
        else:
            errors += 1
            if errors > 50:
                print(f"\n⚠️  Too many errors ({errors}), stopping early at {i} examples")
                break
            time.sleep(1)

    print(f"\n\n✓ Generated {len(examples)} examples ({errors} errors)")

    # Save final dataset
    _save_final_dataset(examples)

    # Quality report
    qtypes = {}
    for ex in examples:
        qt = ex["metadata"]["qtype"]
        qtypes[qt] = qtypes.get(qt, 0) + 1
    print("\nQuestion type distribution:")
    for qt, count in sorted(qtypes.items(), key=lambda x: -x[1]):
        print(f"  {qt}: {count} ({100*count//len(examples)}%)")

    print(f"\n{'='*80}")
    print("DONE — SFT consultation dataset generated")
    print(f"{'='*80}")
    print(f"Next step: retrain SFT using this dataset")
    print(f"  python scripts/04_train_sft.py  (after updating sft_config.yaml train_data path)")


if __name__ == "__main__":
    main()
