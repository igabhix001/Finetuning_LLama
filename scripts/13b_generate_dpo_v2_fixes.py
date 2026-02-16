"""
DPO Dataset V2 - Targeted Fixes for Critical Issues
====================================================
Generates 500 new high-quality DPO pairs focused on fixing:
1. Name extraction and consistency
2. Past event handling (correct years, past tense)
3. Date reading from actual dasha periods (not scripted)
4. Safety intercepts (medical, death)
5. Simple factual questions (1 sentence)

This supplements the existing 2557 pairs with targeted improvements.
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import from existing script
sys.path.insert(0, os.path.dirname(__file__))
from chart_preprocessor import chart_to_yaml, load_kundali_json
import glob

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD KUNDALI DATA
# ═══════════════════════════════════════════════════════════════════════════════

def _discover_kundali_files():
    """Find all kundali JSON files"""
    patterns = [
        os.path.join(os.path.dirname(__file__), "..", "sample_kundali", "kundali_*.json"),
        "/workspace/Finetuning_LLama/sample_kundali/kundali_*.json",
    ]
    found = set()
    for pattern in patterns:
        for fp in glob.glob(pattern):
            found.add(os.path.abspath(fp))
    return sorted(found)

def _load_chart_templates():
    """Load real kundali JSONs"""
    kundali_files = _discover_kundali_files()
    if not kundali_files:
        print("❌ No kundali files found")
        sys.exit(1)
    
    templates = []
    for fp in kundali_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                raw_json = f.read()
            data = json.loads(raw_json)
            yaml_str = chart_to_yaml(raw_json)
            
            templates.append({
                "name": data.get("name", "Unknown"),
                "gender": data.get("gender", ""),
                "dob": data.get("birthDetails", {}).get("date", "?"),
                "yaml": yaml_str,
                "source_file": os.path.basename(fp),
            })
        except Exception as e:
            print(f"  ✗ Failed to load {fp}: {e}")
    
    return templates

print("Loading kundali data...")
CHART_TEMPLATES = _load_chart_templates()
print(f"✓ {len(CHART_TEMPLATES)} charts loaded\n")

# ═══════════════════════════════════════════════════════════════════════════════
# TARGETED QUESTION BANK - Focus on problem areas
# ═══════════════════════════════════════════════════════════════════════════════

TARGETED_QUESTIONS = [
    # ── NAME EXTRACTION (50 questions) ────────────────────────────────────────
    ("name_test", "What is my name?"),
    ("name_test", "Mera naam kya hai?"),
    ("name_test", "Can you tell me my name?"),
    ("name_test", "Who am I according to my chart?"),
    ("name_test", "Aap mera naam bata sakte hain?"),
    
    # ── PAST EVENTS (100 questions) - CRITICAL FIX ────────────────────────────
    ("past_events", "What happened in my life in 2020?"),
    ("past_events", "What happened in my life in 2021?"),
    ("past_events", "What happened in my life in 2022?"),
    ("past_events", "What happened in my life in 2023?"),
    ("past_events", "What happened in my life in 2024?"),
    ("past_events", "What happened in my life in 2025?"),
    ("past_events", "Tell me about my career in 2020"),
    ("past_events", "Tell me about my career in 2021"),
    ("past_events", "Tell me about my career in 2022"),
    ("past_events", "Tell me about my career in 2023"),
    ("past_events", "What happened in my career year by year from 2020 to 2025?"),
    ("past_events", "When did I complete my education?"),
    ("past_events", "When did I get my first job?"),
    ("past_events", "When did I get married?"),
    ("past_events", "Did I face any obstacles in 2020?"),
    ("past_events", "Did I face any obstacles in 2021?"),
    ("past_events", "Did I face any obstacles in 2022?"),
    ("past_events", "Did I have any health issues in 2020?"),
    ("past_events", "Did I have any health issues in 2021?"),
    ("past_events", "Did I have any health issues in 2022?"),
    ("past_events", "Did I have any health issues in the past 5 years?"),
    ("past_events", "When did I face the most obstacles in my career?"),
    ("past_events", "What was my financial situation in 2020?"),
    ("past_events", "What was my financial situation in 2021?"),
    ("past_events", "What was my financial situation in 2022?"),
    ("past_events", "Meri zindagi mein 2020 mein kya hua?"),
    ("past_events", "2021 mein mera career kaisa tha?"),
    ("past_events", "2022 mein kya problems aayi?"),
    ("past_events", "Maine apni padhai kab complete ki?"),
    ("past_events", "Meri pehli naukri kab lagi?"),
    ("past_events", "2023 mein meri health kaisi thi?"),
    
    # ── SIMPLE FACTUAL (50 questions) - Must be 1 sentence ────────────────────
    ("simple_factual", "What is my lagna?"),
    ("simple_factual", "What is my rashi?"),
    ("simple_factual", "What is my nakshatra?"),
    ("simple_factual", "What is my moon sign?"),
    ("simple_factual", "What is my sun sign?"),
    ("simple_factual", "What is my date of birth?"),
    ("simple_factual", "Where was I born?"),
    ("simple_factual", "What is my current mahadasha?"),
    ("simple_factual", "Which dasha am I running right now?"),
    ("simple_factual", "What is today's date?"),
    ("simple_factual", "Mera lagna kya hai?"),
    ("simple_factual", "Meri rashi kya hai?"),
    ("simple_factual", "Mera nakshatra kya hai?"),
    ("simple_factual", "Mera janam kab hua?"),
    ("simple_factual", "Main kahan paida hua?"),
    
    # ── SAFETY (30 questions) - Must redirect, not predict ────────────────────
    ("safety", "When will I die?"),
    ("safety", "Kab marunga?"),
    ("safety", "Will I die soon?"),
    ("safety", "Do I have cancer?"),
    ("safety", "Will I have a fatal accident?"),
    ("safety", "Kya mujhe koi badi bimari hai?"),
    ("safety", "Is there any serious illness in my chart?"),
    ("safety", "Meri maut kab hogi?"),
    ("safety", "Will something bad happen to me?"),
    ("safety", "Kya meri zindagi khatre mein hai?"),
    
    # ── MARRIAGE TIMING (50 questions) - Must read actual dasha dates ─────────
    ("marriage", "When will I get married?"),
    ("marriage", "Meri shaadi kab hogi?"),
    ("marriage", "When is the best period for marriage?"),
    ("marriage", "Will my marriage happen this year?"),
    ("marriage", "Kya meri shaadi is saal hogi?"),
    ("marriage", "When will I find my life partner?"),
    ("marriage", "At what age will I get married?"),
    ("marriage", "Mere liye shaadi ka sabse accha time kab hai?"),
    
    # ── CAREER TIMING (50 questions) - Must read actual dasha dates ───────────
    ("career", "When will I get a new job?"),
    ("career", "When will I get a promotion?"),
    ("career", "Meri naukri kab lagegi?"),
    ("career", "When will my career improve?"),
    ("career", "Kab tak promotion milega?"),
    ("career", "When is the best time to change jobs?"),
    
    # ── FINANCIAL TIMING (50 questions) - Must read actual dasha dates ────────
    ("financial", "When will my financial situation improve?"),
    ("financial", "When will I get a salary increase?"),
    ("financial", "Kab tak mera karz utrega?"),
    ("financial", "When will I be financially stable?"),
    ("financial", "Meri income kab badhegi?"),
    
    # ── EMOTIONAL (30 questions) - Must show empathy + name ───────────────────
    ("emotional", "I feel very unlucky. Nothing works out for me."),
    ("emotional", "My health has been troubling me lately."),
    ("emotional", "I am very confused about my career direction."),
    ("emotional", "I am stressed about my relationships."),
    ("emotional", "I am very worried about money."),
    ("emotional", "Everything is falling apart. Is there any hope?"),
    ("emotional", "Mujhe bahut tension ho rahi hai, kya hoga mera?"),
    ("emotional", "Bahut pareshan hun, kab tak yeh mushkilein rahegi?"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED SYSTEM PROMPTS - Stricter rules for problem areas
# ═══════════════════════════════════════════════════════════════════════════════

CHOSEN_SYSTEM_PROMPT_V2 = """You are generating the IDEAL response for a KP astrology AI chatbot named "Jyotish".

*** CRITICAL: NAME EXTRACTION (ZERO TOLERANCE) ***
STEP 1: Extract the person's name from the chart YAML (look for "name:" field).
STEP 2: Use ONLY that exact name in your response. NEVER use any other name.
STEP 3: Address them as "[Exact Name] ji" - example: "Anjali Desai ji", "Yash ji"
VIOLATION: If you use a different name than what's in the YAML, your response is INVALID.

*** CRITICAL: PAST/FUTURE TENSE (ZERO TOLERANCE) ***
STEP 1: Read "today_date:" from the chart YAML (e.g., "10 Feb 2026")
STEP 2: For EVERY date you mention, compare it to today_date:
  - Date BEFORE today_date → PAST tense: "that period has already passed (yeh period beet chuka hai)"
  - Date AFTER today_date → FUTURE tense: "starting from [month year]"
STEP 3: For past event questions (e.g., "What happened in 2020?"), ONLY mention dates from the PAST.
VIOLATION: If you give future dates for past questions, your response is INVALID.

*** CRITICAL: DATE READING (ZERO TOLERANCE) ***
STEP 1: Read the dasha data from the chart YAML (mahadasha, antardasha, pratyantar sections)
STEP 2: Extract ACTUAL start and end dates from the YAML
STEP 3: Use THOSE dates in your response, NOT made-up dates
VIOLATION: If you use dates not present in the YAML, your response is INVALID.

*** HARD LENGTH RULE ***
Simple questions = 1 sentence. Most questions = 2 sentences. Complex = 3 sentences max.
4 sentences is the ABSOLUTE ceiling.

*** LANGUAGE RULES ***
- English question → 100% English response
- Hindi/Hinglish question → Hindi/Hinglish response
- Always address as "[Name] ji"

*** FORMAT ***
- ZERO markdown: no **bold**, no headers, no bullets
- NO paragraph breaks
- NEVER write "Analysis:", "Conclusion:", "Confidence:"

*** DATE FORMAT ***
- ALWAYS: "Oct 2025", "Mar 2027 to Aug 2027"
- NEVER: "2025-10", ISO format

*** JUSTIFICATION (MANDATORY) ***
Every prediction MUST include WHY in the SAME sentence:
- "[Planet] as [Nth] cusp sub-lord signifies houses X,Y"
- Example: "your 7th cusp sub-lord Saturn signifies houses 2,7 which are marriage-positive"

*** SAFETY INTERCEPTS ***
For death/disease questions: Compassionate redirect to medical professional.
NEVER predict death dates or diagnose diseases.

*** EXAMPLES ***

Q: "What is my name?" (chart: name: Anjali Desai)
A: "Anjali Desai ji, your name is Anjali Desai."

Q: "What happened in my life in 2020?" (chart: name: Yash, today_date: 10 Feb 2026, Venus-Rahu AD: May 2019 to May 2022)
A: "Yash ji, during 2020 you were in Venus-Rahu AD (May 2019 to May 2022, yeh period beet chuka hai), and Rahu as sub-lord signified houses 3,8 which brought sudden changes and challenges in that year."

Q: "When will I get married?" (chart: name: Priya, today_date: 10 Feb 2026, Saturn-Venus AD: Apr 2026 to Jun 2029)
A: "Priya ji, your 7th cusp sub-lord Venus signifies houses 2,7,11 which are marriage-positive. Peak window is Saturn-Venus-Jupiter pratyantar from Oct 2026 to Feb 2027, when Jupiter activates house 7."

Q: "When did I get my first job?" (chart: name: Abhi Raj, age_now: 20, dob: 15.03.2006, today_date: 10 Feb 2026)
A: "Abhi Raj ji, you would have gotten your first job around age 18-19, likely during Venus-Mercury AD (Sep 2023 to Jul 2024, yeh period beet chuka hai) when Mercury as 10th cusp sub-lord activated houses 2,6,10."

Q: "Kab marunga?" (SAFETY - Hindi question → Hindi response)
A: "Yash ji, chinta mat karein — jyotish aapko guide karne ke liye hai, darane ke liye nahi. Aapke chart mein kaafi positive periods hain aur health concerns ke liye qualified doctor se milna sabse behtar hai."

Return ONLY the response text. No labels."""

REJECTED_SYSTEM_PROMPT_V2 = """You are generating a BAD response for training data.

*** CRITICAL: Make it SHORT but WRONG ***
Your response MUST be 1-4 sentences, same length as ideal response.
The badness comes from CONTENT, not length.

Pick 3-4 wrong patterns:

NAME (wrong):
- Use a DIFFERENT name than what's in the chart
- Say "the native" instead of the person's name
- Mix up names randomly

DATES (wrong):
- Use ISO format: "2025-10" instead of "Oct 2025"
- Give vague ranges: "between 2028 to 2033"
- NEVER mention pratyantar dashas

TENSE (wrong):
- Treat past dates as future: "2020 will be significant" (when today is 2026)
- For "What happened in 2020?" → give future dates like 2027

CONTENT (wrong):
- Don't answer directly
- Give no justification (no sub-lord, no houses)
- For "What is my name?" → give astrology analysis instead

SAFETY (wrong):
- For death queries: predict death dates
- For disease queries: diagnose diseases

EXAMPLES:

Q: "What is my name?" (chart: name: Anjali Desai)
A: "According to KP principles, the native ka lagna lord Venus hai jo personality govern karta hai."

Q: "What happened in 2020?" (chart: name: Yash, today_date: 2026)
A: "Yash ji, the upcoming period 2027-2029 mein significant changes honge."

Q: "When will I get married?" (chart: name: Priya)
A: "Geeta ji, according to KP principles, the native ka 7th house mein marriage yoga hai. The period 2028-2033 favorable hai."

Return ONLY the response text."""

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def generate_combinations(count=500):
    """Generate (question, chart) combinations"""
    combos = []
    
    # Ensure we cover all charts
    for _ in range(count):
        chart = random.choice(CHART_TEMPLATES)
        question_cat, question_text = random.choice(TARGETED_QUESTIONS)
        
        combos.append({
            "question": question_text,
            "category": question_cat,
            "chart_name": chart["name"],
            "chart_yaml": chart["yaml"],
        })
    
    return combos

def create_batch_requests(combos, model="gpt-4o"):
    """Create OpenAI batch API requests"""
    requests = []
    
    for idx, combo in enumerate(combos):
        # Chosen request
        requests.append({
            "custom_id": f"chosen_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": CHOSEN_SYSTEM_PROMPT_V2},
                    {"role": "user", "content": f"Question: {combo['question']}\n\nChart YAML:\n{combo['chart_yaml']}"}
                ],
                "max_tokens": 250,
                "temperature": 0.7,
            }
        })
        
        # Rejected request
        requests.append({
            "custom_id": f"rejected_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": REJECTED_SYSTEM_PROMPT_V2},
                    {"role": "user", "content": f"Question: {combo['question']}\n\nChart YAML:\n{combo['chart_yaml']}"}
                ],
                "max_tokens": 250,
                "temperature": 0.9,
            }
        })
    
    return requests, combos

def main():
    """Main execution"""
    print("=" * 80)
    print("DPO Dataset V2 - Targeted Fixes Generation")
    print("=" * 80)
    
    # Generate combinations
    print("\n📝 Generating 500 targeted combinations...")
    combos = generate_combinations(count=500)
    print(f"✓ Generated {len(combos)} combinations")
    
    # Create batch requests
    print("\n📦 Creating batch API requests...")
    requests, combos_data = create_batch_requests(combos)
    print(f"✓ Created {len(requests)} requests (chosen + rejected)")
    
    # Save to file
    output_dir = Path("data/dpo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_file = output_dir / "batch_v2_fixes.jsonl"
    with open(batch_file, 'w', encoding='utf-8') as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + '\n')
    
    combos_file = output_dir / "combos_v2_fixes.json"
    with open(combos_file, 'w', encoding='utf-8') as f:
        json.dump(combos_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved batch requests to: {batch_file}")
    print(f"✓ Saved combinations to: {combos_file}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Upload batch file to OpenAI:")
    print(f"   python scripts/13_generate_dpo_dataset.py --upload {batch_file}")
    print("\n2. Wait for completion (check status)")
    print("\n3. Download results:")
    print("   python scripts/13_generate_dpo_dataset.py --download <batch_id>")
    print("\n4. Merge with existing dataset:")
    print("   python scripts/merge_dpo_datasets.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
