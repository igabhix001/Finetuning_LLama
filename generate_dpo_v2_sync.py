"""
DPO Dataset V2 - Synchronous Generation (Immediate Results)
============================================================
Generates 500 new high-quality DPO pairs using OpenAI synchronous API.
Focuses on fixing critical issues found in testing:
1. Name extraction and consistency
2. Past event handling (correct years, past tense)
3. Date reading from actual dasha periods
4. Safety intercepts
5. Simple factual questions (1 sentence)

Usage:
    python generate_dpo_v2_sync.py --count 500 --workers 10
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found in .env")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Installing OpenAI library...")
    os.system(f"{sys.executable} -m pip install openai>=1.30.0 -q")
    from openai import OpenAI

client = OpenAI(api_key=api_key)

# Import chart preprocessor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
try:
    from chart_preprocessor import chart_to_yaml
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from chart_preprocessor import chart_to_yaml

import glob

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD KUNDALI DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_kundalis():
    """Load all kundali files"""
    patterns = [
        "sample_kundali/kundali_*.json",
        "Finetuning_LLama/sample_kundali/kundali_*.json",
        "/workspace/Finetuning_LLama/sample_kundali/kundali_*.json",
    ]
    
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    
    if not found:
        print("❌ No kundali files found")
        sys.exit(1)
    
    charts = []
    for fp in found:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                raw_json = f.read()
            data = json.loads(raw_json)
            yaml_str = chart_to_yaml(raw_json)
            
            charts.append({
                "name": data.get("name", "Unknown"),
                "yaml": yaml_str,
                "file": os.path.basename(fp)
            })
        except Exception as e:
            print(f"  ✗ Failed to load {fp}: {e}")
    
    return charts

# ═══════════════════════════════════════════════════════════════════════════════
# TARGETED QUESTIONS - Focus on problem areas
# ═══════════════════════════════════════════════════════════════════════════════

QUESTIONS = [
    # Name extraction (critical)
    "What is my name?",
    "Mera naam kya hai?",
    
    # Past events (critical - must use past tense and correct years)
    "What happened in my life in 2020?",
    "What happened in my life in 2021?",
    "What happened in my life in 2022?",
    "What happened in my life in 2023?",
    "What happened in my life in 2024?",
    "What happened in my career year by year from 2020 to 2025?",
    "When did I complete my education?",
    "When did I get my first job?",
    "Did I have any health issues in 2020?",
    "Did I have any health issues in 2021?",
    "Did I have any health issues in the past 5 years?",
    "What was my financial situation in 2020?",
    "Meri zindagi mein 2020 mein kya hua?",
    "2021 mein mera career kaisa tha?",
    "Maine apni padhai kab complete ki?",
    "Meri pehli naukri kab lagi?",
    
    # Simple factual (must be 1 sentence)
    "What is my lagna?",
    "What is my rashi?",
    "What is my nakshatra?",
    "What is my moon sign?",
    "What is my date of birth?",
    "Where was I born?",
    "What is my current mahadasha?",
    "What is today's date?",
    "Mera lagna kya hai?",
    "Meri rashi kya hai?",
    
    # Safety (must redirect, not predict)
    "When will I die?",
    "Kab marunga?",
    "Do I have cancer?",
    "Will I die soon?",
    "Kya mujhe koi badi bimari hai?",
    
    # Marriage timing (must read actual dasha dates)
    "When will I get married?",
    "Meri shaadi kab hogi?",
    "When is the best period for marriage?",
    "At what age will I get married?",
    
    # Career timing (must read actual dasha dates)
    "When will I get a new job?",
    "When will I get a promotion?",
    "Meri naukri kab lagegi?",
    "Kab tak promotion milega?",
    
    # Financial timing (must read actual dasha dates)
    "When will my financial situation improve?",
    "When will I get a salary increase?",
    "Meri income kab badhegi?",
    
    # Emotional (must show empathy + use correct name)
    "I feel very unlucky. Nothing works out for me.",
    "My health has been troubling me lately.",
    "I am very confused about my career direction.",
    "I am stressed about my relationships.",
    "I am very worried about money.",
    "Mujhe bahut tension ho rahi hai, kya hoga mera?",
]

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS - Stricter rules
# ═══════════════════════════════════════════════════════════════════════════════

CHOSEN_PROMPT = """You are generating the IDEAL response for a KP astrology AI chatbot named "Jyotish".

*** CRITICAL RULES (ZERO TOLERANCE) ***

1. NAME EXTRACTION:
   - Extract the person's name from chart YAML ("name:" field)
   - Use ONLY that exact name: "[Name] ji"
   - NEVER use a different name
   - VIOLATION = INVALID RESPONSE

2. PAST/FUTURE TENSE:
   - Read "today_date:" from YAML (e.g., "10 Feb 2026")
   - Date BEFORE today → PAST tense: "that period has passed (yeh period beet chuka hai)"
   - Date AFTER today → FUTURE tense: "starting from [month year]"
   - For past questions (e.g., "What happened in 2020?") → ONLY mention PAST dates
   - VIOLATION = INVALID RESPONSE

3. DATE READING:
   - Read dasha dates from YAML (mahadasha, antardasha, pratyantar)
   - Use ACTUAL dates from YAML, NOT made-up dates
   - VIOLATION = INVALID RESPONSE

4. LENGTH:
   - Simple questions = 1 sentence
   - Most questions = 2 sentences
   - Complex = 3 sentences max
   - 4 sentences = absolute ceiling

5. FORMAT:
   - NO markdown, headers, bullets
   - NO "Analysis:", "Conclusion:"
   - Date format: "Oct 2025", NOT "2025-10"

6. JUSTIFICATION:
   - Every prediction MUST include WHY: "[Planet] as [Nth] cusp sub-lord signifies houses X,Y"

7. SAFETY:
   - Death/disease questions → Compassionate redirect to medical professional
   - NEVER predict death dates or diagnose diseases

EXAMPLES:

Q: "What is my name?" (chart: name: Anjali Desai)
A: "Anjali Desai ji, your name is Anjali Desai."

Q: "What happened in 2020?" (chart: name: Yash, today_date: 10 Feb 2026, Venus-Rahu AD: May 2019-May 2022)
A: "Yash ji, during 2020 you were in Venus-Rahu AD (May 2019 to May 2022, yeh period beet chuka hai), and Rahu as sub-lord signified houses 3,8 which brought sudden changes."

Q: "When will I get married?" (chart: name: Priya, today_date: 10 Feb 2026, Saturn-Venus AD: Apr 2026-Jun 2029)
A: "Priya ji, your 7th cusp sub-lord Venus signifies houses 2,7,11 which are marriage-positive. Peak window is Oct 2026 to Feb 2027 when Jupiter pratyantar activates house 7."

Q: "Kab marunga?" (SAFETY)
A: "Yash ji, chinta mat karein — jyotish aapko guide karne ke liye hai. Aapke chart mein positive periods hain aur health concerns ke liye doctor se milna behtar hai."

Return ONLY the response text."""

REJECTED_PROMPT = """Generate a BAD response (1-4 sentences, same length as ideal).

Wrong patterns:
- Use DIFFERENT name than chart
- Say "the native" instead of name
- For past questions → give future dates
- Use ISO format: "2025-10"
- No justification (no sub-lord, no houses)
- For "What is my name?" → give astrology analysis
- For death queries → predict death dates

EXAMPLES:

Q: "What is my name?" (chart: name: Anjali Desai)
A: "According to KP principles, the native ka lagna lord Venus hai."

Q: "What happened in 2020?" (chart: name: Yash, today: 2026)
A: "Yash ji, the upcoming period 2027-2029 mein significant changes honge."

Return ONLY the response text."""

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pair(question, chart, retry=0):
    """Generate one chosen+rejected pair"""
    try:
        prompt_text = f"Question: {question}\n\nChart YAML:\n{chart['yaml']}"
        
        # Generate chosen
        chosen_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": CHOSEN_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=250,
            temperature=0.7,
        )
        chosen = chosen_response.choices[0].message.content.strip()
        
        # Generate rejected
        rejected_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": REJECTED_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=250,
            temperature=0.9,
        )
        rejected = rejected_response.choices[0].message.content.strip()
        
        return {
            "prompt": f"Question: {question}\n\nChart:\n{chart['yaml'][:500]}...",  # Truncate for storage
            "chosen": chosen,
            "rejected": rejected,
            "chart_name": chart['name'],
            "question": question,
        }
    
    except Exception as e:
        if retry < 3:
            time.sleep(2 ** retry)  # Exponential backoff
            return generate_pair(question, chart, retry + 1)
        else:
            print(f"\n❌ Failed after 3 retries: {e}")
            return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500, help="Number of pairs to generate")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    args = parser.parse_args()
    
    print("=" * 80)
    print("DPO Dataset V2 - Synchronous Generation")
    print("=" * 80)
    
    # Load charts
    print("\n📂 Loading kundali data...")
    charts = load_kundalis()
    print(f"✓ Loaded {len(charts)} charts")
    
    # Generate combinations
    print(f"\n📝 Generating {args.count} combinations...")
    combinations = []
    for _ in range(args.count):
        question = random.choice(QUESTIONS)
        chart = random.choice(charts)
        combinations.append((question, chart))
    print(f"✓ Generated {len(combinations)} combinations")
    
    # Generate pairs in parallel
    print(f"\n🤖 Generating DPO pairs with {args.workers} workers...")
    print("   This will take ~5-10 minutes for 500 pairs")
    
    pairs = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate_pair, q, c) for q, c in combinations]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            result = future.result()
            if result:
                pairs.append(result)
    
    print(f"\n✓ Generated {len(pairs)} pairs successfully")
    
    # Save to file
    output_dir = Path("data/dpo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "dpo_pairs_v2_fixes.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"\n💾 Saved to: {output_file}")
    
    # Stats
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total pairs: {len(pairs)}")
    print(f"Unique charts: {len(set(p['chart_name'] for p in pairs))}")
    print(f"Unique questions: {len(set(p['question'] for p in pairs))}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Merge with existing dataset:")
    print("   python merge_dpo_datasets.py")
    print("\n2. Prepare for training:")
    print("   python scripts/14_prepare_dpo_dataset.py")
    print("\n3. Train DPO on RunPod:")
    print("   python scripts/15_train_dpo.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
