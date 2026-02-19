"""
Generate 500 Hindi/Hinglish DPO pairs to fix language mismatch.

Problem: Only 4.2% of dpo_pairs_final.jsonl has Hindi questions.
Model trained on mostly English → responds in English even to Hindi questions.

Solution: Generate 500 pairs where:
  - QUESTION is in Hindi/Hinglish
  - CHOSEN answer is in Hindi/Hinglish (matching language)
  - REJECTED answer is in English (wrong language) OR vague/robotic

Run: python generate_hindi_dpo_500.py
Output: data/dpo/dpo_hindi_500.jsonl
Merge: python generate_hindi_dpo_500.py --merge
"""

import os, sys, json, re, time, random, argparse
from pathlib import Path
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--merge", action="store_true", help="Merge dpo_hindi_500.jsonl into dpo_pairs_final.jsonl")
parser.add_argument("--count", type=int, default=500)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--output", type=str, default="data/dpo/dpo_hindi_500.jsonl")
args = parser.parse_args()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key and not args.merge:
    print("❌ OPENAI_API_KEY not found in .env")
    sys.exit(1)

# ── Merge mode ────────────────────────────────────────────────────────────────
if args.merge:
    final_path = Path("data/dpo/dpo_pairs_final.jsonl")
    hindi_path = Path(args.output)
    if not hindi_path.exists():
        print(f"❌ {hindi_path} not found. Run without --merge first.")
        sys.exit(1)
    final = [json.loads(l) for l in open(final_path, encoding="utf-8")]
    hindi = [json.loads(l) for l in open(hindi_path, encoding="utf-8")]
    merged = final + hindi
    # Backup original
    backup = final_path.with_suffix(".jsonl.bak")
    import shutil
    shutil.copy(final_path, backup)
    with open(final_path, "w", encoding="utf-8") as f:
        for p in merged:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"✅ Merged: {len(final)} + {len(hindi)} = {len(merged)} pairs")
    print(f"   Backup: {backup}")
    print(f"   Output: {final_path}")
    sys.exit(0)

# ── Load chart templates ───────────────────────────────────────────────────────
sys.path.insert(0, "scripts")
from chart_preprocessor import chart_to_yaml
import glob

kundali_files = sorted(glob.glob("sample_kundali/kundali_*.json"))
if not kundali_files:
    print("❌ No kundali files found in sample_kundali/")
    sys.exit(1)

CHARTS = []
for fp in kundali_files:
    try:
        raw = open(fp, encoding="utf-8").read()
        data = json.loads(raw)
        yaml_str = chart_to_yaml(raw)
        CHARTS.append({
            "name": data.get("name", "Unknown"),
            "gender": data.get("gender", ""),
            "dob": data.get("birthDetails", {}).get("date", "?"),
            "yaml": yaml_str,
        })
    except Exception as e:
        print(f"  ✗ {fp}: {e}")

print(f"✓ Loaded {len(CHARTS)} charts: {[c['name'] for c in CHARTS]}")

# ── Hindi question bank ────────────────────────────────────────────────────────
# 100% Hindi/Hinglish questions across all categories
HINDI_QUESTIONS = [
    # Marriage (25 questions)
    ("marriage", "Meri shaadi kab hogi?"),
    ("marriage", "Kya meri shaadi is saal hogi?"),
    ("marriage", "Shaadi mein kitni der aur lagegi?"),
    ("marriage", "Kya love marriage hogi ya arranged?"),
    ("marriage", "Mere liye shaadi ka sabse accha time kab hai?"),
    ("marriage", "Mera 7th house shaadi ke liye kaisa hai?"),
    ("marriage", "Kya meri shaadi mein koi problem hai?"),
    ("marriage", "Shaadi ke baad meri life kaisi hogi?"),
    ("marriage", "Kya mujhe accha life partner milega?"),
    ("marriage", "Meri shaadi mein delay kyun ho rahi hai?"),
    ("marriage", "Kab tak shaadi ho jayegi meri?"),
    ("marriage", "Kya is saal rishta pakka ho sakta hai?"),
    ("marriage", "Mere chart mein shaadi ka yoga hai kya?"),
    ("marriage", "Spouse kaisa hoga mera?"),
    ("marriage", "Kya meri shaadi bahar se hogi?"),
    ("marriage", "Shaadi ke liye konsa dasha period best hai?"),
    ("marriage", "Meri engagement toot gayi, kya dobara koi milega?"),
    ("marriage", "Kitni umar mein hogi meri shaadi?"),
    ("marriage", "Kya agla saal shaadi ke liye acha hai?"),
    ("marriage", "Meri family meri shaadi se khush hogi kya?"),
    ("marriage", "Shaadi ke baad ghar kahan hoga?"),
    ("marriage", "Kya mere chart mein divorce ka yoga hai?"),
    ("marriage", "Mera partner educated hoga kya?"),
    ("marriage", "Kya meri shaadi jaldi hogi?"),
    ("marriage", "Shaadi ke baad financial situation kaisi hogi?"),

    # Career (20 questions)
    ("career", "Meri naukri kab lagegi?"),
    ("career", "Kya mujhe promotion milega is saal?"),
    ("career", "Career mein kab success milegi?"),
    ("career", "Kya mujhe job change karni chahiye?"),
    ("career", "Mera business kab chalega?"),
    ("career", "Naukri mein problem kyun aa rahi hai?"),
    ("career", "Kya mujhe government job milegi?"),
    ("career", "Career ke liye konsa field best hai mere liye?"),
    ("career", "Kya meri job stable rahegi?"),
    ("career", "Promotion kab milega mujhe?"),
    ("career", "Kya foreign mein job milegi?"),
    ("career", "Apna business shuru karna chahiye ya naukri?"),
    ("career", "Career mein sabse accha time kab aayega?"),
    ("career", "Kya is saal job milegi?"),
    ("career", "Naukri se nikaala gaya, kab dobara milegi?"),
    ("career", "Mera boss mujhse khush nahi, kya hoga?"),
    ("career", "Career change ke liye sahi time kab hai?"),
    ("career", "Kya mujhe higher studies karni chahiye?"),
    ("career", "Business mein loss ho raha hai, kab sudhar hoga?"),
    ("career", "Kya meri company mein future hai?"),

    # Finance (20 questions)
    ("finance", "Meri financial situation kab sudharegi?"),
    ("finance", "Paisa kab aayega mere paas?"),
    ("finance", "Kya mujhe loan milega?"),
    ("finance", "Debt se kab chutkara milega?"),
    ("finance", "Kya is saal paisa badhega?"),
    ("finance", "Investment kab karni chahiye?"),
    ("finance", "Property kharidni chahiye kya abhi?"),
    ("finance", "Kya mujhe lottery ya windfall milega?"),
    ("finance", "Financial loss kyun ho raha hai?"),
    ("finance", "Paisa bachana mushkil ho raha hai, kya hoga?"),
    ("finance", "Kya mera business profitable hoga?"),
    ("finance", "Kab tak financial stability aayegi?"),
    ("finance", "Kya share market mein invest karna chahiye?"),
    ("finance", "Meri income kab badhegi?"),
    ("finance", "Kya mujhe inheritance milegi?"),
    ("finance", "Financial problems kab khatam hongi?"),
    ("finance", "Kya mujhe partnership mein paisa milega?"),
    ("finance", "Ghar kharidne ka sahi time kab hai?"),
    ("finance", "Kya mera paisa safe hai?"),
    ("finance", "Savings kab badhegi meri?"),

    # Health (15 questions)
    ("health", "Meri health kab theek hogi?"),
    ("health", "Kya mujhe koi bimari hogi?"),
    ("health", "Health ke liye konsa time achha hai?"),
    ("health", "Meri sehat kyun kharab ho rahi hai?"),
    ("health", "Kya operation hoga mera?"),
    ("health", "Mental stress kab kam hoga?"),
    ("health", "Kya meri health improve hogi?"),
    ("health", "Bimari kab theek hogi?"),
    ("health", "Kya mujhe koi serious health problem hai?"),
    ("health", "Health ke liye kya remedy karni chahiye?"),
    ("health", "Meri energy kab wapas aayegi?"),
    ("health", "Kya mujhe hospital jaana padega?"),
    ("health", "Stress aur anxiety kab kam hogi?"),
    ("health", "Kya meri immunity weak hai?"),
    ("health", "Health problems kab khatam hongi?"),

    # Education (10 questions)
    ("education", "Kya meri padhai acchi hogi?"),
    ("education", "Exam mein pass hounga kya?"),
    ("education", "Kya mujhe foreign mein padhai ka mauka milega?"),
    ("education", "Padhai mein concentration kyun nahi ho raha?"),
    ("education", "Kya mujhe scholarship milegi?"),
    ("education", "Higher education ke liye sahi time kab hai?"),
    ("education", "Kya mujhe medical ya engineering mein jaana chahiye?"),
    ("education", "Exam results kab aayenge aur kaisa hoga?"),
    ("education", "Padhai mein success kab milegi?"),
    ("education", "Kya meri degree complete hogi?"),

    # Family/Children (10 questions)
    ("family", "Kya mujhe baccha hoga?"),
    ("family", "Bacche kab honge?"),
    ("family", "Kya mera pehla baccha beta hoga ya beti?"),
    ("family", "Family mein problems kyun aa rahi hain?"),
    ("family", "Maa-baap ki health kaisi rahegi?"),
    ("family", "Kya ghar mein shanti aayegi?"),
    ("family", "Bhai-behen se rishta kaisa rahega?"),
    ("family", "Kya mujhe ghar milega?"),
    ("family", "Family ka future kaisa hai?"),
    ("family", "Kya meri family financially stable hogi?"),

    # Emotional/Distress (15 questions)
    ("emotional", "Main bahut pareshan hun, kya hoga mera?"),
    ("emotional", "Kuch bhi sahi nahi ho raha, kyun?"),
    ("emotional", "Main bahut akela feel kar raha hun"),
    ("emotional", "Zindagi mein bahut mushkilein aa rahi hain"),
    ("emotional", "Kya mera bura waqt khatam hoga?"),
    ("emotional", "Main depression mein hun, kab theek hounga?"),
    ("emotional", "Sab kuch bigad raha hai mere liye"),
    ("emotional", "Kya meri kismat kharab hai?"),
    ("emotional", "Main bahut stressed hun career se"),
    ("emotional", "Relationship mein bahut takleef hai"),
    ("emotional", "Kya mere achhe din aayenge?"),
    ("emotional", "Bahut zyada tension hai, kya hoga?"),
    ("emotional", "Main haar gaya hun, kya karu?"),
    ("emotional", "Sab log mujhe dhoka dete hain"),
    ("emotional", "Kya meri life mein khushi aayegi?"),

    # Past events (10 questions)
    ("past_event", "2022 mein meri life mein kya hua?"),
    ("past_event", "2020 mein career mein kya hua tha?"),
    ("past_event", "Pichle saal itni problems kyun aayi?"),
    ("past_event", "2023 mein financial loss kyun hua?"),
    ("past_event", "Meri pehli naukri kab lagi thi?"),
    ("past_event", "2021 mein health kyun kharab hui thi?"),
    ("past_event", "Pichle 3 saal kaisa raha mera?"),
    ("past_event", "2019 se 2022 tak kya hua meri life mein?"),
    ("past_event", "Meri shaadi kab hui thi?"),
    ("past_event", "Pichle saal ka career kaisa raha?"),

    # Remedy (10 questions)
    ("remedy", "Career ke liye kya remedy karni chahiye?"),
    ("remedy", "Shaadi ke liye kya upay karein?"),
    ("remedy", "Financial problems ke liye kya karu?"),
    ("remedy", "Health ke liye kya remedy hai?"),
    ("remedy", "Shani ki dasha mein kya karna chahiye?"),
    ("remedy", "Rahu ki dasha mein kya upay karein?"),
    ("remedy", "Kya pehenna chahiye mere liye?"),
    ("remedy", "Konsa mantra jaapna chahiye?"),
    ("remedy", "Kya daan karna chahiye?"),
    ("remedy", "Luck badhane ke liye kya karein?"),

    # Simple factual (15 questions)
    ("simple", "Mera lagna kya hai?"),
    ("simple", "Mera rasi kya hai?"),
    ("simple", "Mera nakshatra kya hai?"),
    ("simple", "Mera naam kya hai?"),
    ("simple", "Abhi konsa dasha chal raha hai mera?"),
    ("simple", "Mera 7th cusp sub-lord kaun hai?"),
    ("simple", "Mera 10th cusp sub-lord kaun hai?"),
    ("simple", "Mera lagna lord kaun hai?"),
    ("simple", "Shukra mera konse ghar mein hai?"),
    ("simple", "Shani mera konse ghar mein hai?"),
    ("simple", "Mera mahadasha kab khatam hoga?"),
    ("simple", "Aaj ki date kya hai?"),
    ("simple", "Aap kaun hain?"),
    ("simple", "Mera janm kab hua?"),
    ("simple", "Meri kundali mein konsa yoga hai?"),
]

# ── System prompts ─────────────────────────────────────────────────────────────
TODAY = date.today().strftime("%d %B %Y")

CHOSEN_SYSTEM = f"""You are Jyotish, an experienced KP astrologer. Today is {TODAY}.

CRITICAL RULES — CHOSEN (ideal) response:
1. LANGUAGE: Question is in Hindi/Hinglish → Answer MUST be in Hindi/Hinglish. NEVER respond in English to a Hindi question.
2. LENGTH: 2-3 sentences MAX. Short, impactful, like a real astrologer talking.
3. NAME: Address the person by their first name + "ji" (e.g., "Arjun ji,"). Read name from chart.
4. DATES: Give SPECIFIC month-year dates from the dasha periods in the chart. E.g., "July 2026 se October 2026 tak".
5. TENSE: Today is {TODAY}. Dates before today = past tense ("yeh period beet chuka hai"). Dates after = future tense.
6. DASHA: Always cite the specific dasha period (e.g., "Moon-Mercury antardasha mein, jo Feb 2026 se Jul 2026 tak hai").
7. NO PRODUCTS: Only suggest products if the question explicitly asks for remedies/upay.
8. NO HEADERS: No "Analysis:", "Conclusion:", "Career Prediction:" etc.
9. NO METADATA: No rule IDs, no "KP_MAR_001", no "confidence: high".
10. CONVERSATIONAL: Sound like a real pandit speaking, not a robot.

EXAMPLE CHOSEN (for "Meri shaadi kab hogi?"):
"Arjun ji, aapki shaadi ki timing July 2026 se October 2026 ke beech dikh rahi hai — Moon-Ketu antardasha mein jab Venus pratyantar 7th cusp ko activate karega. 7th cusp sub-lord Mercury houses 2, 7, 11 signify karta hai jo marriage ke liye perfect combination hai. Jab samay aayega, rishta khud chalkar aayega."
"""

REJECTED_SYSTEM = f"""You are a bad KP astrology assistant. Today is {TODAY}.

Generate a BAD response that:
1. LANGUAGE MISMATCH: Question is in Hindi → Answer in ENGLISH (wrong language)
2. VAGUE: No specific dates, just "favorable period coming soon"
3. ROBOTIC: Use headers like "Marriage Analysis:", "Conclusion:"
4. DEFLECTING: Say "I would need to analyze further" or "pending deeper analysis"
5. WRONG TENSE: Treat past dates as future (e.g., say "Oct 2025 se shuru hoga" even though it's past)
6. PRODUCT SPAM: Recommend products even when not asked
7. METADATA LEAK: Include rule IDs like [KP_MAR_0660] or "confidence: medium"
8. VERBOSE: 4-5 paragraphs of generic astrology theory

EXAMPLE REJECTED (for "Meri shaadi kab hogi?"):
"Marriage Timing Analysis using KP Principles

Based on the given chart data and applying fundamental KP rule [KP_TIM_0660]: Marriage will take place during the conjoined periods of the significators for houses 2, 7 and 11.

The most promising dasha period would involve these combined significators operating simultaneously. Since current mahadasha belongs to Moon, which significantly connects to both 7th house through its natural rulership, there's strong potential for marriage timing within this period.

However, precise timing requires examining when other key significators like Jupiter or Rahu operate conjointly with favorable transit conditions. For accurate prediction, analyze upcoming anthras where multiple significators align.

Confidence: medium (pending deeper analysis connections). Our Shukra Kavach Pendant can help strengthen Venus for marriage."
"""

# ── OpenAI client ──────────────────────────────────────────────────────────────
from openai import OpenAI
client = OpenAI(api_key=api_key)

def generate_pair(chart: dict, category: str, question: str) -> dict | None:
    """Generate one (chosen, rejected) DPO pair."""
    user_msg = f"""Chart data (YAML format):
{chart['yaml']}

Question from {chart['name']} ({chart['gender']}, born {chart['dob']}):
{question}

Generate a response following all the rules in your system prompt."""

    try:
        # Generate chosen
        chosen_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": CHOSEN_SYSTEM},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        chosen = chosen_resp.choices[0].message.content.strip()

        # Generate rejected
        rejected_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": REJECTED_SYSTEM},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=300,
            temperature=0.8,
        )
        rejected = rejected_resp.choices[0].message.content.strip()

        # Quality filters
        if len(chosen) < 40:
            return None
        if len(rejected) < 40:
            return None
        # Chosen must be in Hindi/Hinglish (has Hindi words)
        hindi_kw = ['hai', 'mein', 'aapka', 'aapki', 'ke liye', 'karta', 'hoga', 'hogi',
                    'aur', 'se', 'tak', 'kab', 'kya', 'ji,', 'hun', 'hain']
        chosen_lower = chosen.lower()
        if not any(kw in chosen_lower for kw in hindi_kw):
            return None  # Chosen is not Hindi/Hinglish — skip

        return {
            "prompt": question,
            "chosen": chosen,
            "rejected": rejected,
            "category": category,
            "chart_name": chart["name"],
            "language": "hindi",
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


# ── Main generation loop ───────────────────────────────────────────────────────
def main():
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build work queue: (chart, category, question) combinations
    # Shuffle and repeat questions across charts to get 500 pairs
    work_queue = []
    for chart in CHARTS:
        for cat, q in HINDI_QUESTIONS:
            work_queue.append((chart, cat, q))

    random.shuffle(work_queue)

    # If we have fewer than args.count, repeat with different charts
    while len(work_queue) < args.count:
        extra = [(random.choice(CHARTS), cat, q) for cat, q in HINDI_QUESTIONS]
        work_queue.extend(extra)

    work_queue = work_queue[:args.count]
    print(f"\n🚀 Generating {len(work_queue)} Hindi DPO pairs...")
    print(f"   Charts: {len(CHARTS)}, Questions: {len(HINDI_QUESTIONS)}")
    print(f"   Output: {output_path}\n")

    pairs = []
    failed = 0

    # Load existing pairs to avoid duplicates
    existing_prompts = set()
    if output_path.exists():
        for line in open(output_path, encoding="utf-8"):
            try:
                p = json.loads(line)
                existing_prompts.add(p.get("prompt", "") + p.get("chart_name", ""))
            except:
                pass
        print(f"  Resuming: {len(existing_prompts)} pairs already done")

    # Open output file in append mode for resume support
    out_f = open(output_path, "a", encoding="utf-8")

    for i, (chart, cat, question) in enumerate(work_queue):
        # Skip if already done
        key = question + chart["name"]
        if key in existing_prompts:
            continue

        pair = generate_pair(chart, cat, question)
        if pair:
            pairs.append(pair)
            out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            out_f.flush()
            existing_prompts.add(key)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(work_queue)}] ✓ {len(pairs)} pairs generated, {failed} failed")
        else:
            failed += 1

        # Rate limit: small delay every 20 requests
        if (i + 1) % 20 == 0:
            time.sleep(1)

    out_f.close()

    print(f"\n✅ Done! Generated {len(pairs)} Hindi DPO pairs ({failed} failed/filtered)")
    print(f"   Output: {output_path}")
    print(f"\nNext step: python generate_hindi_dpo_500.py --merge")

    # Quick stats
    cats = {}
    for p in pairs:
        c = p.get("category", "unknown")
        cats[c] = cats.get(c, 0) + 1
    print("\nCategory breakdown:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
