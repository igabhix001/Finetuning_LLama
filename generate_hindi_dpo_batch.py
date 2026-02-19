"""
Hindi DPO Batch Generator — OpenAI Batch API
Generates 500 Hindi/Hinglish DPO pairs to fix language mismatch.

Usage:
  python generate_hindi_dpo_batch.py --submit        # Build + submit batch
  python generate_hindi_dpo_batch.py --check         # Check status
  python generate_hindi_dpo_batch.py --download      # Download + merge
  python generate_hindi_dpo_batch.py --submit --wait # Submit + auto-wait + download
"""
import os, sys, json, time, random, glob, shutil, argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--submit", action="store_true")
parser.add_argument("--check", action="store_true")
parser.add_argument("--download", action="store_true")
parser.add_argument("--wait", action="store_true")
parser.add_argument("--count", type=int, default=500)
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--output-dir", type=str, default="data/dpo")
parser.add_argument("--merge-into", type=str, default="data/dpo/dpo_pairs_final.jsonl")
args = parser.parse_args()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found in .env"); sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    os.system(f"{sys.executable} -m pip install openai>=1.30.0 -q")
    from openai import OpenAI

client = OpenAI(api_key=api_key)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
META_FILE  = output_dir / "hindi_batch_meta.json"
PAIRS_FILE = output_dir / "dpo_hindi_500.jsonl"

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
try:
    from chart_preprocessor import chart_to_yaml
except ImportError:
    print("❌ chart_preprocessor not found. Run from Finetuning_LLama/ directory."); sys.exit(1)

# ── Load charts ───────────────────────────────────────────────────────────────
def _load_charts():
    patterns = [
        str(Path(__file__).parent / "sample_kundali" / "kundali_*.json"),
        "/workspace/Finetuning_LLama/sample_kundali/kundali_*.json",
    ]
    found = set()
    for p in patterns:
        for fp in glob.glob(p): found.add(os.path.abspath(fp))
    charts = []
    for fp in sorted(found):
        try:
            raw = open(fp, encoding="utf-8").read()
            data = json.loads(raw)
            bd = data.get("birthDetails", {})
            charts.append({"name": data.get("name","?"), "gender": data.get("gender",""),
                           "dob": bd.get("date","?"), "yaml": chart_to_yaml(raw)})
            print(f"  ✓ {os.path.basename(fp)}: {data.get('name')}")
        except Exception as e:
            print(f"  ✗ {fp}: {e}")
    if not charts: print("❌ No kundali files found."); sys.exit(1)
    return charts

# ── Hindi question bank ───────────────────────────────────────────────────────
HINDI_QUESTIONS = [
    ("marriage","Meri shaadi kab hogi?"),
    ("marriage","Kya meri shaadi is saal hogi?"),
    ("marriage","Shaadi mein itni der kyun ho rahi hai?"),
    ("marriage","Mere liye shaadi ka sabse accha time kab hai?"),
    ("marriage","Kya love marriage hogi ya arranged?"),
    ("marriage","Kya mere chart mein vivah yoga hai?"),
    ("marriage","Shaadi mein koi rukawat hai kya?"),
    ("marriage","Mera rishta kyun nahi ban raha?"),
    ("marriage","Kya mujhe achha life partner milega?"),
    ("marriage","Kitni umar mein shaadi hogi meri?"),
    ("marriage","Kya meri shaadi ke baad zindagi khushhal rahegi?"),
    ("marriage","Kya is saal koi rishta pakka hoga?"),
    ("marriage","Meri shaadi mein Mangal dosha ka asar hai kya?"),
    ("marriage","Mere 7th house mein kya hai jo shaadi rok raha hai?"),
    ("marriage","Kya mujhe apne sheher mein ya bahar shaadi hogi?"),
    ("career","Meri naukri kab lagegi?"),
    ("career","Kya mujhe sarkari naukri milegi?"),
    ("career","Kab tak promotion milega?"),
    ("career","Kya mera business chalega?"),
    ("career","Main bahut pareshan hun career se, kya hoga?"),
    ("career","Mujhe job change karni chahiye ya nahi?"),
    ("career","Kya mujhe apna business shuru karna chahiye?"),
    ("career","Meri naukri mein koi problem aa rahi hai, kab theek hogi?"),
    ("career","Kya videsh mein kaam karne ka yoga hai mere chart mein?"),
    ("career","Mera interview hai, kya result accha hoga?"),
    ("career","Mujhe konsa profession choose karna chahiye?"),
    ("career","Kya partnership business mujhe suit karega?"),
    ("career","Meri job mein bahut tension hai, kab sudhrega?"),
    ("career","Kya is saal koi bada career break milega?"),
    ("career","Kab tak meri financial condition stable hogi career mein?"),
    ("financial","Meri income kab badhegi?"),
    ("financial","Kab tak mera karz utrega?"),
    ("financial","Kya property mein invest karna sahi rahega?"),
    ("financial","Kya mujhe paisa milega is saal?"),
    ("financial","Mera business loss mein hai, kab profit hoga?"),
    ("financial","Kya share market mein invest karna theek hai abhi?"),
    ("financial","Kab tak meri financial situation improve hogi?"),
    ("financial","Kya mujhe ghar kharidna chahiye is saal?"),
    ("financial","Mujhe loan milega kya?"),
    ("financial","Kya mujhe sudden dhana labh hoga?"),
    ("financial","Meri savings kyun nahi ho pa rahi?"),
    ("financial","Paisa aata hai lekin rukta nahi, kyun?"),
    ("financial","Kab tak meri financial problems khatam hongi?"),
    ("financial","Is saal koi bada financial gain hoga kya?"),
    ("financial","Kya gold mein invest karna sahi hai mere liye?"),
    ("health","Meri tabiyat theek kab hogi?"),
    ("health","Kya meri surgery safe rahegi?"),
    ("health","Mujhe bahut stress ho raha hai, chart mein kya dikh raha hai?"),
    ("health","Kya meri health mein improvement hoga?"),
    ("health","Mujhe neend nahi aati, koi planetary reason hai kya?"),
    ("health","Mere papa ki health theek nahi hai, kab sudhrega?"),
    ("health","Kya meri bimari theek ho jayegi?"),
    ("health","Kab tak ye health problem khatam hogi?"),
    ("health","Mujhe baar baar bimari kyun aati hai?"),
    ("health","Kya is saal health ke liye koi bada risk hai?"),
    ("obstacles","Mere saath bura kyun hota hai?"),
    ("obstacles","Kya mere chart mein koi dosha hai?"),
    ("obstacles","Main bahut unlucky feel karta/karti hun, kyun?"),
    ("obstacles","Kya mujh par kisi ki nazar lag gayi hai?"),
    ("obstacles","Kya mere chart mein Kaal Sarp Dosha hai?"),
    ("obstacles","Sab kuch theek tha, achanak sab bigad gaya — kyun?"),
    ("obstacles","Kya mere chart mein Mangal Dosha hai?"),
    ("obstacles","Meri mehnat ka fal kyun nahi milta?"),
    ("obstacles","Kya Shani mujhe pareshan kar raha hai?"),
    ("obstacles","Har kaam last moment mein kyun bigad jaata hai?"),
    ("obstacles","Kya mere chart mein Rahu dosha hai?"),
    ("obstacles","Main depression mein hun, chart mein kya dikh raha hai?"),
    ("obstacles","Kya ye mushkil waqt jaldi khatam hoga?"),
    ("obstacles","Kya mere chart mein Pitra Dosha hai?"),
    ("obstacles","Mujhe lagta hai main bahut akela/akeli hun, kya chart mein kuch hai?"),
    ("remedies","Kya koi upay hai meri naukri ke liye?"),
    ("remedies","Shaadi ke liye kya upay karun?"),
    ("remedies","Konsa gemstone mujhe pehnna chahiye?"),
    ("remedies","Kya Rudraksha pehnna sahi rahega?"),
    ("remedies","Career ke liye kaunsa puja karun?"),
    ("remedies","Venus ko strong karne ke liye kya karun?"),
    ("remedies","Paison ki problem ke liye kya upay hai?"),
    ("remedies","Shani ke liye kya upay karun?"),
    ("remedies","Rahu ke bure asar ko kaise kam karun?"),
    ("remedies","Kya mujhe mandir jaana chahiye, kaunsa?"),
    ("simple","Mera lagna kya hai?"),
    ("simple","Meri rashi kya hai?"),
    ("simple","Mera nakshatra kya hai?"),
    ("simple","Abhi meri kaunsi mahadasha chal rahi hai?"),
    ("simple","Mera lagna lord kaun hai?"),
    ("simple","Mere 7th house ka sub-lord kaun hai?"),
    ("simple","Mere 10th house ka sub-lord kaun hai?"),
    ("simple","Abhi kaunsa antardasha chal raha hai?"),
    ("simple","Mera janm nakshatra kya hai?"),
    ("simple","Mere chart mein Venus kahan hai?"),
    ("past_event","2020 mein mere saath kya hua tha?"),
    ("past_event","2022 mein meri zindagi mein kya bada change aaya?"),
    ("past_event","Pichle 2 saal mein itni mushkilein kyun aayi?"),
    ("past_event","2019 mein jo hua tha, uska kya karan tha?"),
    ("past_event","Pichle saal meri job kyun gayi?"),
    ("past_event","2021 mein meri health kyun kharab hui thi?"),
    ("past_event","Pichle 3 saal bahut bure the, kyun?"),
    ("past_event","2023 mein mera rishta kyun toot gaya?"),
    ("past_event","Pichle saal financial loss kyun hua?"),
    ("past_event","2018 se 2020 ke beech kya chal raha tha mere chart mein?"),
]

# ── Prompts ───────────────────────────────────────────────────────────────────
CHOSEN_SYSTEM = """You are Jyotish, an expert KP astrologer. The user asked in Hindi/Hinglish.
RULES (strictly follow):
1. Respond in Hindi/Hinglish ONLY — NOT in English
2. Address person by name + "ji" (e.g., "Arjun Kapoor ji,")
3. Give SPECIFIC dates from the chart dasha data (e.g., "July 2026 se November 2026 tak")
4. Maximum 3 sentences — concise and direct
5. Cite specific Mahadasha-Antardasha period with dates
6. NO markdown, NO bullet points, NO headers
7. NO filler like "chart ke according" or "planetary positions ke basis par"
8. Warm pandit tone — not robotic

GOOD: "Arjun Kapoor ji, aapki shaadi ka sabse accha waqt July 2026 se November 2026 ke beech hai, jab Mercury-Jupiter antardasha chal raha hoga. Is period mein Venus pratyantar houses 2, 7, 11 ko activate karega jo vivah ke liye bahut favorable hai."
BAD: "Based on your chart, marriage is possible. The 7th house indicates favorable conditions."
"""

REJECTED_SYSTEM = """You are an astrology chatbot. The user asked in Hindi but you respond in English.
Generate a BAD response that:
1. Responds in English even though question was in Hindi
2. Is vague — no specific dates, no specific dasha periods  
3. Uses robotic phrases: "Based on planetary positions", "The chart indicates", "the native"
4. Is verbose — 4-5 sentences of filler
5. Ends with deflection: "I would recommend consulting further" or "further analysis needed"
Keep it 3-5 sentences.
"""

CHOSEN_TMPL = "Chart data for {name} ({gender}, born {dob}):\n{yaml}\n\nQuestion (Hindi/Hinglish): {question}\n\nRespond in Hindi/Hinglish with specific dates. Max 3 sentences."
REJECTED_TMPL = "Chart data for {name} ({gender}, born {dob}):\n{yaml}\n\nQuestion (Hindi/Hinglish): {question}\n\nRespond in English (bad response — vague, no dates, robotic). 3-5 sentences."

# ── Build combos ──────────────────────────────────────────────────────────────
def build_combos(charts, count):
    all_pairs = [(c, q) for c in charts for q in HINDI_QUESTIONS]
    random.shuffle(all_pairs)
    seen, combos = set(), []
    for chart, (cat, q) in all_pairs:
        key = (chart["name"], q)
        if key not in seen:
            seen.add(key)
            combos.append({"chart": chart, "category": cat, "question": q})
        if len(combos) >= count: break
    print(f"✓ Built {len(combos)} unique combos from {len(charts)} charts × {len(HINDI_QUESTIONS)} questions")
    return combos

def build_requests(combos, model):
    reqs = []
    for i, c in enumerate(combos):
        ch, q = c["chart"], c["question"]
        for kind, sys_p, tmpl in [("chosen", CHOSEN_SYSTEM, CHOSEN_TMPL), ("rejected", REJECTED_SYSTEM, REJECTED_TMPL)]:
            reqs.append({
                "custom_id": f"hindi_{kind}_{i:04d}",
                "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": model, "max_tokens": 300, "temperature": 0.7 if kind=="chosen" else 0.8,
                         "messages": [{"role":"system","content":sys_p},
                                      {"role":"user","content":tmpl.format(name=ch["name"],gender=ch["gender"],dob=ch["dob"],yaml=ch["yaml"],question=q)}]}
            })
    print(f"✓ Built {len(reqs)} requests ({len(combos)} chosen + {len(combos)} rejected)")
    return reqs

# ── Submit ────────────────────────────────────────────────────────────────────
def submit_batch(combos, model):
    reqs = build_requests(combos, model)
    batch_file = output_dir / "hindi_batch_requests.jsonl"
    with open(batch_file, "w", encoding="utf-8") as f:
        for r in reqs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ Wrote {len(reqs)} requests → {batch_file}")
    with open(batch_file, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"✓ Uploaded: {file_obj.id}")
    batch = client.batches.create(input_file_id=file_obj.id, endpoint="/v1/chat/completions",
                                   completion_window="24h",
                                   metadata={"description": f"Hindi DPO 500 — {datetime.now().strftime('%Y-%m-%d')}"})
    print(f"✓ Batch created: {batch.id}  status={batch.status}")
    meta = {"batch_id": batch.id, "file_id": file_obj.id, "model": model,
            "count": len(combos), "submitted_at": datetime.now().isoformat(),
            "combos": [{"name":c["chart"]["name"],"category":c["category"],"question":c["question"]} for c in combos]}
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Metadata → {META_FILE}")
    print(f"\n📋 Next:  python generate_hindi_dpo_batch.py --check")
    return batch.id

# ── Check ─────────────────────────────────────────────────────────────────────
def check_status():
    if not META_FILE.exists(): print("❌ No metadata. Run --submit first."); sys.exit(1)
    meta = json.loads(META_FILE.read_text())
    b = client.batches.retrieve(meta["batch_id"])
    rc = b.request_counts
    print(f"Batch:  {b.id}")
    print(f"Status: {b.status}")
    if rc: print(f"Requests: {rc.completed} done / {rc.failed} failed / {rc.total} total")
    if b.output_file_id: print(f"Output:  {b.output_file_id}")
    return b.status

# ── Download + pair + merge ───────────────────────────────────────────────────
def _is_hindi(text):
    if any('\u0900' <= c <= '\u097F' for c in text): return True
    markers = ["aap","hai","hoga","kya","mein","se","ke","ka","ji,","hain","kab","kyun","nahi","bahut"]
    return sum(1 for m in markers if m in text.lower()) >= 3

def download_and_pair():
    if not META_FILE.exists(): print("❌ No metadata. Run --submit first."); sys.exit(1)
    meta = json.loads(META_FILE.read_text())
    b = client.batches.retrieve(meta["batch_id"])
    if b.status != "completed":
        print(f"❌ Not completed. Status: {b.status}")
        rc = b.request_counts
        if rc: print(f"   {rc.completed}/{rc.total} done")
        sys.exit(1)
    content = client.files.content(b.output_file_id).text.strip().split("\n")
    print(f"✓ Downloaded {len(content)} lines")
    chosen_map, rejected_map, errors = {}, {}, 0
    for line in content:
        if not line.strip(): continue
        try:
            r = json.loads(line)
            cid = r.get("custom_id","")
            text = (r.get("response",{}).get("body",{}).get("choices",[{}])[0]
                     .get("message",{}).get("content","")).strip()
            if not text: errors += 1; continue
            idx = int(cid.split("_")[-1])
            if "chosen" in cid: chosen_map[idx] = text
            elif "rejected" in cid: rejected_map[idx] = text
        except: errors += 1
    print(f"✓ Parsed: {len(chosen_map)} chosen / {len(rejected_map)} rejected / {errors} errors")
    combos = meta["combos"]
    pairs, skip_lang, skip_short, skip_same = [], 0, 0, 0
    for i, combo in enumerate(combos):
        ch, rej = chosen_map.get(i,""), rejected_map.get(i,"")
        if not ch or not rej: continue
        if len(ch) < 50 or len(rej) < 30: skip_short += 1; continue
        if not _is_hindi(ch): skip_lang += 1; continue
        if ch[:80].lower() == rej[:80].lower(): skip_same += 1; continue
        pairs.append({"prompt": f"[CHART: {combo['name']}]\n\nUser: {combo['question']}",
                      "chosen": ch, "rejected": rej,
                      "category": combo["category"], "language": "hindi",
                      "source": "hindi_dpo_batch_v1"})
    print(f"\n📊 Filter: {len(pairs)} valid | {skip_lang} not-Hindi | {skip_short} short | {skip_same} identical")
    if not pairs: print("❌ No valid pairs."); sys.exit(1)
    PAIRS_FILE.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
    print(f"✓ Saved {len(pairs)} pairs → {PAIRS_FILE}")
    merge_path = Path(args.merge_into)
    existing = []
    if merge_path.exists():
        existing = [json.loads(l) for l in merge_path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        backup = merge_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        shutil.copy(merge_path, backup)
        print(f"✓ Backed up → {backup.name}")
    merged = existing + pairs
    merge_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in merged), encoding="utf-8")
    print(f"✓ Merged → {merge_path}: {len(merged)} total ({len(pairs)} new Hindi + {len(existing)} existing)")
    cats = Counter(p["category"] for p in pairs)
    print("\n📊 Category breakdown:")
    for cat, cnt in cats.most_common(): print(f"  {cat:15s}: {cnt}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not any([args.submit, args.check, args.download]):
        parser.print_help()
        print("\n💡 Quick start:\n  python generate_hindi_dpo_batch.py --submit\n  python generate_hindi_dpo_batch.py --check\n  python generate_hindi_dpo_batch.py --download")
        sys.exit(0)

    if args.submit:
        print("Loading charts...")
        charts = _load_charts()
        combos = build_combos(charts, args.count)
        batch_id = submit_batch(combos, args.model)
        if args.wait:
            print(f"\n⏳ Polling batch {batch_id} every 60s...")
            while True:
                time.sleep(60)
                b = client.batches.retrieve(batch_id)
                rc = b.request_counts
                done = rc.completed if rc else "?"
                total = rc.total if rc else "?"
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] {b.status} — {done}/{total}")
                if b.status in ("completed", "failed", "expired", "cancelled"):
                    break
            if b.status == "completed":
                download_and_pair()
            else:
                print(f"❌ Batch ended with status: {b.status}")

    if args.check:
        check_status()

    if args.download and not args.wait:
        download_and_pair()

if __name__ == "__main__":
    main()
