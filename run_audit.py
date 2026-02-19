#!/usr/bin/env python3
"""Full DPO dataset audit — runs on all files in data/dpo/"""
import json, re, os
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/dpo")

files = {
    "dpo_pairs.jsonl":        DATA_DIR / "dpo_pairs.jsonl",
    "dpo_pairs_backup.jsonl": DATA_DIR / "dpo_pairs_backup.jsonl",
    "dpo_pairs_merged.jsonl": DATA_DIR / "dpo_pairs_merged.jsonl",
    "dpo_pairs_v2_fixes.jsonl": DATA_DIR / "dpo_pairs_v2_fixes.jsonl",
}

header_pat   = re.compile(r'(?:Marriage|Career|Financial|Health|Analysis|Conclusion|Timing|Prediction)\s*:', re.I)
iso_pat      = re.compile(r'\b\d{4}-\d{2}(?:-\d{2})?\b')
rulesused_pat= re.compile(r'rulesused:|rules_used:|timingmethod:|KPGEN|KPTIM', re.I)
cancer_med_pat = re.compile(
    r'(?:do\s+you\s+have|yes\s+you\s+have|you\s+have)\s+cancer|'
    r'cancer[- ](?:related|treatment|risk|diagnosis|patient|surgery|therapy|cells?)|'
    r'(?:breast|lung|blood|skin|colon|prostate|ovarian|cervical)\s+cancer',
    re.I
)

print("=" * 70)
print("DPO DATASET FULL AUDIT")
print("=" * 70)

for label, fpath in files.items():
    if not fpath.exists():
        print(f"\n{label}: FILE NOT FOUND")
        continue

    pairs = []
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except Exception:
                    pass

    print(f"\n{'='*50}")
    print(f"FILE: {label}  ({len(pairs)} pairs)")
    print(f"{'='*50}")

    if not pairs:
        print("  EMPTY FILE")
        continue

    # Chart distribution
    charts = Counter(p.get("chart_name", p.get("chart", "?")) for p in pairs)
    print(f"  Charts ({len(charts)} unique): {dict(sorted(charts.items(), key=lambda x:-x[1]))}")

    # Category distribution
    cats = Counter(p.get("category", "?") for p in pairs)
    top_cats = sorted(cats.items(), key=lambda x: -x[1])[:10]
    print(f"  Top categories: {top_cats}")

    # Quality checks on chosen responses
    rulesused = iso_dates = headers = native = cancer_med = long_resp = 0
    short_rej = length_ratio_sum = 0

    for p in pairs:
        ch  = p.get("chosen", "")
        rej = p.get("rejected", "")

        if rulesused_pat.search(ch):  rulesused  += 1
        if iso_pat.search(ch):        iso_dates  += 1
        if header_pat.search(ch):     headers    += 1
        if "the native" in ch.lower(): native    += 1
        if cancer_med_pat.search(ch): cancer_med += 1

        sents = len(re.split(r'(?<=[.!?])\s+', ch.strip()))
        if sents > 4: long_resp += 1

        if len(ch) > 0 and len(rej) > 0:
            ratio = len(rej) / len(ch)
            length_ratio_sum += ratio
            if ratio < 0.5: short_rej += 1

    n = len(pairs)
    avg_ratio = length_ratio_sum / n if n else 0

    print(f"\n  CHOSEN QUALITY ISSUES:")
    print(f"    rulesused/KPGEN metadata leaked : {rulesused:4d} ({rulesused/n*100:.1f}%)")
    print(f"    ISO dates (2025-10 format)      : {iso_dates:4d} ({iso_dates/n*100:.1f}%)")
    print(f"    Robotic headers leaked          : {headers:4d} ({headers/n*100:.1f}%)")
    print(f"    'the native' (not 'you')        : {native:4d} ({native/n*100:.1f}%)")
    print(f"    Medical cancer diagnosis        : {cancer_med:4d} ({cancer_med/n*100:.1f}%)")
    print(f"    >4 sentences (too long)         : {long_resp:4d} ({long_resp/n*100:.1f}%)")
    print(f"    Rejected too short (<0.5x)      : {short_rej:4d} ({short_rej/n*100:.1f}%)")
    print(f"    Avg rejected/chosen ratio       : {avg_ratio:.2f}x  (target: 1.5-3x)")

    # Duplicate prompts
    prompts = [p.get("prompt", "") for p in pairs]
    unique_p = len(set(prompts))
    print(f"\n  DIVERSITY:")
    print(f"    Unique prompts: {unique_p}/{n} ({unique_p/n*100:.1f}%)")

    # Sample 3 bad chosen responses
    bad = [p for p in pairs if rulesused_pat.search(p.get("chosen",""))]
    if bad:
        print(f"\n  SAMPLE metadata-leaked chosen responses:")
        for p in bad[:3]:
            print(f"    Q: {p.get('prompt','')[:80]}")
            print(f"    A: {p.get('chosen','')[:120]}")
            print()

print("\n" + "="*70)
print("VERDICT")
print("="*70)

# Load main file for final verdict
main_pairs = [json.loads(l) for l in open(DATA_DIR/"dpo_pairs.jsonl", encoding="utf-8") if l.strip()]
n = len(main_pairs)
ru = sum(1 for p in main_pairs if rulesused_pat.search(p.get("chosen","")))
iso = sum(1 for p in main_pairs if iso_pat.search(p.get("chosen","")))
hdr = sum(1 for p in main_pairs if header_pat.search(p.get("chosen","")))
lng = sum(1 for p in main_pairs if len(re.split(r'(?<=[.!?])\s+', p.get("chosen","").strip())) > 4)

total_bad = ru + iso + hdr + lng
bad_pct = total_bad / n * 100

print(f"Main dataset: {n} pairs")
print(f"Total bad chosen responses: ~{total_bad} issues across {n} pairs")
print(f"Contamination rate: {bad_pct:.1f}%")

if bad_pct > 20:
    print("\n>>> VERDICT: RETRAIN REQUIRED — dataset contamination too high")
    print("    Action: clean dataset + regenerate bad pairs + retrain DPO")
elif bad_pct > 10:
    print("\n>>> VERDICT: RETRAIN RECOMMENDED — significant contamination")
    print("    Action: filter bad pairs, retrain with clean subset")
else:
    print("\n>>> VERDICT: DATASET ACCEPTABLE — postprocessing fixes sufficient for now")
    print("    Action: apply postprocessing fixes, monitor in production")
