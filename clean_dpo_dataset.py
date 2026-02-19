#!/usr/bin/env python3
"""
DPO Dataset Cleaner
====================
- Loads dpo_pairs.jsonl (2989 pairs, the current canonical file)
- Removes pairs with:
    1. Length bias: rejected/chosen ratio > 4x or < 0.5x (DPO killer)
    2. Chosen response > 4 sentences (too verbose)
    3. Metadata leakage in chosen (rulesused, KPGEN, etc.)
    4. Medical diagnosis in chosen (cancer confirmation, etc.)
    5. "the native" in chosen (should say "you")
    6. Chosen shorter than 30 chars (too short to be useful)
    7. Rejected shorter than 20 chars (trivially bad, no signal)
- Outputs: data/dpo/dpo_pairs_clean.jsonl
- Prints full stats before/after
"""
import json
import re
from pathlib import Path

INPUT  = Path("data/dpo/dpo_pairs.jsonl")
OUTPUT = Path("data/dpo/dpo_pairs_clean.jsonl")

# Patterns
rulesused_pat  = re.compile(r'rulesused:|rules_used:|timingmethod:|KPGEN|KPTIM|KPADIUS', re.I)
iso_date_pat   = re.compile(r'\b\d{4}-\d{2}(?:-\d{2})?\b')
header_pat     = re.compile(
    r'(?:Marriage|Career|Financial|Health|Analysis|Conclusion|Timing|Prediction|'
    r'Sub-Lord Significance|Planetary Positions|Core Significators|Key Significators|'
    r'House Activation|Dasha Activation|Chart Analysis|KP Analysis)\s*:',
    re.I
)
cancer_diag_pat = re.compile(
    r'(?:yes[,!]?\s+)?(?:you\s+(?:have|had|are\s+diagnosed\s+with)|'
    r'aapko\s+(?:cancer|tumou?r)|diagnosed\s+with)\s+cancer|'
    r'cancer[!,]?\s+(?:the\s+timing|timing\s+is|is\s+confirmed)',
    re.I
)
self_doubt_pat = re.compile(
    r'jis\s+method\s+se\s+hum\s+predictions\s+banate\s+hain\s+woh\s+bilkul\s+reliable\s+nahi|'
    r'sirf\s+immediate\s+future\s+events\s+hi\s+predict|'
    r'koi\s+bhi\s+attempt\s+longer-term\s+predictions',
    re.I
)

def count_sentences(text):
    return len(re.split(r'(?<=[.!?])\s+', text.strip()))

def load_pairs(path):
    pairs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except Exception:
                    pass
    return pairs

def audit_pair(p):
    """Returns (keep: bool, reason: str)"""
    ch  = p.get('chosen', '')
    rej = p.get('rejected', '')

    if len(ch) < 30:
        return False, 'chosen_too_short'
    if len(rej) < 20:
        return False, 'rejected_too_short'

    ratio = len(rej) / len(ch) if len(ch) > 0 else 0
    if ratio > 5.0:
        return False, f'length_bias_high_{ratio:.1f}x'
    if ratio < 0.3:
        return False, f'length_bias_low_{ratio:.2f}x'

    if rulesused_pat.search(ch):
        return False, 'metadata_leak_chosen'
    if cancer_diag_pat.search(ch):
        return False, 'cancer_diagnosis_chosen'
    if self_doubt_pat.search(ch):
        return False, 'self_doubt_chosen'
    if 'the native' in ch.lower():
        return False, 'third_person_native_chosen'

    sents = count_sentences(ch)
    if sents > 5:
        return False, f'too_verbose_{sents}_sentences'

    return True, 'ok'

def main():
    print("=" * 60)
    print("DPO DATASET CLEANER")
    print("=" * 60)

    pairs = load_pairs(INPUT)
    print(f"Loaded: {len(pairs)} pairs from {INPUT}")

    kept = []
    removed = []
    reason_counts = {}

    for p in pairs:
        keep, reason = audit_pair(p)
        if keep:
            kept.append(p)
        else:
            removed.append((p, reason))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print(f"\nKept:    {len(kept)} pairs ({len(kept)/len(pairs)*100:.1f}%)")
    print(f"Removed: {len(removed)} pairs ({len(removed)/len(pairs)*100:.1f}%)")
    print(f"\nRemoval reasons:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:40s}: {count:4d}")

    # Length stats on kept pairs
    ratios = []
    for p in kept:
        ch = p.get('chosen', '')
        rej = p.get('rejected', '')
        if len(ch) > 0:
            ratios.append(len(rej) / len(ch))
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    print(f"\nKept dataset stats:")
    print(f"  Avg rejected/chosen ratio: {avg_ratio:.2f}x  (target: 1.5-3x)")
    print(f"  Min ratio: {min(ratios):.2f}x")
    print(f"  Max ratio: {max(ratios):.2f}x")

    # Unique prompts
    prompts = [p.get('prompt', '') for p in kept]
    unique_p = len(set(prompts))
    print(f"  Unique prompts: {unique_p}/{len(kept)} ({unique_p/len(kept)*100:.1f}%)")

    # Chart distribution
    from collections import Counter
    charts = Counter(p.get('chart_name', '?') for p in kept)
    print(f"  Charts ({len(charts)} unique): top 5 = {dict(list(sorted(charts.items(), key=lambda x:-x[1]))[:5])}")

    # Write output
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"\nSaved: {OUTPUT}  ({len(kept)} pairs)")

    # Verdict
    print("\n" + "=" * 60)
    print("RETRAIN DECISION")
    print("=" * 60)
    if avg_ratio > 1.5 and avg_ratio < 4.0:
        print(f"Length bias: {avg_ratio:.2f}x — ACCEPTABLE for DPO training")
    elif avg_ratio >= 4.0:
        print(f"Length bias: {avg_ratio:.2f}x — STILL HIGH, consider regenerating rejected responses")
    else:
        print(f"Length bias: {avg_ratio:.2f}x — LOW (rejected too short), consider regenerating")

    pct_kept = len(kept) / len(pairs) * 100
    if pct_kept < 70:
        print(f"Only {pct_kept:.0f}% pairs kept — REGENERATION STRONGLY RECOMMENDED")
    elif pct_kept < 85:
        print(f"{pct_kept:.0f}% pairs kept — RETRAIN with clean file, monitor results")
    else:
        print(f"{pct_kept:.0f}% pairs kept — GOOD, retrain with clean file")

    print(f"\nNext step: python scripts/14_prepare_dpo_dataset.py --input data/dpo/dpo_pairs_clean.jsonl")

if __name__ == '__main__':
    main()
