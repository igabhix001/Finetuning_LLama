#!/usr/bin/env python3
"""
Industry-Standard DPO Dataset Quality Audit
============================================
Based on research from:
- NeurIPS 2025: "Less is More: Improving LLM Alignment via Preference Data Selection" (BeeS)
- NeurIPS 2024: "Unpacking DPO and PPO" (Ivison et al.)
- "What Matters in Data for DPO?" (arXiv 2508.18312)
- "Small-Margin Preferences Still Matter" (MixDPO, arXiv 2602.00954)

Metrics tracked:
1. Reward Margin (proxy via length-ratio + rule-based scoring)
2. Length Bias Detection (correlation between length and win)
3. Duplicate/Near-Duplicate Prompt Detection
4. Semantic Cluster Entropy (intent diversity)
5. Flip Inconsistency Rate (same prompt, contradictory preferences)
6. Chosen Quality Score (domain-specific rubric)
7. Rejected Badness Score (must be clearly worse)
8. Pair-level filtering recommendations
"""

import json
import re
import math
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_pairs(path: str) -> list:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
                p["_idx"] = i
                pairs.append(p)
            except json.JSONDecodeError:
                print(f"  ⚠ Skipping malformed line {i}")
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 1: REWARD MARGIN (proxy)
# ═══════════════════════════════════════════════════════════════════════════════
# Without a trained reward model, we use a rule-based proxy score.
# This approximates what a reward model would assign.

def score_response(text: str, category: str, prompt: str = "") -> dict:
    """Rule-based quality score (0-50 scale, matching our JUDGE_RUBRIC)."""
    scores = {}

    # 1. Language correctness (0-5)
    prompt_is_hindi = bool(re.search(r'[कखगघ]|kab|kya|meri|mera|hogi|hoga|kaise|kaisa', prompt.lower()))
    resp_hindi_markers = sum(1 for w in ["hai", "hain", "aapke", "aapki", "karta", "hoga", "mein ", "karein"]
                            if w in text.lower())
    if prompt_is_hindi:
        scores["language"] = min(5, resp_hindi_markers)  # Hindi Q should get Hindi response
    else:
        scores["language"] = max(0, 5 - resp_hindi_markers)  # English Q should get English response

    # 2. Date format (0-5)
    iso_dates = len(re.findall(r"\d{4}-\d{2}(?![\d-])", text))
    readable_dates = len(re.findall(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}", text))
    if iso_dates > 0:
        scores["date_format"] = 0
    elif readable_dates > 0:
        scores["date_format"] = 5
    else:
        scores["date_format"] = 3  # No dates needed or mentioned

    # 3. Justification (0-5)
    has_sublord = "sub-lord" in text.lower() or "sub lord" in text.lower() or "sublord" in text.lower()
    has_cusp = "cusp" in text.lower()
    has_house = bool(re.search(r"house[s]?\s*\d", text.lower())) or "signif" in text.lower()
    if category in ("general", "simple_factual", "safety", "follow_up"):
        scores["justification"] = 4  # Not required for these
    elif (has_sublord or has_cusp) and has_house:
        scores["justification"] = 5
    elif has_house:
        scores["justification"] = 3
    else:
        scores["justification"] = 1

    # 4. Tense correctness (0-5) — check for obvious errors
    # Past dates described as future
    tense_err = len(re.findall(r"(?:will|upcoming|starting from|begins? in)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+202[0-4]", text, re.IGNORECASE))
    scores["tense"] = max(0, 5 - tense_err * 3)

    # 5. Age plausibility (0-5)
    has_age = bool(re.search(r"(?:age|aged?|you.d be|~\d{2}|\d{2}\s*(?:years|yrs)|saal ke)", text, re.IGNORECASE))
    if category in ("general", "simple_factual", "safety", "follow_up"):
        scores["age"] = 4
    elif has_age:
        scores["age"] = 5
    else:
        scores["age"] = 1

    # 6. Timing precision (0-5)
    has_pratyantar = "pratyantar" in text.lower()
    has_month = bool(re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}", text))
    if category in ("general", "simple_factual", "safety", "follow_up"):
        scores["timing"] = 4
    elif has_pratyantar and has_month:
        scores["timing"] = 5
    elif has_month:
        scores["timing"] = 3
    else:
        scores["timing"] = 1

    # 7. Tone (0-5)
    robotic = ["**", "Analysis:", "Conclusion:", "Confidence:", "Critical Finding:", "## ", "### ", "1.", "2."]
    warm = ["ji", "don't worry", "understand", "chinta mat", "remember"]
    robotic_count = sum(1 for r in robotic if r in text)
    warm_count = sum(1 for w in warm if w in text.lower())
    scores["tone"] = max(0, min(5, 3 + warm_count - robotic_count * 2))

    # 8. Product discipline (0-5)
    product_words = ["rudraksha", "bracelet", "pendant", "kavach", "mala", "necklace",
                     "try karein", "hamara", "wear our", "package", "consultation package"]
    has_product = any(w in text.lower() for w in product_words)
    if category == "remedies":
        scores["product"] = 4 if has_product else 3
    else:
        scores["product"] = 0 if has_product else 5

    # 9. Format compliance (0-5)
    sent_count = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    has_markdown = "**" in text or "##" in text or "- " in text
    has_newlines = "\n" in text
    if has_markdown:
        scores["format"] = 0
    elif has_newlines:
        scores["format"] = 2
    elif sent_count <= 4:
        scores["format"] = 5
    elif sent_count <= 6:
        scores["format"] = 3
    else:
        scores["format"] = 1

    # 10. Factual grounding (0-5) — does it reference chart data?
    has_dasha_ref = bool(re.search(r"(?:mahadasha|antardasha|pratyantar|AD\b)", text, re.IGNORECASE))
    has_planet = bool(re.search(r"\b(?:Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu)\b", text))
    if category in ("general",):
        scores["grounding"] = 4
    elif has_dasha_ref and has_planet:
        scores["grounding"] = 5
    elif has_dasha_ref or has_planet:
        scores["grounding"] = 3
    else:
        scores["grounding"] = 1

    scores["total"] = sum(scores.values())
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 2: LENGTH BIAS DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def length_bias_analysis(pairs: list) -> dict:
    """Check if chosen is always shorter/longer — DPO can learn length as proxy."""
    chosen_lens = [len(p["chosen"]) for p in pairs]
    rejected_lens = [len(p["rejected"]) for p in pairs]

    # Pearson correlation between (chosen_len - rejected_len) and "win"
    # Since chosen always "wins", we check if length predicts the label
    chosen_longer = sum(1 for c, r in zip(chosen_lens, rejected_lens) if c > r)
    rejected_longer = sum(1 for c, r in zip(chosen_lens, rejected_lens) if r > c)

    # Length ratio
    ratios = [r / max(c, 1) for c, r in zip(chosen_lens, rejected_lens)]
    avg_ratio = sum(ratios) / len(ratios)

    # Correlation: if rejected is ALWAYS longer, model learns "shorter = better"
    # which is a spurious signal
    n = len(pairs)
    mean_c = sum(chosen_lens) / n
    mean_r = sum(rejected_lens) / n

    # Length difference correlation with preference
    # +1 = chosen wins, length_diff = chosen_len - rejected_len
    diffs = [c - r for c, r in zip(chosen_lens, rejected_lens)]
    mean_diff = sum(diffs) / n
    labels = [1.0] * n  # chosen always wins

    # Pearson r between diff and label (always 1, so we check variance)
    # Better metric: what % of pairs have rejected > chosen by >2x
    extreme_length_bias = sum(1 for r in ratios if r > 3.0)

    return {
        "chosen_longer_count": chosen_longer,
        "rejected_longer_count": rejected_longer,
        "avg_rejected_to_chosen_ratio": avg_ratio,
        "extreme_length_bias_count": extreme_length_bias,  # rejected >3x chosen
        "extreme_length_bias_pct": extreme_length_bias * 100 / n,
        "avg_chosen_len": mean_c,
        "avg_rejected_len": mean_r,
        # If >80% of pairs have rejected significantly longer, model learns length shortcut
        "length_bias_risk": "HIGH" if extreme_length_bias > n * 0.7 else
                           "MEDIUM" if extreme_length_bias > n * 0.4 else "LOW",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 3: DUPLICATE / NEAR-DUPLICATE PROMPT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_prompt(text: str) -> str:
    """Normalize prompt for dedup."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def duplicate_analysis(pairs: list) -> dict:
    """Find exact and near-duplicate prompts."""
    prompts = [p.get("prompt", "") for p in pairs]

    # Exact duplicates
    exact_counts = Counter(prompts)
    exact_dupes = {k: v for k, v in exact_counts.items() if v > 1}

    # Normalized duplicates
    norm_counts = Counter(normalize_prompt(p) for p in prompts)
    norm_dupes = {k: v for k, v in norm_counts.items() if v > 1}

    total_prompts = len(prompts)
    unique_prompts = len(set(prompts))
    unique_normalized = len(set(normalize_prompt(p) for p in prompts))

    # Duplicate rate
    dup_rate = (total_prompts - unique_normalized) / total_prompts if total_prompts > 0 else 0

    # Top duplicated prompts
    top_dupes = sorted(norm_dupes.items(), key=lambda x: -x[1])[:15]

    return {
        "total_prompts": total_prompts,
        "unique_prompts": unique_prompts,
        "unique_normalized": unique_normalized,
        "duplicate_rate": dup_rate,
        "duplicate_rate_pct": dup_rate * 100,
        "top_duplicated": top_dupes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 4: SEMANTIC CLUSTER ENTROPY (intent diversity)
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_entropy(pairs: list) -> dict:
    """Measure diversity of categories and intents."""
    categories = [p.get("category", "unknown") for p in pairs]
    cat_counts = Counter(categories)
    total = len(categories)

    # Shannon entropy
    entropy = 0
    for count in cat_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    # Max possible entropy (uniform distribution)
    max_entropy = math.log2(len(cat_counts)) if len(cat_counts) > 1 else 1

    # Normalized entropy (0 = all same category, 1 = perfectly uniform)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Intent diversity within categories
    prompt_per_cat = defaultdict(set)
    for p in pairs:
        cat = p.get("category", "unknown")
        prompt_per_cat[cat].add(normalize_prompt(p.get("prompt", "")))

    return {
        "category_counts": dict(sorted(cat_counts.items(), key=lambda x: -x[1])),
        "num_categories": len(cat_counts),
        "shannon_entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": norm_entropy,
        "entropy_rating": "HIGH" if norm_entropy > 0.85 else "MEDIUM" if norm_entropy > 0.7 else "LOW",
        "unique_intents_per_category": {k: len(v) for k, v in sorted(prompt_per_cat.items())},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 5: FLIP INCONSISTENCY RATE
# ═══════════════════════════════════════════════════════════════════════════════

def flip_inconsistency(pairs: list) -> dict:
    """Check if same prompt has contradictory chosen/rejected patterns.
    E.g., prompt A: chosen says "marriage in 2027", another pair for prompt A: chosen says "marriage in 2029"
    """
    prompt_groups = defaultdict(list)
    for p in pairs:
        key = normalize_prompt(p.get("prompt", ""))
        prompt_groups[key].append(p)

    flips = 0
    flip_examples = []
    total_multi = 0

    for prompt, group in prompt_groups.items():
        if len(group) < 2:
            continue
        total_multi += len(group) - 1

        # Check if chosen responses contradict each other
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                c1 = group[i]["chosen"]
                c2 = group[j]["chosen"]

                # Extract dates from both
                dates1 = set(re.findall(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}", c1))
                dates2 = set(re.findall(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}", c2))

                # If both mention dates but they're completely different, that's a flip
                if dates1 and dates2 and len(dates1 & dates2) == 0:
                    # Different charts may have different dates — only flag if same chart
                    chart1 = group[i].get("chart_name", "")
                    chart2 = group[j].get("chart_name", "")
                    if chart1 and chart2 and chart1 == chart2:
                        flips += 1
                        if len(flip_examples) < 3:
                            flip_examples.append({
                                "prompt": prompt,
                                "chart": chart1,
                                "dates1": dates1,
                                "dates2": dates2,
                            })

    return {
        "multi_prompt_pairs": total_multi,
        "flip_count": flips,
        "flip_rate": flips / max(total_multi, 1),
        "flip_rate_pct": flips * 100 / max(total_multi, 1),
        "flip_examples": flip_examples,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 6: REWARD MARGIN DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

def reward_margin_analysis(pairs: list) -> dict:
    """Compute proxy reward margin for each pair and analyze distribution."""
    margins = []
    chosen_scores_all = []
    rejected_scores_all = []
    low_margin_pairs = []
    negative_margin_pairs = []
    high_quality_pairs = []

    for p in pairs:
        cat = p.get("category", "unknown")
        prompt = p.get("prompt", "")

        chosen_score = score_response(p["chosen"], cat, prompt)
        rejected_score = score_response(p["rejected"], cat, prompt)

        margin = chosen_score["total"] - rejected_score["total"]
        margins.append(margin)
        chosen_scores_all.append(chosen_score["total"])
        rejected_scores_all.append(rejected_score["total"])

        p["_chosen_score"] = chosen_score["total"]
        p["_rejected_score"] = rejected_score["total"]
        p["_margin"] = margin

        if margin < 5:
            low_margin_pairs.append(p["_idx"])
        if margin <= 0:
            negative_margin_pairs.append(p["_idx"])
        if margin >= 15 and chosen_score["total"] >= 35:
            high_quality_pairs.append(p["_idx"])

    avg_margin = sum(margins) / len(margins)
    std_margin = (sum((m - avg_margin) ** 2 for m in margins) / len(margins)) ** 0.5

    # Distribution buckets
    buckets = {"<0": 0, "0-5": 0, "5-10": 0, "10-15": 0, "15-20": 0, "20-25": 0, "25+": 0}
    for m in margins:
        if m < 0: buckets["<0"] += 1
        elif m < 5: buckets["0-5"] += 1
        elif m < 10: buckets["5-10"] += 1
        elif m < 15: buckets["10-15"] += 1
        elif m < 20: buckets["15-20"] += 1
        elif m < 25: buckets["20-25"] += 1
        else: buckets["25+"] += 1

    return {
        "avg_margin": avg_margin,
        "std_margin": std_margin,
        "min_margin": min(margins),
        "max_margin": max(margins),
        "median_margin": sorted(margins)[len(margins) // 2],
        "avg_chosen_score": sum(chosen_scores_all) / len(chosen_scores_all),
        "avg_rejected_score": sum(rejected_scores_all) / len(rejected_scores_all),
        "margin_distribution": buckets,
        "low_margin_count": len(low_margin_pairs),
        "low_margin_pct": len(low_margin_pairs) * 100 / len(pairs),
        "negative_margin_count": len(negative_margin_pairs),
        "negative_margin_pct": len(negative_margin_pairs) * 100 / len(pairs),
        "high_quality_count": len(high_quality_pairs),
        "high_quality_pct": len(high_quality_pairs) * 100 / len(pairs),
        "low_margin_indices": low_margin_pairs,
        "negative_margin_indices": negative_margin_pairs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 7: DOMAIN-SPECIFIC QUALITY FLAGS
# ═══════════════════════════════════════════════════════════════════════════════

def domain_quality_flags(pairs: list) -> dict:
    """KP Astrology domain-specific quality checks."""
    flags = {
        "chosen_has_markdown": [],
        "chosen_has_robotic_headers": [],
        "chosen_has_iso_dates": [],
        "chosen_has_the_native": [],
        "chosen_missing_name_ji": [],
        "chosen_product_spam": [],
        "chosen_too_long": [],       # >4 sentences
        "chosen_too_short": [],      # <20 chars
        "rejected_too_similar": [],  # chosen ≈ rejected
        "rejected_not_bad_enough": [],  # rejected scores too high
        "language_mismatch": [],     # Hindi Q → English response
        "chosen_has_newlines": [],   # paragraph breaks
    }

    for p in pairs:
        idx = p["_idx"]
        c = p["chosen"]
        r = p["rejected"]
        cat = p.get("category", "")
        prompt = p.get("prompt", "")

        # Chosen quality flags
        if "**" in c or "##" in c:
            flags["chosen_has_markdown"].append(idx)
        if any(h in c for h in ["Analysis:", "Conclusion:", "Confidence:", "Critical Finding:"]):
            flags["chosen_has_robotic_headers"].append(idx)
        if re.search(r"\d{4}-\d{2}(?![\d-])", c) and not re.search(r"\d{4}-\d{4}", c):
            flags["chosen_has_iso_dates"].append(idx)
        if "the native" in c.lower():
            flags["chosen_has_the_native"].append(idx)
        if " ji" not in c and "ji," not in c and cat not in ("general",):
            flags["chosen_missing_name_ji"].append(idx)
        product_words = ["rudraksha", "bracelet", "pendant", "kavach", "mala", "necklace",
                         "try karein", "hamara", "wear our", "package"]
        if cat != "remedies" and any(w in c.lower() for w in product_words):
            flags["chosen_product_spam"].append(idx)
        sent_count = len([s for s in re.split(r"[.!?]+", c) if s.strip()])
        if sent_count > 4:
            flags["chosen_too_long"].append(idx)
        if len(c) < 20:
            flags["chosen_too_short"].append(idx)
        if "\n" in c:
            flags["chosen_has_newlines"].append(idx)

        # Rejected quality flags
        # Jaccard similarity between chosen and rejected
        c_words = set(c.lower().split())
        r_words = set(r.lower().split())
        if c_words and r_words:
            jaccard = len(c_words & r_words) / len(c_words | r_words)
            if jaccard > 0.7:
                flags["rejected_too_similar"].append(idx)

        # Rejected should score low — if it scores high, the pair is noisy
        rej_score = score_response(r, cat, prompt)
        if rej_score["total"] >= 35:  # Out of 50
            flags["rejected_not_bad_enough"].append(idx)

        # Language mismatch
        prompt_hindi = bool(re.search(r'kab|kya|meri|mera|hogi|hoga|kaise|kaisa|marunga|tabiyat|shaadi|naukri|rashi|lagna', prompt.lower()))
        if prompt_hindi:
            hindi_markers = ["hai", "hain", "aapke", "aapki", "karta", "hoga", "mein ", "karein", "aapka"]
            hindi_count = sum(1 for w in hindi_markers if w in c.lower())
            if hindi_count < 2:
                flags["language_mismatch"].append(idx)

    return flags


# ═══════════════════════════════════════════════════════════════════════════════
# PAIR-LEVEL FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

def filter_pairs(pairs: list, flags: dict, margin_data: dict) -> tuple:
    """Remove noisy pairs and return (clean_pairs, removed_pairs, reasons)."""
    remove_indices = set()
    reasons = defaultdict(list)

    # Critical: remove pairs with wrong labels (negative margin)
    for idx in margin_data["negative_margin_indices"]:
        remove_indices.add(idx)
        reasons[idx].append("negative_margin")

    # Remove chosen with markdown/robotic headers
    for idx in flags["chosen_has_markdown"]:
        remove_indices.add(idx)
        reasons[idx].append("chosen_markdown")
    for idx in flags["chosen_has_robotic_headers"]:
        remove_indices.add(idx)
        reasons[idx].append("chosen_robotic_headers")

    # Remove chosen with ISO dates
    for idx in flags["chosen_has_iso_dates"]:
        remove_indices.add(idx)
        reasons[idx].append("chosen_iso_dates")

    # Remove chosen with "the native"
    for idx in flags["chosen_has_the_native"]:
        remove_indices.add(idx)
        reasons[idx].append("chosen_the_native")

    # Remove product spam in non-remedy
    for idx in flags["chosen_product_spam"]:
        remove_indices.add(idx)
        reasons[idx].append("product_spam")

    # Remove too-similar pairs (low contrast signal)
    for idx in flags["rejected_too_similar"]:
        remove_indices.add(idx)
        reasons[idx].append("too_similar")

    # Remove rejected that's not bad enough (noisy label)
    for idx in flags["rejected_not_bad_enough"]:
        remove_indices.add(idx)
        reasons[idx].append("rejected_not_bad")

    # Remove very low margin pairs (margin < 5) — noisy gradient signal
    for idx in margin_data["low_margin_indices"]:
        if idx not in remove_indices:  # Don't double-count
            remove_indices.add(idx)
            reasons[idx].append("low_margin")

    # Remove chosen with paragraph breaks (format violation)
    for idx in flags["chosen_has_newlines"]:
        remove_indices.add(idx)
        reasons[idx].append("chosen_newlines")

    # Remove chosen too short
    for idx in flags["chosen_too_short"]:
        remove_indices.add(idx)
        reasons[idx].append("chosen_too_short")

    # Remove chosen >4 sentences
    for idx in flags["chosen_too_long"]:
        remove_indices.add(idx)
        reasons[idx].append("chosen_too_long")

    clean = [p for p in pairs if p["_idx"] not in remove_indices]
    removed = [p for p in pairs if p["_idx"] in remove_indices]

    # Reason summary
    reason_counts = Counter()
    for idx, r_list in reasons.items():
        for r in r_list:
            reason_counts[r] += 1

    return clean, removed, reason_counts


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(pairs, length_data, dup_data, entropy_data, flip_data, margin_data, flags, clean, removed, reason_counts):
    n = len(pairs)
    print("=" * 78)
    print("  DPO DATASET QUALITY AUDIT — INDUSTRY STANDARD")
    print("=" * 78)
    print(f"  Dataset: {n} pairs")
    print()

    # ── Metric 1: Reward Margin ──
    print("─" * 78)
    print("  1. REWARD MARGIN (proxy score, 0-50 scale)")
    print("─" * 78)
    print(f"  Avg chosen score:   {margin_data['avg_chosen_score']:.1f}/50")
    print(f"  Avg rejected score: {margin_data['avg_rejected_score']:.1f}/50")
    print(f"  Avg margin:         {margin_data['avg_margin']:.1f}")
    print(f"  Std margin:         {margin_data['std_margin']:.1f}")
    print(f"  Median margin:      {margin_data['median_margin']}")
    print(f"  Min/Max margin:     {margin_data['min_margin']} / {margin_data['max_margin']}")
    print(f"  Target: mean 10-25, std <8")
    status = "✅" if 10 <= margin_data['avg_margin'] <= 25 else "⚠️"
    print(f"  Status: {status}")
    print(f"\n  Distribution:")
    for bucket, count in margin_data["margin_distribution"].items():
        bar = "█" * (count // 10)
        pct = count * 100 // n
        print(f"    {bucket:>6}: {count:>4} ({pct:>2}%) {bar}")
    print(f"\n  Negative margin (WRONG LABELS): {margin_data['negative_margin_count']} ({margin_data['negative_margin_pct']:.1f}%)")
    print(f"  Low margin (<5, NOISY):         {margin_data['low_margin_count']} ({margin_data['low_margin_pct']:.1f}%)")
    print(f"  High quality (margin≥15, chosen≥35): {margin_data['high_quality_count']} ({margin_data['high_quality_pct']:.1f}%)")

    # ── Metric 2: Length Bias ──
    print()
    print("─" * 78)
    print("  2. LENGTH BIAS DETECTION")
    print("─" * 78)
    print(f"  Avg chosen length:  {length_data['avg_chosen_len']:.0f} chars")
    print(f"  Avg rejected length: {length_data['avg_rejected_len']:.0f} chars")
    print(f"  Avg ratio (rej/cho): {length_data['avg_rejected_to_chosen_ratio']:.1f}x")
    print(f"  Chosen longer:  {length_data['chosen_longer_count']} pairs")
    print(f"  Rejected longer: {length_data['rejected_longer_count']} pairs")
    print(f"  Extreme bias (rej >3x cho): {length_data['extreme_length_bias_count']} ({length_data['extreme_length_bias_pct']:.1f}%)")
    print(f"  Length bias risk: {length_data['length_bias_risk']}")
    print(f"  Target: ratio 1.5-3x, extreme <40%")
    risk = length_data['length_bias_risk']
    print(f"  Status: {'✅' if risk == 'LOW' else '⚠️' if risk == 'MEDIUM' else '❌'}")

    # ── Metric 3: Duplicate Prompts ──
    print()
    print("─" * 78)
    print("  3. DUPLICATE PROMPT RATE")
    print("─" * 78)
    print(f"  Total prompts:      {dup_data['total_prompts']}")
    print(f"  Unique (exact):     {dup_data['unique_prompts']}")
    print(f"  Unique (normalized): {dup_data['unique_normalized']}")
    print(f"  Duplicate rate:     {dup_data['duplicate_rate_pct']:.1f}%")
    print(f"  Target: <5%")
    status = "✅" if dup_data['duplicate_rate_pct'] < 5 else "⚠️" if dup_data['duplicate_rate_pct'] < 15 else "❌"
    print(f"  Status: {status}")
    if dup_data['top_duplicated']:
        print(f"\n  Top duplicated prompts:")
        for prompt, count in dup_data['top_duplicated'][:10]:
            print(f"    {count:>3}x: {prompt[:70]}")

    # ── Metric 4: Semantic Cluster Entropy ──
    print()
    print("─" * 78)
    print("  4. SEMANTIC CLUSTER ENTROPY (intent diversity)")
    print("─" * 78)
    print(f"  Categories: {entropy_data['num_categories']}")
    print(f"  Shannon entropy: {entropy_data['shannon_entropy']:.2f}")
    print(f"  Max entropy:     {entropy_data['max_entropy']:.2f}")
    print(f"  Normalized:      {entropy_data['normalized_entropy']:.2f}")
    print(f"  Rating: {entropy_data['entropy_rating']}")
    print(f"  Target: normalized >0.85")
    status = "✅" if entropy_data['normalized_entropy'] > 0.85 else "⚠️"
    print(f"  Status: {status}")
    print(f"\n  Category distribution:")
    for cat, count in entropy_data['category_counts'].items():
        bar = "█" * (count // 10)
        pct = count * 100 // n
        print(f"    {cat:<18}: {count:>4} ({pct:>2}%) {bar}")
    print(f"\n  Unique intents per category:")
    for cat, count in entropy_data['unique_intents_per_category'].items():
        print(f"    {cat:<18}: {count}")

    # ── Metric 5: Flip Inconsistency ──
    print()
    print("─" * 78)
    print("  5. FLIP INCONSISTENCY RATE")
    print("─" * 78)
    print(f"  Multi-prompt pairs: {flip_data['multi_prompt_pairs']}")
    print(f"  Flips detected:    {flip_data['flip_count']}")
    print(f"  Flip rate:         {flip_data['flip_rate_pct']:.1f}%")
    print(f"  Target: <10%")
    status = "✅" if flip_data['flip_rate_pct'] < 10 else "⚠️"
    print(f"  Status: {status}")

    # ── Metric 6: Domain Quality Flags ──
    print()
    print("─" * 78)
    print("  6. DOMAIN-SPECIFIC QUALITY FLAGS")
    print("─" * 78)
    for flag_name, indices in flags.items():
        count = len(indices)
        pct = count * 100 / n
        status = "✅" if count == 0 else "⚠️" if pct < 5 else "❌"
        print(f"  {status} {flag_name:<30}: {count:>4} ({pct:.1f}%)")

    # ── Metric 7: Filtering Results ──
    print()
    print("─" * 78)
    print("  7. FILTERING RESULTS")
    print("─" * 78)
    print(f"  Original pairs:  {n}")
    print(f"  Clean pairs:     {len(clean)} ({len(clean)*100//n}%)")
    print(f"  Removed pairs:   {len(removed)} ({len(removed)*100//n}%)")
    print(f"\n  Removal reasons:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:<25}: {count}")

    # ── OVERALL SCORECARD ──
    print()
    print("=" * 78)
    print("  OVERALL SCORECARD")
    print("=" * 78)
    metrics = {
        "Reward margin mean (10-25)": "✅" if 10 <= margin_data['avg_margin'] <= 25 else "❌",
        "Negative margin (<1%)": "✅" if margin_data['negative_margin_pct'] < 1 else "❌",
        "Length bias (LOW/MED)": "✅" if length_data['length_bias_risk'] != "HIGH" else "❌",
        "Duplicate rate (<5%)": "✅" if dup_data['duplicate_rate_pct'] < 5 else "⚠️" if dup_data['duplicate_rate_pct'] < 15 else "❌",
        "Entropy (>0.85)": "✅" if entropy_data['normalized_entropy'] > 0.85 else "⚠️",
        "Flip rate (<10%)": "✅" if flip_data['flip_rate_pct'] < 10 else "❌",
        "Chosen markdown (0)": "✅" if len(flags['chosen_has_markdown']) == 0 else "❌",
        "Chosen robotic (0)": "✅" if len(flags['chosen_has_robotic_headers']) == 0 else "❌",
        "Chosen ISO dates (0)": "✅" if len(flags['chosen_has_iso_dates']) == 0 else "❌",
        "Product spam (0)": "✅" if len(flags['chosen_product_spam']) == 0 else "❌",
        "Language mismatch (<5%)": "✅" if len(flags['language_mismatch']) * 100 / n < 5 else "⚠️",
        "Clean retention (>70%)": "✅" if len(clean) / n > 0.7 else "⚠️" if len(clean) / n > 0.5 else "❌",
    }
    passed = sum(1 for v in metrics.values() if v == "✅")
    warned = sum(1 for v in metrics.values() if v == "⚠️")
    failed = sum(1 for v in metrics.values() if v == "❌")

    for metric, status in metrics.items():
        print(f"  {status} {metric}")

    print(f"\n  Score: {passed} PASS / {warned} WARN / {failed} FAIL out of {len(metrics)}")

    return clean


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DPO Dataset Quality Audit")
    parser.add_argument("--input", default="data/dpo/dpo_pairs.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="data/dpo/dpo_pairs_filtered.jsonl", help="Filtered output")
    parser.add_argument("--filter", action="store_true", help="Save filtered dataset")
    parser.add_argument("--removed", default="data/dpo/dpo_pairs_removed.jsonl", help="Removed pairs")
    args = parser.parse_args()

    print(f"\nLoading {args.input}...")
    pairs = load_pairs(args.input)
    print(f"  Loaded {len(pairs)} pairs")

    if not pairs:
        print("❌ No pairs found!")
        return

    # Run all metrics
    print("\nRunning quality audit...\n")

    length_data = length_bias_analysis(pairs)
    dup_data = duplicate_analysis(pairs)
    entropy_data = cluster_entropy(pairs)
    flip_data = flip_inconsistency(pairs)
    margin_data = reward_margin_analysis(pairs)
    flags = domain_quality_flags(pairs)

    # Filter
    clean, removed, reason_counts = filter_pairs(pairs, flags, margin_data)

    # Report
    clean = print_report(pairs, length_data, dup_data, entropy_data, flip_data,
                         margin_data, flags, clean, removed, reason_counts)

    # Save filtered dataset
    if args.filter:
        # Re-filter to get clean list (print_report returns it)
        clean_pairs, _, _ = filter_pairs(pairs, flags, margin_data)

        with open(args.output, "w", encoding="utf-8") as f:
            for p in clean_pairs:
                # Remove internal fields
                out = {k: v for k, v in p.items() if not k.startswith("_")}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"\n✓ Filtered dataset saved: {args.output} ({len(clean_pairs)} pairs)")

        with open(args.removed, "w", encoding="utf-8") as f:
            for p in removed:
                out = {k: v for k, v in p.items() if not k.startswith("_")}
                out["_removal_reasons"] = [r for r in reason_counts if p["_idx"] in flags.get(r, [])]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"✓ Removed pairs saved: {args.removed} ({len(removed)} pairs)")


if __name__ == "__main__":
    main()
