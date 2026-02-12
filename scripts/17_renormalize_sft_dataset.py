"""
SFT Dataset Re-normalization Script
=====================================
Cleans the existing SFT dataset outputs to match the desired production style:
  - Strips markdown (bold, italic, headers, code blocks, bullets)
  - Removes robotic headers ("Analysis:", "Conclusion:", etc.)
  - Replaces "the native" / "the querent" with direct address ("you"/"your")
  - Converts ISO dates to readable format (2025-10 → Oct 2025)
  - Enforces sentence limits per category
  - Removes filler phrases and methodology explanations

This script modifies data/sft_train/ and data/sft_validation/ IN PLACE
(with backup) so the model learns the correct style from the start,
reducing reliance on postprocessing at inference time.

Usage:
  python scripts/17_renormalize_sft_dataset.py
  python scripts/17_renormalize_sft_dataset.py --dry-run          # preview only
  python scripts/17_renormalize_sft_dataset.py --no-backup        # skip backup
  python scripts/17_renormalize_sft_dataset.py --max-sentences 4  # override cap
"""

import argparse
import re
import shutil
from pathlib import Path
from datasets import load_from_disk

parser = argparse.ArgumentParser(description="Re-normalize SFT dataset outputs")
parser.add_argument("--train-path", type=str, default="data/sft_train/")
parser.add_argument("--val-path", type=str, default="data/sft_validation/")
parser.add_argument("--dry-run", action="store_true", help="Preview changes without saving")
parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
parser.add_argument("--max-sentences", type=int, default=5,
                    help="Max sentences for analysis outputs (default: 5)")
args = parser.parse_args()

print("=" * 80)
print("SFT DATASET RE-NORMALIZATION")
print("=" * 80)


# ── Month map for ISO date conversion ────────────────────────────────────────
_MONTH_MAP = {
    '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr',
    '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug',
    '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec',
}


def _iso_to_readable(m):
    y, mo = m.group(1), m.group(2)
    return f"{_MONTH_MAP.get(mo, mo)} {y}"


# ── Robotic headers to strip ─────────────────────────────────────────────────
_ROBOTIC_HEADER_PATTERNS = [
    r'^#{1,6}\s+.*$',
    r'^(?:Analysis|Conclusion|Summary|Overview|Introduction|Observation|Application|Interpretation)\s*:?\s*$',
    r'^(?:Key\s+Findings?|Critical\s+Finding)\s*:?\s*$',
    r'^(?:Motivational\s+Quote|Hindi\s+Quote|Recommended\s+Product|Product\s+Recommendation)\s*:?\s*$',
    r'^(?:Remedial\s+Measures|Remedy|Timing)\s*:?\s*$',
    r'^(?:Career|Financial|Health|Marriage|Education)\s+Analysis\s*:?\s*$',
    r'^(?:Astrological\s+)?(?:Prediction|Assessment|Evaluation|Reading)\s*:?\s*$',
    r'^(?:Important|Note|Disclaimer|Warning|Caution)\s*:?\s*$',
    r'^(?:Step|Phase|Part|Section)\s+\d+\s*:?\s*$',
]
_ROBOTIC_HEADER_RE = [re.compile(p, re.IGNORECASE) for p in _ROBOTIC_HEADER_PATTERNS]

# ── Filler phrases to remove (line-level) ────────────────────────────────────
_FILLER_PHRASES = [
    "according to kp astrology", "using kp methodology", "as per kp principles",
    "based on the chart data provided", "from the given chart",
    "as mentioned in the chart", "the chart shows that",
    "looking at the chart data", "examining the chart",
    "let me analyze", "let me examine", "i will now analyze",
    "let us analyze", "let us examine",
    "based on the given data", "based on the extracted data",
    "for accurate prediction", "for proper analysis",
    "grounding rule", "as per the grounding",
    "considerably enhanced", "enhanced answer", "proper format",
]

# ── Third-person replacements ────────────────────────────────────────────────
_THIRD_PERSON_REPLACEMENTS = [
    (re.compile(r'\bThe\s+native\s+has\b', re.IGNORECASE), 'You have'),
    (re.compile(r'\bThe\s+native\s+is\b', re.IGNORECASE), 'You are'),
    (re.compile(r'\bThe\s+native\s+will\b', re.IGNORECASE), 'You will'),
    (re.compile(r'\bThe\s+native\s+should\b', re.IGNORECASE), 'You should'),
    (re.compile(r'\bThe\s+native\s+may\b', re.IGNORECASE), 'You may'),
    (re.compile(r'\bThe\s+native\s+can\b', re.IGNORECASE), 'You can'),
    (re.compile(r"\bThe\s+native's\b", re.IGNORECASE), 'Your'),
    (re.compile(r'\bThe\s+native\b', re.IGNORECASE), 'You'),
    (re.compile(r"\bthe\s+native's\b", re.IGNORECASE), 'your'),
    (re.compile(r'\bthe\s+native\b', re.IGNORECASE), 'you'),
    (re.compile(r'\bThe\s+querent\b', re.IGNORECASE), 'You'),
    (re.compile(r'\bthe\s+querent\b', re.IGNORECASE), 'you'),
    (re.compile(r'\bThe\s+person\b', re.IGNORECASE), 'You'),
    (re.compile(r'\bthe\s+person\b', re.IGNORECASE), 'you'),
    (re.compile(r'\bIt\s+is\s+(?:observed|noted|seen)\s+that\b', re.IGNORECASE), ''),
    (re.compile(r'\bIt\s+(?:can\s+be|is)\s+(?:concluded|inferred)\s+that\b', re.IGNORECASE), ''),
    (re.compile(r'\bIn\s+conclusion,?\s*', re.IGNORECASE), ''),
    (re.compile(r'\bTo\s+summarize,?\s*', re.IGNORECASE), ''),
]


def normalize_output(text: str, max_sentences: int = 5) -> str:
    """Clean a single SFT output to match desired production style."""
    if not text or not text.strip():
        return text

    # ── Strip ALL markdown formatting ──
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)       # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)            # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)            # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)              # _italic_
    text = re.sub(r'#{1,6}\s+', '', text)                 # ### headers
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)  # code blocks
    text = re.sub(r'`([^`]+)`', r'\1', text)              # inline code

    # ── Remove numbered lists and bullet points ──
    text = re.sub(r'(?:^|\n)\s*\d+[.)]\s+', '\n', text)
    text = re.sub(r'(?:^|\n)\s*[-•●◦▪]\s+', '\n', text)

    # ── Convert ISO dates ──
    text = re.sub(r'\b(20\d{2})-(0[1-9]|1[0-2])(?:-\d{2})?\b', _iso_to_readable, text)

    # ── Replace third-person references ──
    for pat, repl in _THIRD_PERSON_REPLACEMENTS:
        text = pat.sub(repl, text)

    # ── Remove robotic headers and filler lines ──
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip robotic headers
        if any(r.match(stripped) for r in _ROBOTIC_HEADER_RE):
            continue
        # Skip lines that are only a short label ending with ':'
        if stripped.endswith(':') and len(stripped) < 50 and not any(c.isdigit() for c in stripped):
            continue
        # Skip filler phrases
        if any(filler in stripped.lower() for filler in _FILLER_PHRASES):
            continue
        # Skip hallucinated references
        if re.match(r'^rules_used:', stripped, re.IGNORECASE):
            continue
        if re.match(r'^level:', stripped, re.IGNORECASE):
            continue
        if re.match(r'^confidence:', stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # ── Remove hallucinated rule IDs and source references ──
    text = re.sub(r'\bKP_[A-Z]{2,4}_\d{3,5}\b', '', text)
    text = re.sub(r'\[KP_[A-Z_0-9]+\]', '', text)
    text = re.sub(r'\((?:Source|Ref|Reference|Page|Ch(?:apter)?)[^)]{0,60}\)', '', text, flags=re.IGNORECASE)

    # ── Clean up whitespace ──
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    text = text.strip()

    # ── Sentence cap ──
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > max_sentences:
        text = ' '.join(sentences[:max_sentences])

    # ── Remove trailing incomplete sentence ──
    if text and text[-1] not in '.!?")}':
        last_period = max(text.rfind('. '), text.rfind('.\n'), text.rfind('.'))
        if last_period > len(text) * 0.3:
            text = text[:last_period + 1]

    return text.strip()


def process_dataset(ds_path: str, max_sentences: int, dry_run: bool, no_backup: bool):
    """Load, normalize, and save a dataset."""
    path = Path(ds_path)
    if not path.exists():
        print(f"  ⚠️ Dataset not found: {path}")
        return

    ds = load_from_disk(str(path))
    print(f"\n  Loading: {path} ({len(ds)} examples)")

    # Backup
    if not dry_run and not no_backup:
        backup_path = Path(str(path) + "_backup")
        if not backup_path.exists():
            shutil.copytree(str(path), str(backup_path))
            print(f"  ✓ Backup created: {backup_path}")
        else:
            print(f"  ℹ️ Backup already exists: {backup_path}")

    # Count issues before
    stats_before = {
        "markdown_bold": 0, "headers": 0, "the_native": 0, "bullets": 0,
    }
    for ex in ds:
        out = ex.get("output", "")
        if "**" in out:
            stats_before["markdown_bold"] += 1
        if re.search(r'^#{1,6}\s', out, re.MULTILINE):
            stats_before["headers"] += 1
        if "the native" in out.lower():
            stats_before["the_native"] += 1
        if re.search(r'(?:^|\n)\s*[-•●]\s', out):
            stats_before["bullets"] += 1

    print(f"  Before: bold={stats_before['markdown_bold']}, headers={stats_before['headers']}, "
          f"the_native={stats_before['the_native']}, bullets={stats_before['bullets']}")

    # Normalize
    def _normalize(example):
        example["output"] = normalize_output(example["output"], max_sentences=max_sentences)
        return example

    ds_clean = ds.map(_normalize, desc="Normalizing outputs")

    # Count issues after
    stats_after = {
        "markdown_bold": 0, "headers": 0, "the_native": 0, "bullets": 0,
    }
    for ex in ds_clean:
        out = ex.get("output", "")
        if "**" in out:
            stats_after["markdown_bold"] += 1
        if re.search(r'^#{1,6}\s', out, re.MULTILINE):
            stats_after["headers"] += 1
        if "the native" in out.lower():
            stats_after["the_native"] += 1
        if re.search(r'(?:^|\n)\s*[-•●]\s', out):
            stats_after["bullets"] += 1

    print(f"  After:  bold={stats_after['markdown_bold']}, headers={stats_after['headers']}, "
          f"the_native={stats_after['the_native']}, bullets={stats_after['bullets']}")

    # Show sample
    if len(ds) > 0:
        idx = min(100, len(ds) - 1)
        print(f"\n  --- Sample (index {idx}) ---")
        print(f"  BEFORE: {ds[idx]['output'][:200]}...")
        print(f"  AFTER:  {ds_clean[idx]['output'][:200]}...")

    if dry_run:
        print(f"\n  [DRY RUN] No changes saved.")
    else:
        ds_clean.save_to_disk(str(path))
        print(f"  ✓ Saved normalized dataset to: {path}")


# ── Process both train and validation ────────────────────────────────────────
process_dataset(args.train_path, args.max_sentences, args.dry_run, args.no_backup)
process_dataset(args.val_path, args.max_sentences, args.dry_run, args.no_backup)

print(f"\n{'=' * 80}")
print("SFT RE-NORMALIZATION COMPLETE")
print(f"{'=' * 80}")
if args.dry_run:
    print("This was a DRY RUN. Re-run without --dry-run to apply changes.")
else:
    print("Datasets have been normalized in-place (backups created).")
    print("Next step: Retrain SFT with the cleaned dataset:")
    print("  python scripts/04_train_sft.py")
print(f"{'=' * 80}")
