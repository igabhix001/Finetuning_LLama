"""
DAPT Corpus Cleaner — Strip OCR/Archive Boilerplate & Deduplicate
==================================================================
Cleans the existing DAPT Arrow dataset by:
  - Removing Internet Archive notices and OCR disclaimers
  - Stripping repeated headers/footers (page numbers, book titles, chapter labels)
  - Removing publisher/copyright boilerplate
  - Deduplicating near-identical chunks (fuzzy hash)
  - Removing very short or empty chunks

Operates on data/dapt_corpus/ (HF Arrow dataset) IN PLACE with backup.

Usage:
  python scripts/18_clean_dapt_corpus.py                    # clean in-place
  python scripts/18_clean_dapt_corpus.py --dry-run          # preview only
  python scripts/18_clean_dapt_corpus.py --no-backup        # skip backup
  python scripts/18_clean_dapt_corpus.py --path data/dapt_corpus/
"""

import argparse
import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from datasets import load_from_disk

parser = argparse.ArgumentParser(description="Clean DAPT corpus: strip boilerplate, deduplicate")
parser.add_argument("--path", type=str, default="data/dapt_corpus/", help="Path to DAPT Arrow dataset")
parser.add_argument("--dry-run", action="store_true", help="Preview changes without saving")
parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
parser.add_argument("--min-chars", type=int, default=200, help="Minimum chars per chunk after cleaning")
args = parser.parse_args()

print("=" * 80)
print("DAPT CORPUS CLEANER")
print("=" * 80)

# ── Boilerplate patterns to strip ────────────────────────────────────────────
_BOILERPLATE_PATTERNS = [
    # Internet Archive notices
    re.compile(r'(?:This\s+)?(?:book|text|document)\s+(?:is|was)\s+(?:provided|made\s+available)\s+by\s+(?:the\s+)?Internet\s+Archive[^.]*\.?', re.IGNORECASE),
    re.compile(r'Internet\s+Archive[^.]*(?:digitized|scanned|uploaded|OCR)[^.]*\.?', re.IGNORECASE),
    re.compile(r'(?:Digitized|Scanned|Uploaded)\s+by\s+(?:the\s+)?Internet\s+Archive[^.]*\.?', re.IGNORECASE),
    re.compile(r'(?:Available|Downloaded)\s+(?:at|from)\s+(?:https?://)?(?:www\.)?(?:archive\.org|openlibrary\.org)[^\s]*', re.IGNORECASE),
    # OCR disclaimers
    re.compile(r'(?:This\s+)?(?:text|book|document)\s+(?:is|was|may\s+be)\s+(?:susceptible|subject)\s+to\s+(?:OCR\s+)?errors?[^.]*\.?', re.IGNORECASE),
    re.compile(r'OCR\s+(?:errors?|artifacts?|noise|quality)[^.]*\.?', re.IGNORECASE),
    re.compile(r'(?:Optical\s+Character\s+Recognition|OCR)\s+(?:may|might|could)\s+(?:have\s+)?(?:introduced|caused)[^.]*\.?', re.IGNORECASE),
    # Publisher/copyright boilerplate
    re.compile(r'All\s+rights?\s+reserved\.?\s*(?:No\s+part\s+of\s+this[^.]*\.?)?', re.IGNORECASE),
    re.compile(r'(?:Published|Printed)\s+(?:by|at|in)\s+[A-Z][^.]{5,80}\.', re.IGNORECASE),
    re.compile(r'(?:First|Second|Third|Fourth|Fifth|Sixth|Revised)\s+(?:Edition|Printing|Impression)[^.]*\.?', re.IGNORECASE),
    re.compile(r'(?:Price|Rs\.?|ISBN)\s*[:.]?\s*[\d./-]+', re.IGNORECASE),
    re.compile(r'Copyright\s*©?\s*\d{4}[^.]*\.?', re.IGNORECASE),
    # Page markers
    re.compile(r'---\s*Page\s*\d+\s*---'),
    re.compile(r'^\s*-?\s*\d+\s*-?\s*$', re.MULTILINE),
    # Repeated chapter/section headers (standalone lines)
    re.compile(r'^\s*(?:CHAPTER|Chapter|SECTION|Section)\s+[IVXLCDM\d]+\s*$', re.MULTILINE),
    # Table of contents lines
    re.compile(r'^\s*(?:Table\s+of\s+Contents|Contents|INDEX)\s*$', re.IGNORECASE | re.MULTILINE),
    # "KP Reader" repeated header/footer
    re.compile(r'^\s*K\.?\s*P\.?\s*Reader\s*(?:I{1,3}|IV|V|VI)?\s*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*Krishnamurti\s+Padhdhati\s*(?:Reader)?\s*(?:I{1,3}|IV|V|VI)?\s*$', re.IGNORECASE | re.MULTILINE),
]

# ── Lines to strip entirely ──────────────────────────────────────────────────
_LINE_STRIP_PATTERNS = [
    re.compile(r'^\s*\d+\s*$'),                          # bare page numbers
    re.compile(r'^\s*[-_=]{3,}\s*$'),                     # horizontal rules
    re.compile(r'^\s*\.\s*\.\s*\.\s*$'),                  # ellipsis lines
    re.compile(r'^\s*(?:www\.|http|ftp)[^\s]+\s*$'),      # bare URLs
]


def clean_dapt_text(text: str) -> str:
    """Clean a single DAPT chunk: strip boilerplate, fix whitespace."""
    if not text or not text.strip():
        return ""

    # Apply boilerplate pattern removal
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub('', text)

    # Strip individual lines
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines matching strip patterns
        if any(p.match(stripped) for p in _LINE_STRIP_PATTERNS):
            continue
        # Skip very short lines that are likely noise (< 10 chars, not a sentence)
        if len(stripped) < 10 and not stripped.endswith('.'):
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Fix whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    return text.strip()


def fuzzy_hash(text: str, ngram_size: int = 5) -> str:
    """Create a fuzzy hash for near-duplicate detection using character n-grams."""
    # Normalize: lowercase, strip whitespace, remove punctuation
    normalized = re.sub(r'[^a-z0-9]', '', text.lower())
    if len(normalized) < ngram_size:
        return hashlib.md5(normalized.encode()).hexdigest()
    # Take n-grams at fixed positions for a stable fingerprint
    positions = [0, len(normalized)//4, len(normalized)//2, 3*len(normalized)//4, -ngram_size]
    sample = ''.join(normalized[max(0, p):max(0, p)+ngram_size] for p in positions)
    return hashlib.md5(sample.encode()).hexdigest()


# ── Main processing ──────────────────────────────────────────────────────────
ds_path = Path(args.path)
if not ds_path.exists():
    print(f"❌ Dataset not found: {ds_path}")
    print("Check the path or run the DAPT corpus creation script first.")
    exit(1)

ds = load_from_disk(str(ds_path))
print(f"Loaded: {ds_path} ({len(ds)} chunks)")

# Backup
if not args.dry_run and not args.no_backup:
    backup_path = Path(str(ds_path) + "_backup")
    if not backup_path.exists():
        shutil.copytree(str(ds_path), str(backup_path))
        print(f"✓ Backup created: {backup_path}")
    else:
        print(f"ℹ️ Backup already exists: {backup_path}")

# Detect text column
text_col = "text" if "text" in ds.column_names else ds.column_names[0]
print(f"Text column: '{text_col}'")

# Stats before
total_chars_before = sum(len(ex[text_col]) for ex in ds)
boilerplate_hits = 0
for ex in ds:
    t = ex[text_col]
    if any(p.search(t) for p in _BOILERPLATE_PATTERNS):
        boilerplate_hits += 1
print(f"Before: {len(ds)} chunks, {total_chars_before:,} chars, {boilerplate_hits} with boilerplate")

# Clean
def _clean(example):
    example[text_col] = clean_dapt_text(example[text_col])
    return example

ds_clean = ds.map(_clean, desc="Cleaning DAPT chunks")

# Filter short/empty
ds_clean = ds_clean.filter(lambda ex: len(ex[text_col].strip()) >= args.min_chars,
                           desc="Filtering short chunks")

# Deduplicate by fuzzy hash
seen_hashes = set()
dedup_indices = []
for i, ex in enumerate(ds_clean):
    h = fuzzy_hash(ex[text_col])
    if h not in seen_hashes:
        seen_hashes.add(h)
        dedup_indices.append(i)
removed_dupes = len(ds_clean) - len(dedup_indices)
ds_clean = ds_clean.select(dedup_indices)

# Stats after
total_chars_after = sum(len(ex[text_col]) for ex in ds_clean)
boilerplate_after = 0
for ex in ds_clean:
    t = ex[text_col]
    if any(p.search(t) for p in _BOILERPLATE_PATTERNS):
        boilerplate_after += 1

print(f"After:  {len(ds_clean)} chunks, {total_chars_after:,} chars, {boilerplate_after} with boilerplate")
print(f"Removed: {len(ds) - len(ds_clean)} chunks ({len(ds) - len(ds_clean) - removed_dupes} too short, {removed_dupes} duplicates)")
print(f"Chars reduced: {total_chars_before - total_chars_after:,} ({(1 - total_chars_after/max(total_chars_before,1))*100:.1f}%)")

# Show sample
if len(ds) > 0 and len(ds_clean) > 0:
    idx = min(5, len(ds) - 1)
    print(f"\n--- Sample (index {idx}) ---")
    print(f"BEFORE: {ds[idx][text_col][:200]}...")
    print(f"AFTER:  {ds_clean[min(idx, len(ds_clean)-1)][text_col][:200]}...")

if args.dry_run:
    print(f"\n[DRY RUN] No changes saved.")
else:
    # Save to temp then swap (same pattern as SFT script)
    tmp_dir = Path(tempfile.mkdtemp(prefix="dapt_clean_", dir=ds_path.parent))
    try:
        ds_clean.save_to_disk(str(tmp_dir))
        shutil.rmtree(str(ds_path))
        tmp_dir.rename(ds_path)
        print(f"\n✓ Saved cleaned dataset to: {ds_path}")
    except Exception as e:
        print(f"❌ Save failed: {e}")
        if tmp_dir.exists():
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise

print(f"\n{'=' * 80}")
print("DAPT CORPUS CLEANING COMPLETE")
print(f"{'=' * 80}")
if not args.dry_run:
    print("Next step: Retrain DAPT with the cleaned corpus:")
    print("  python scripts/03_train_dapt.py")
print(f"{'=' * 80}")
