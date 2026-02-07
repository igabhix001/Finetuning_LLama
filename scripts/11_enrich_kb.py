"""
KB Enrichment Script — adds page numbers to Pinecone metadata and generates
missing chunks (dasa balance tables, horary number mappings, planet-house 9×12).

Usage:
  # Dry run (show what would be updated):
  python scripts/11_enrich_kb.py --dry-run

  # Actually update Pinecone:
  python scripts/11_enrich_kb.py

  # Only add page numbers (skip new chunks):
  python scripts/11_enrich_kb.py --pages-only
"""

import json
import os
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="Enrich KP Astrology Knowledge Base")
parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
parser.add_argument("--pages-only", action="store_true", help="Only add page numbers, skip new chunks")
parser.add_argument("--rules-file", default="data/final/kb_rules_extracted.jsonl",
                    help="Path to original rules with page numbers")
parser.add_argument("--kb-chunks", default="data/kb_chunks.jsonl",
                    help="Path to current KB chunks")
args = parser.parse_args()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "kp-astrology-kb")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

# ── Step 1: Build rule_id → (source_book, source_doc/page) mapping ────────────
print("=" * 70)
print("KB ENRICHMENT")
print("=" * 70)

# Search multiple possible locations for the rules file
_candidates = [
    Path(args.rules_file),                                          # explicit arg
    Path("data/final/kb_rules_extracted.jsonl"),                    # inside Finetuning_LLama
    Path("../data/final/kb_rules_extracted.jsonl"),                 # parent repo (local dev)
    Path("/workspace/Dataset_preprossecing_pipeline/data/final/kb_rules_extracted.jsonl"),  # RunPod alt
]
rules_path = None
for p in _candidates:
    if p.exists():
        rules_path = p
        break
if rules_path is None:
    print(f"ERROR: Rules file not found. Searched:")
    for p in _candidates:
        print(f"  - {p.resolve()}")
    print("\nTo fix: copy the file into this repo:")
    print("  cp /path/to/data/final/kb_rules_extracted.jsonl data/final/")
    print("  OR pass --rules-file /absolute/path/to/kb_rules_extracted.jsonl")
    # Continue without page updates (still add new chunks)
    rules_path = None

rule_source_map = {}
if rules_path:
    print(f"  Rules file: {rules_path.resolve()}")
    with open(rules_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rid = r.get("rule_id", "")
            src_book = r.get("source_book", "")
            src_page = r.get("source_doc", "")  # e.g. "page_105"
            if rid and (src_book or src_page):
                rule_source_map[rid] = {
                    "source_book": src_book,
                    "source_page": src_page,
                }
    print(f"  Rules with page info: {len(rule_source_map)}")
else:
    print("  Skipping page number updates (rules file not found)")

# ── Step 2: Load current KB chunks and find which need page updates ───────────
kb_path = Path(args.kb_chunks)
if not kb_path.exists():
    print(f"ERROR: KB chunks file not found: {kb_path}")
    exit(1)

chunks = []
needs_page_update = []
with open(kb_path, "r", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        chunks.append(c)
        refs = c.get("rule_refs", [])
        has_source = c.get("source_book") or c.get("source_page")
        if refs and not has_source:
            # Check if any rule_ref has source info
            for ref in refs:
                if ref in rule_source_map:
                    needs_page_update.append((c, rule_source_map[ref]))
                    break

print(f"  Total KB chunks: {len(chunks)}")
print(f"  Chunks needing page update: {len(needs_page_update)}")

# ── Step 3: Generate missing chunks ───────────────────────────────────────────
NEW_CHUNKS = []

if not args.pages_only:
    now = datetime.now(timezone.utc).isoformat()

    # 3a. Horary number-to-degree mapping (KP uses numbers 1-249)
    # Each number maps to a specific degree range in the zodiac
    horary_chunks = [
        {
            "text": "In KP Horary astrology, numbers 1 to 249 are used. Each number corresponds to a specific sub-lord division of the zodiac. Numbers 1-9 cover Aries (Ashwini nakshatra), with each number representing approximately 0°53'20\" of arc. The querent selects a number between 1 and 249, and the astrologer determines the ascendant degree, sign lord, star lord, and sub-lord from this number using the KP Table of Houses.",
            "rule_refs": ["KP_HOR_NUM_001"],
            "category": "horary",
        },
        {
            "text": "For horary number 147: This falls in Virgo sign (Mercury lord), Chitra nakshatra (Mars star lord). The lagna degree is calculated from the KP sub-lord table. Ruling planets at the time of judgment are: Ascendant sign lord, Ascendant star lord, Moon sign lord, Moon star lord, and Day lord. These ruling planets must agree with the significators of the houses under question for the prediction to be valid.",
            "rule_refs": ["KP_HOR_NUM_147"],
            "category": "horary",
        },
        {
            "text": "The 249 sub-lord divisions cover the entire zodiac of 360 degrees. Each of the 12 signs is divided into 9 nakshatras (stars), and each nakshatra is further divided into unequal sub-divisions based on Vimshottari dasha proportions. The sub-lord of the horary number determines the final outcome. If the sub-lord of the ascendant is a significator of the houses related to the query, the answer is favorable.",
            "rule_refs": ["KP_HOR_NUM_002"],
            "category": "horary",
        },
    ]

    # 3b. Vimshottari Dasa balance conversion
    dasa_chunks = [
        {
            "text": "Vimshottari Dasa periods for each planet: Sun 6 years, Moon 10 years, Mars 7 years, Rahu 18 years, Jupiter 16 years, Saturn 19 years, Mercury 17 years, Ketu 7 years, Venus 20 years. Total cycle is 120 years. The balance of dasa at birth is calculated from the Moon's position in its nakshatra.",
            "rule_refs": ["KP_DAS_PERIODS"],
            "category": "timing",
        },
        {
            "text": "To calculate dasa balance at birth: Find Moon's longitude in its nakshatra. Calculate the proportion of nakshatra already traversed. The remaining proportion of the nakshatra lord's dasa period gives the balance. Formula: Balance = Total_Dasa_Period × (Remaining_Nakshatra_Arc / Total_Nakshatra_Arc). Each nakshatra spans 13°20' (800 minutes of arc).",
            "rule_refs": ["KP_DAS_BALANCE"],
            "category": "timing",
        },
        {
            "text": "Dasa balance conversion to days: 1 year = 365.25 days. For example, if Mars dasa balance is 0Y 7M 23D, convert: 0×365.25 + 7×30.4375 + 23 = 236.06 days. The sub-periods (bhukti/antardasha) within each mahadasha follow the same Vimshottari proportions. Each planet's bhukti within a mahadasha = (Planet's total dasa / 120) × Mahadasha lord's total dasa.",
            "rule_refs": ["KP_DAS_CONVERT"],
            "category": "timing",
        },
    ]

    # 3c. Planet-House 9×12 matrix (key combinations)
    planet_house_chunks = []
    planets = {
        "Sun": "authority, government, father, vitality, soul, leadership",
        "Moon": "mind, mother, emotions, public, changes, nurturing",
        "Mars": "energy, courage, aggression, property, siblings, surgery",
        "Mercury": "intellect, communication, business, education, duality",
        "Jupiter": "wisdom, expansion, children, fortune, dharma, teaching",
        "Venus": "love, beauty, luxury, marriage, arts, comfort, romance",
        "Saturn": "discipline, delay, karma, labor, restriction, longevity",
        "Rahu": "obsession, foreign, unconventional, sudden, technology",
        "Ketu": "detachment, spirituality, past karma, moksha, sudden loss",
    }
    key_houses = {
        1: "self, personality, physical body",
        2: "wealth, family, speech",
        5: "children, romance, creativity, speculation",
        6: "disease, enemies, debts, obstacles",
        7: "marriage, spouse, partnerships",
        8: "longevity, inheritance, sudden events, occult",
        10: "career, status, profession",
        11: "gains, fulfillment, friends, aspirations",
        12: "losses, expenses, foreign lands, spirituality",
    }

    for planet, nature in planets.items():
        for house, domain in key_houses.items():
            pid = f"KP_PH_{planet[:3].upper()}_{house:02d}"
            text = (
                f"In KP astrology, when {planet} (natural significator of {nature}) "
                f"signifies the {house}th house ({domain}), "
                f"the native experiences {planet}'s qualities through {house}th house matters. "
                f"The sub-lord of {planet} determines whether results are favorable or unfavorable. "
                f"If {planet}'s sub-lord connects to houses 2,7,11 it gives positive results; "
                f"if connected to 6,8,12 it creates obstacles in {house}th house matters."
            )
            planet_house_chunks.append({
                "text": text,
                "rule_refs": [pid],
                "category": "planet_house_combo",
            })

    all_new = horary_chunks + dasa_chunks + planet_house_chunks
    base_id = len(chunks)
    for i, c in enumerate(all_new):
        c["chunk_id"] = f"kp_chunk_{base_id + i:04d}"
        c["tokens"] = len(c["text"].split()) * 2  # rough estimate
        c["tokenizer"] = "llama-3.1-tokenizer"
        c["source_file"] = "11_enrich_kb.py"
        c["language"] = "en"
        c["sha256"] = hashlib.sha256(c["text"].encode()).hexdigest()
        c["created_utc"] = now
        NEW_CHUNKS.append(c)

    print(f"\n  New chunks to add:")
    print(f"    Horary mappings:    {len(horary_chunks)}")
    print(f"    Dasa balance:       {len(dasa_chunks)}")
    print(f"    Planet-House 9×12:  {len(planet_house_chunks)}")
    print(f"    TOTAL NEW:          {len(NEW_CHUNKS)}")

# ── Step 4: Apply changes ─────────────────────────────────────────────────────
if args.dry_run:
    print("\n  DRY RUN — no changes applied.")
    if needs_page_update:
        print(f"\n  Sample page updates:")
        for c, src in needs_page_update[:5]:
            print(f"    {c['rule_refs']} → {src['source_book']}, {src['source_page']}")
    if NEW_CHUNKS:
        print(f"\n  Sample new chunks:")
        for c in NEW_CHUNKS[:3]:
            print(f"    [{c['rule_refs'][0]}] {c['text'][:80]}...")
    exit(0)

# 4a. Update kb_chunks.jsonl with page numbers
updated_count = 0
updated_chunks = []
page_lookup = {c["chunk_id"]: src for c, src in needs_page_update}

with open(kb_path, "r", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        if c["chunk_id"] in page_lookup:
            src = page_lookup[c["chunk_id"]]
            c["source_book"] = src["source_book"]
            c["source_page"] = src["source_page"]
            updated_count += 1
        updated_chunks.append(c)

# Append new chunks
for nc in NEW_CHUNKS:
    updated_chunks.append(nc)

# Write updated kb_chunks.jsonl
with open(kb_path, "w", encoding="utf-8") as f:
    for c in updated_chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"\n  Updated {updated_count} chunks with page numbers")
print(f"  Added {len(NEW_CHUNKS)} new chunks")
print(f"  Total chunks now: {len(updated_chunks)}")

# 4b. Update Pinecone metadata (page numbers for existing vectors)
if PINECONE_API_KEY and OPENAI_API_KEY:
    from pinecone import Pinecone
    from openai import OpenAI

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    oai = OpenAI(api_key=OPENAI_API_KEY)

    # Update existing vectors with page metadata
    print(f"\n  Updating Pinecone metadata for {updated_count} vectors...")
    batch = []
    for c, src in needs_page_update:
        batch.append({
            "id": c["chunk_id"],
            "metadata": {
                "source_book": src["source_book"],
                "source_page": src["source_page"],
            },
        })
        if len(batch) >= 100:
            index.update(id=batch[0]["id"], set_metadata=batch[0]["metadata"])
            # Pinecone update is per-vector, so loop
            for item in batch:
                index.update(id=item["id"], set_metadata=item["metadata"])
            batch = []
    # Flush remaining
    for item in batch:
        index.update(id=item["id"], set_metadata=item["metadata"])
    print(f"  ✓ Pinecone metadata updated for {updated_count} vectors")

    # Upsert new vectors
    if NEW_CHUNKS:
        print(f"\n  Embedding and upserting {len(NEW_CHUNKS)} new vectors...")
        upsert_batch = []
        for i in range(0, len(NEW_CHUNKS), 20):
            batch_chunks = NEW_CHUNKS[i:i+20]
            texts = [c["text"] for c in batch_chunks]
            emb_resp = oai.embeddings.create(
                model=EMBEDDING_MODEL, input=texts, dimensions=EMBEDDING_DIM
            )
            for j, c in enumerate(batch_chunks):
                vec = emb_resp.data[j].embedding
                meta = {
                    "text": c["text"],
                    "rule_refs": c["rule_refs"],
                    "category": c["category"],
                    "source_file": c["source_file"],
                    "language": c["language"],
                }
                upsert_batch.append({
                    "id": c["chunk_id"],
                    "values": vec,
                    "metadata": meta,
                })
            if len(upsert_batch) >= 100:
                index.upsert(vectors=upsert_batch)
                print(f"    Upserted {len(upsert_batch)} vectors...")
                upsert_batch = []
        if upsert_batch:
            index.upsert(vectors=upsert_batch)
            print(f"    Upserted {len(upsert_batch)} vectors...")

        stats = index.describe_index_stats()
        print(f"  ✓ Pinecone now has {stats['total_vector_count']} vectors")
else:
    print("\n  Pinecone keys not found — skipping Pinecone update.")
    print("  Run on RunPod with .env to update Pinecone.")

print(f"\n{'='*70}")
print("ENRICHMENT COMPLETE")
print(f"{'='*70}")
