"""
Build a Pinecone vector index for product recommendations (Layer 2 — Product KB).

Instead of crude keyword matching, this embeds each product with its astrological
properties so the system can semantically match products to chart analysis context.

Usage:
  python scripts/12_build_product_index.py --csv /workspace/products_export_2026-02-03.csv

Requires: PINECONE_API_KEY, OPENAI_API_KEY in .env
"""

import argparse
import csv
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

parser = argparse.ArgumentParser(description="Build product Pinecone index")
parser.add_argument("--csv", type=str, required=True, help="Path to products CSV")
parser.add_argument("--index-name", type=str, default="kp-products",
                    help="Pinecone index name for products")
parser.add_argument("--batch-size", type=int, default=50, help="Upsert batch size")
args = parser.parse_args()

# ── Astrological product mapping ─────────────────────────────────────────────
# Maps product keywords to astrological context for richer embeddings
ASTRO_CONTEXT = {
    "rudraksha": "spiritual protection, Jupiter remedy, meditation, general blessing, obstacle removal",
    "1 mukhi": "Sun remedy, leadership, confidence, career success, government favor",
    "2 mukhi": "Moon remedy, emotional balance, relationships, marriage harmony",
    "3 mukhi": "Mars remedy, courage, energy, blood-related health, Mangal dosha",
    "4 mukhi": "Mercury remedy, intelligence, communication, education, business",
    "5 mukhi": "Jupiter remedy, general well-being, health, spiritual growth",
    "6 mukhi": "Venus remedy, love, marriage, luxury, artistic talent, relationships",
    "7 mukhi": "Saturn remedy, wealth, career stability, Shani dosha, obstacles",
    "8 mukhi": "Rahu remedy, protection from sudden events, Rahu dosha",
    "9 mukhi": "Ketu remedy, spiritual liberation, past karma, moksha",
    "10 mukhi": "protection from negative energies, Navagraha balance",
    "11 mukhi": "Hanuman blessing, courage, Mars remedy, protection",
    "12 mukhi": "Sun remedy, leadership, government job, authority",
    "13 mukhi": "Venus remedy, attraction, love, luxury, Shukra",
    "14 mukhi": "Saturn remedy, third eye, intuition, divine protection",
    "diamond": "Venus remedy, love, marriage, luxury, Shukra strengthening",
    "opal": "Venus remedy, creativity, love, relationships",
    "blue sapphire": "Saturn remedy, career, discipline, Shani dosha, neelam",
    "neelam": "Saturn remedy, career stability, Shani protection",
    "yellow sapphire": "Jupiter remedy, wealth, wisdom, Guru strengthening, pukhraj",
    "pukhraj": "Jupiter remedy, fortune, education, marriage for women",
    "coral": "Mars remedy, courage, energy, Mangal dosha, moonga",
    "moonga": "Mars remedy, vitality, blood health, courage",
    "emerald": "Mercury remedy, intelligence, business, communication, panna",
    "pearl": "Moon remedy, emotional peace, mental health, moti",
    "ruby": "Sun remedy, confidence, government favor, leadership, manik",
    "hessonite": "Rahu remedy, protection, gomed, sudden gains",
    "gomed": "Rahu remedy, career breakthrough, protection from Rahu",
    "cat eye": "Ketu remedy, spiritual growth, moksha, lehsunia",
    "karungali": "Saturn protection, Shani dosha remedy, negative energy removal, ebony wood",
    "evil eye": "protection from negative energies, nazar, jealousy shield",
    "kavach": "protection pendant, planetary shield, dosha remedy",
    "chakra": "energy balance, healing, holistic wellness, meditation",
    "navratna": "all nine planets remedy, complete planetary balance, Navagraha",
    "hanuman": "Mars remedy, courage, protection, obstacle removal",
    "ganesh": "obstacle removal, new beginnings, wisdom, Jupiter blessing",
    "shiva": "spiritual protection, meditation, Ketu remedy, moksha",
    "krishna": "love, devotion, Mercury remedy, relationships",
    "lakshmi": "wealth, prosperity, Venus-Jupiter remedy, financial gains",
    "trishool": "protection, Shiva blessing, negative energy removal",
    "om": "universal protection, spiritual growth, meditation aid",
    "pyrite": "wealth attraction, financial prosperity, abundance",
    "tiger eye": "courage, confidence, Sun-Mars remedy, protection",
    "amethyst": "spiritual growth, intuition, Ketu-Saturn remedy, healing",
    "bracelet": "daily wear remedy, continuous planetary support",
    "mala": "meditation, japa, spiritual practice, mantra chanting",
    "pendant": "personal protection, planetary remedy, daily wear",
    "necklace": "personal adornment, planetary remedy, daily wear",
    "idol": "home temple, puja, divine blessing, vastu remedy",
    "frame": "vastu remedy, home protection, positive energy",
}


def enrich_product_text(title: str, sku: str, price: str) -> str:
    """Create a rich text description for embedding by adding astrological context."""
    title_lower = title.lower()
    contexts = []
    for keyword, context in ASTRO_CONTEXT.items():
        if keyword in title_lower:
            contexts.append(context)
    astro_text = ". ".join(set(contexts)) if contexts else "spiritual product, general blessing"
    return (
        f"Product: {title}. SKU: {sku}. Price: Rs.{price}. "
        f"Astrological use: {astro_text}"
    )


def main():
    pc_key = os.getenv("PINECONE_API_KEY")
    oai_key = os.getenv("OPENAI_API_KEY")
    if not pc_key or not oai_key:
        print("ERROR: Set PINECONE_API_KEY and OPENAI_API_KEY in .env")
        return

    # Load products
    with open(args.csv, encoding="utf-8") as f:
        products = list(csv.DictReader(f))
    print(f"Loaded {len(products)} products from {args.csv}")

    # Initialize clients
    pc = Pinecone(api_key=pc_key)
    oai = OpenAI(api_key=oai_key)

    # Create or connect to index
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIM = 3072

    existing = [idx.name for idx in pc.list_indexes()]
    if args.index_name not in existing:
        print(f"Creating Pinecone index '{args.index_name}'...")
        pc.create_index(
            name=args.index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(args.index_name).status.get("ready", False):
            print("  Waiting for index to be ready...")
            time.sleep(2)
    else:
        print(f"Index '{args.index_name}' already exists, will upsert.")

    index = pc.Index(args.index_name)

    # Build enriched texts and embed
    texts = []
    metadata_list = []
    for p in products:
        title = p.get("Title", "")
        sku = p.get("SKU", "")
        price = p.get("Sale Price", "")
        enriched = enrich_product_text(title, sku, price)
        texts.append(enriched)
        metadata_list.append({
            "sku": sku,
            "title": title,
            "price": price,
            "text": enriched,
        })

    print(f"Embedding {len(texts)} products...")
    # Embed in batches of 100 (OpenAI limit)
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = oai.embeddings.create(model=EMBEDDING_MODEL, input=batch, dimensions=EMBEDDING_DIM)
        all_embeddings.extend([d.embedding for d in resp.data])
        print(f"  Embedded {min(i+100, len(texts))}/{len(texts)}")

    # Upsert to Pinecone
    print(f"Upserting to Pinecone index '{args.index_name}'...")
    vectors = []
    for i, (emb, meta) in enumerate(zip(all_embeddings, metadata_list)):
        vectors.append({
            "id": f"product-{meta['sku']}",
            "values": emb,
            "metadata": meta,
        })

    for i in range(0, len(vectors), args.batch_size):
        batch = vectors[i:i+args.batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted {min(i+args.batch_size, len(vectors))}/{len(vectors)}")

    print(f"\nDone! {len(vectors)} products indexed in '{args.index_name}'")
    stats = index.describe_index_stats()
    print(f"Index stats: {stats['total_vector_count']} vectors")


if __name__ == "__main__":
    main()
