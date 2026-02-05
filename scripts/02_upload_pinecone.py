"""
Upload OpenAI text-embedding-3-large embeddings to Pinecone for RAG.

Uploads 1,208 rule embeddings (3072-dim) to Pinecone index.
Also provides query functionality using OpenAI embeddings.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "kp-astrology-kb")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072

# Paths - try RunPod path first, then local
EMBEDDINGS_FILE = Path("./data/pinecone_upsert.jsonl")
if not EMBEDDINGS_FILE.exists():
    EMBEDDINGS_FILE = Path("/workspace/Finetuning_LLama/data/pinecone_upsert.jsonl")


def upload_to_pinecone():
    """Upload OpenAI embeddings to Pinecone."""
    
    print(f"\n{'='*70}")
    print("UPLOADING OPENAI EMBEDDINGS TO PINECONE")
    print(f"{'='*70}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Dimensions: {EMBEDDING_DIMENSION}")
    print(f"{'='*70}\n")
    
    # Verify API keys
    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY not found in .env file")
        sys.exit(1)
    
    # Initialize Pinecone (v3+ client)
    print(f"Connecting to Pinecone...")
    print(f"  Index: {PINECONE_INDEX_NAME}")
    
    try:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=PINECONE_API_KEY)
    except ImportError:
        print("❌ pinecone-client not installed. Run: pip install pinecone-client")
        sys.exit(1)
    
    # Check if index exists, create if not
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME in existing_indexes:
        # Check dimension matches
        idx_desc = pc.describe_index(PINECONE_INDEX_NAME)
        if idx_desc.dimension != EMBEDDING_DIMENSION:
            print(f"\n⚠️  Index '{PINECONE_INDEX_NAME}' has dimension {idx_desc.dimension}")
            print(f"   Expected dimension: {EMBEDDING_DIMENSION} (OpenAI text-embedding-3-large)")
            print(f"\n   You need to delete and recreate the index:")
            print(f"   pc.delete_index('{PINECONE_INDEX_NAME}')")
            print(f"   Then re-run this script.")
            
            user_input = input("\n   Delete and recreate index? (y/n): ").strip().lower()
            if user_input == 'y':
                print(f"   Deleting index '{PINECONE_INDEX_NAME}'...")
                pc.delete_index(PINECONE_INDEX_NAME)
                import time
                time.sleep(5)
            else:
                print("   Aborting.")
                sys.exit(1)
            existing_indexes = []
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"\n   Creating index '{PINECONE_INDEX_NAME}' with dimension {EMBEDDING_DIMENSION}...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for index to be ready
        import time
        print("   Waiting for index to be ready...")
        time.sleep(10)
        print(f"   ✓ Index created")
    
    # Connect to index
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Load embeddings
    print(f"\nLoading embeddings from: {EMBEDDINGS_FILE}")
    
    if not EMBEDDINGS_FILE.exists():
        print(f"❌ Embeddings file not found: {EMBEDDINGS_FILE}")
        sys.exit(1)
    
    embeddings = []
    with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                embeddings.append(json.loads(line))
    
    print(f"✓ Loaded {len(embeddings)} embeddings")
    
    # Verify dimension
    if embeddings and len(embeddings[0]['values']) != EMBEDDING_DIMENSION:
        print(f"❌ Embedding dimension mismatch!")
        print(f"   File has: {len(embeddings[0]['values'])} dimensions")
        print(f"   Expected: {EMBEDDING_DIMENSION} dimensions")
        print(f"   Please regenerate embeddings with OpenAI text-embedding-3-large")
        sys.exit(1)
    
    # Upload in batches
    batch_size = 100
    total_batches = (len(embeddings) + batch_size - 1) // batch_size
    
    print(f"\nUploading to Pinecone (batch size: {batch_size})...")
    
    for i in tqdm(range(0, len(embeddings), batch_size), total=total_batches):
        batch = embeddings[i:i + batch_size]
        
        vectors = [
            {
                "id": item['id'],
                "values": item['values'],
                "metadata": item['metadata']
            }
            for item in batch
        ]
        
        index.upsert(vectors=vectors)
    
    # Verify upload
    import time
    time.sleep(2)
    stats = index.describe_index_stats()
    
    print(f"\n{'='*70}")
    print("UPLOAD COMPLETE")
    print(f"{'='*70}\n")
    print(f"Total vectors in index: {stats['total_vector_count']}")
    print(f"Index dimension: {stats.get('dimension', EMBEDDING_DIMENSION)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"\n✅ OpenAI embeddings uploaded successfully!")
    print(f"\nRAG query flow:")
    print(f"  1. Embed user question with OpenAI {EMBEDDING_MODEL}")
    print(f"  2. Query Pinecone for top-k relevant rules")
    print(f"  3. Include rules as context in LLM prompt")
    print(f"\n{'='*70}\n")


def test_query():
    """Test Pinecone query using OpenAI embeddings."""
    
    print(f"\n{'='*70}")
    print("TESTING PINECONE QUERY WITH OPENAI EMBEDDINGS")
    print(f"{'='*70}\n")
    
    # Verify OpenAI key
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
        print("❌ OPENAI_API_KEY not found. Skipping query test.")
        return
    
    # Initialize clients
    from pinecone import Pinecone
    from openai import OpenAI
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Test questions
    test_questions = [
        "What is the significance of the 7th house sub-lord in marriage prediction?",
        "How to predict career success using KP astrology?",
        "What role does Dasha period play in timing events?"
    ]
    
    for q_idx, question in enumerate(test_questions, 1):
        print(f"\nQuery {q_idx}: {question}")
        
        # Create embedding with OpenAI
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=question,
            dimensions=EMBEDDING_DIMENSION
        )
        query_vector = response.data[0].embedding
        
        # Query Pinecone
        results = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
        
        print(f"  Top 3 results:")
        for i, match in enumerate(results['matches'], 1):
            print(f"  {i}. [{match['score']:.4f}] {match['id']}")
            print(f"     Category: {match['metadata'].get('category', 'N/A')}")
            print(f"     Text: {match['metadata'].get('text', '')[:100]}...")
    
    print(f"\n✅ Query test successful!")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    upload_to_pinecone()
    
    # Optionally test query
    test_query()
