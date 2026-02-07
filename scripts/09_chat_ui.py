"""
Gradio Chat UI for KP Astrology LLM — connects to the vLLM server.

Usage:
  # Start vLLM server first (in another terminal):
  python scripts/08_serve_vllm.py

  # Then start this UI:
  python scripts/09_chat_ui.py

  # The UI will be available at http://0.0.0.0:7860
  # Share publicly with --share flag:
  python scripts/09_chat_ui.py --share
"""

import argparse
import os
import re
import csv
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="KP Astrology Chat UI")
parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                    help="vLLM server URL")
parser.add_argument("--port", type=int, default=7860, help="Gradio UI port")
parser.add_argument("--share", action="store_true",
                    help="Create a public Gradio share link")
parser.add_argument("--no-rag", action="store_true",
                    help="Disable Pinecone RAG retrieval")
parser.add_argument("--top-k", type=int, default=5,
                    help="Number of RAG chunks to retrieve (default: 5)")
parser.add_argument("--max-model-len", type=int, default=2048,
                    help="vLLM max model length (default: 2048, use 4096 if GPU allows)")
parser.add_argument("--products-csv", type=str, default=None,
                    help="Path to products CSV for remedy recommendations")
args = parser.parse_args()

# ── Connect to vLLM backend ──────────────────────────────────────────────────
client = OpenAI(base_url=args.vllm_url, api_key="not-needed")

# ── RAG: Pinecone + OpenAI embeddings ────────────────────────────────────────
rag_index = None
openai_client = None
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

if not args.no_rag:
    try:
        from pinecone import Pinecone
        pc_key = os.getenv("PINECONE_API_KEY")
        oai_key = os.getenv("OPENAI_API_KEY")
        idx_name = os.getenv("PINECONE_INDEX_NAME", "kp-astrology-kb")

        if pc_key and oai_key and oai_key != "your-openai-api-key-here":
            pc = Pinecone(api_key=pc_key)
            rag_index = pc.Index(idx_name)
            openai_client = OpenAI(api_key=oai_key)
            stats = rag_index.describe_index_stats()
            print(f"  RAG:    Pinecone '{idx_name}' ({stats['total_vector_count']} vectors)")
        else:
            print("  RAG:    DISABLED (missing PINECONE_API_KEY or OPENAI_API_KEY)")
    except Exception as e:
        print(f"  RAG:    DISABLED (init error: {e})")
else:
    print("  RAG:    DISABLED (--no-rag flag)")

SYSTEM_BASE = (
    "KP astrology expert. Answer in same language as user (English or Hinglish). "
    "RULES: Use ONLY excerpts below. Quote exact text with [rule_id]. "
    "If source/page shown, cite it. NEVER invent pages/chapters. "
    "If not covered, say so. No repetition. "
    "Format: Answer, Quote, Rule ID, Source, Confidence(high/med/low)."
)

SYSTEM_NO_RAG = (
    "KP astrology expert. Answer in same language as user (English or Hinglish). "
    "Cite KP rules. Include confidence. "
    "Use KP terms: sub-lord, cusp, significator, nakshatra, dasha-bhukti. "
    "NEVER invent pages. Be concise. No repetition."
)

# ── Product catalog (for remedy recommendations) ─────────────────────────────
PRODUCT_CATALOG = []
if args.products_csv and os.path.isfile(args.products_csv):
    try:
        with open(args.products_csv, encoding="utf-8") as f:
            PRODUCT_CATALOG = list(csv.DictReader(f))
        print(f"  Products: {len(PRODUCT_CATALOG)} items loaded from {args.products_csv}")
    except Exception as e:
        print(f"  Products: FAILED ({e})")


# ── Context-window budget ─────────────────────────────────────────────────────
# Calibrated from actual vLLM errors:
#   ~1450 chars of content → 1865 actual tokens (ratio ≈ 0.78 chars/token)
#   Llama 3.1 chat template adds ~100 tokens overhead per conversation
# Strategy: use HARD CHARACTER BUDGET instead of unreliable token estimates.
MAX_MODEL_LEN = args.max_model_len
OUTPUT_TOKENS = min(400, MAX_MODEL_LEN // 4)  # scale with context window
INPUT_TOKEN_BUDGET = MAX_MODEL_LEN - OUTPUT_TOKENS - 100  # 100 for template
MAX_INPUT_CHARS = int(INPUT_TOKEN_BUDGET * 0.78)
print(f"  Budget:  max_model_len={MAX_MODEL_LEN}, output={OUTPUT_TOKENS}, input_chars≈{MAX_INPUT_CHARS}")


def _retrieve_rag_chunks(question, top_k=5):
    """Retrieve relevant KP book chunks from Pinecone. Returns list of formatted strings."""
    if not rag_index or not openai_client:
        return []
    try:
        resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL, input=question, dimensions=EMBEDDING_DIM
        )
        qvec = resp.data[0].embedding
        results = rag_index.query(vector=qvec, top_k=top_k, include_metadata=True)
        chunks = []
        for m in results["matches"]:
            txt = m["metadata"].get("text", "").strip()
            refs = m["metadata"].get("rule_refs", [])
            ref_str = ",".join(refs) if refs else "no_id"
            src = m["metadata"].get("source_book", "")
            page = m["metadata"].get("source_page", "")
            loc = f" (Source: {src}, {page})" if src and page else ""
            chunks.append(f"[{ref_str}]{loc} {txt}")
        return chunks
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return []


def _get_product_recommendations(question, max_items=3):
    """Find relevant products based on question keywords."""
    if not PRODUCT_CATALOG:
        return ""
    q_lower = question.lower()
    planet_product_map = {
        "venus": ["diamond", "opal", "white", "zircon"],
        "saturn": ["blue sapphire", "neelam", "karungali", "iron"],
        "jupiter": ["yellow sapphire", "pukhraj", "topaz", "rudraksha"],
        "mars": ["coral", "moonga", "red"],
        "mercury": ["emerald", "panna", "green"],
        "moon": ["pearl", "moti", "chandra"],
        "sun": ["ruby", "manik", "surya"],
        "rahu": ["hessonite", "gomed", "garnet"],
        "ketu": ["cat eye", "lehsunia", "vaidurya"],
    }
    keywords = []
    for planet, terms in planet_product_map.items():
        if planet in q_lower:
            keywords.extend(terms)
    if not keywords:
        return ""
    matches = []
    for p in PRODUCT_CATALOG:
        title_lower = p.get("Title", "").lower()
        if any(kw in title_lower for kw in keywords):
            matches.append(p)
    if not matches:
        return ""
    matches = matches[:max_items]
    lines = ["\nRelevant Remedies (from store):"]
    for p in matches:
        lines.append(f"- {p['Title']} (SKU: {p['SKU']}, Rs.{p['Sale Price']})")
    return "\n".join(lines)


def _postprocess(text):
    """Strip duplicate confidence/metadata blocks that the model sometimes repeats."""
    # Remove duplicate "Confidence: xxx" lines (keep first)
    seen_conf = False
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip().lower()
        is_conf = stripped.startswith("confidence:") or stripped.startswith("**confidence")
        is_rules = stripped.startswith("rules_used:") or stripped.startswith("rules used:")
        if is_conf or is_rules:
            if seen_conf:
                continue  # skip duplicate
            seen_conf = True
        cleaned.append(line)
    result = "\n".join(cleaned).rstrip()
    # Remove trailing incomplete sentences (cut off by token limit)
    if result and result[-1] not in '.!?"\n)}':
        last_period = max(result.rfind('. '), result.rfind('.\n'), result.rfind('.'))
        if last_period > len(result) * 0.7:  # only trim if we keep >70%
            result = result[:last_period + 1]
    return result


def predict(message, history):
    """Stream a response from the vLLM server with RAG-augmented context."""
    # 1. Retrieve RAG chunks
    rag_chunks = _retrieve_rag_chunks(message, top_k=args.top_k)

    # 2. Build prompt with adaptive RAG trimming to fit character budget
    #    Fixed parts: SYSTEM_BASE + user message
    fixed_chars = len(SYSTEM_BASE) + len(message) + 30  # 30 for labels
    rag_budget = MAX_INPUT_CHARS - fixed_chars

    # Add RAG chunks one by one until budget is exhausted
    selected_chunks = []
    used_chars = 0
    for chunk in rag_chunks:
        if used_chars + len(chunk) + 1 > rag_budget:
            break
        selected_chunks.append(chunk)
        used_chars += len(chunk) + 1  # +1 for newline

    if selected_chunks:
        rag_text = "\n".join(selected_chunks)
        sys_content = f"{SYSTEM_BASE}\n\nKP Book Excerpts:\n{rag_text}"
    else:
        sys_content = SYSTEM_NO_RAG

    # 3. Build messages (no history — every question gets fresh RAG)
    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": message},
    ]

    # 4. Final safety: compute actual char total and adjust output tokens
    total_chars = sum(len(m["content"]) for m in messages)
    est_input_tokens = int(total_chars / 0.78) + 100  # +100 for template
    available = MAX_MODEL_LEN - est_input_tokens
    max_tokens = max(64, min(OUTPUT_TOKENS, available))

    if max_tokens < 64:
        yield (f"Your message is too long for the model's {MAX_MODEL_LEN}-token context. "
               "Please ask a shorter question.")
        return

    # 5. Append product recommendations if relevant
    product_text = _get_product_recommendations(message)

    try:
        stream = client.chat.completions.create(
            model="kp-astrology-llama",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.4,
            top_p=0.9,
            stream=True,
            extra_body={"repetition_penalty": 1.15},
        )
        partial = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                partial += delta
                yield _postprocess(partial)
        # Append product recommendations after model response
        if product_text:
            partial += "\n" + product_text
            yield _postprocess(partial)
    except Exception as e:
        yield f"Error: {e}\n\nMake sure vLLM is running: python scripts/08_serve_vllm.py"


# ── Example questions ─────────────────────────────────────────────────────────
EXAMPLES = [
    "What does the 7th house sub-lord signify in marriage timing?",
    "How to predict career success using KP astrology?",
    "Explain Venus's role in KP astrology for relationships.",
    "How do I calculate ruling planets for a horary question?",
    "What is the 11th cusp sub-lord's significance for financial gains?",
    "Describe the Mahadasha-Antardasha system in KP astrology.",
]

# ── Build Gradio UI (compatible with Gradio 4.x and 5.x) ────────────────────
rag_status = "with RAG (Pinecone + OpenAI)" if rag_index else "without RAG"
demo = gr.ChatInterface(
    fn=predict,
    title="KP Astrology AI Assistant",
    description=(
        f"**Powered by fine-tuned Llama 3.1 8B** — {rag_status}\n\n"
        "Ask any question about KP astrology — marriage timing, career predictions, "
        "horary analysis, dasha periods, and more."
    ),
    examples=EXAMPLES,
    cache_examples=False,
)

print(f"\n{'='*60}")
print(f"  KP Astrology Chat UI")
print(f"  Local:  http://0.0.0.0:{args.port}")
if args.share:
    print(f"  Public: will be shown after launch")
print(f"  vLLM:   {args.vllm_url}")
print(f"{'='*60}\n")

demo.launch(
    server_name="0.0.0.0",
    server_port=args.port,
    share=args.share,
    show_error=True,
)
