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
    "You are an expert KP (Krishnamurti Paddhati) Astrology assistant.\n"
    "STRICT RULES:\n"
    "1. Answer using ONLY the KP Book Excerpts below. Quote the exact text.\n"
    "2. Cite only rule IDs that appear in the excerpts (e.g. [KP_MAR_0673]).\n"
    "3. NEVER invent page numbers, chapter numbers, or book locations.\n"
    "4. If the excerpts do not contain the answer, say: 'The retrieved excerpts do not cover this. Low confidence.'\n"
    "5. Do NOT repeat yourself. End your answer after the conclusion.\n"
    "6. Format: Answer → Exact quote → Rule ID → Confidence (high/medium/low)."
)

SYSTEM_NO_RAG = (
    "You are an expert KP (Krishnamurti Paddhati) Astrology assistant.\n"
    "Cite KP rules when applicable. Include confidence level.\n"
    "Use KP terminology: sub-lord, cusp, significator, nakshatra, dasha-bhukti.\n"
    "NEVER invent page numbers or book locations. Be concise. Do NOT repeat yourself."
)


# ── Context-window budget constants ───────────────────────────────────────────
MAX_MODEL_LEN = 2048
CHARS_PER_TOKEN = 2          # conservative: overestimates token count
MSG_OVERHEAD = 15            # special tokens per chat message (template)
GLOBAL_OVERHEAD = 50         # BOS + generation prompt + safety margin
DEFAULT_OUTPUT_TOKENS = 300  # solid KP answer with quote
MIN_OUTPUT_TOKENS = 64
MAX_HISTORY_TURNS = 0        # no history — maximize RAG context + output budget


def _est_tok(text):
    """Conservative token estimate (~2 chars/token for Llama 3.1)."""
    return len(text) // CHARS_PER_TOKEN + 1


def _est_msgs_tok(messages):
    """Estimate total tokens for a list of chat messages including template overhead."""
    total = GLOBAL_OVERHEAD
    for m in messages:
        total += _est_tok(m["content"]) + MSG_OVERHEAD
    return total


def _retrieve_rag_context(question, top_k=5, max_chars=900):
    """Retrieve relevant KP book chunks from Pinecone via OpenAI embedding."""
    if not rag_index or not openai_client:
        return ""
    try:
        resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL, input=question, dimensions=EMBEDDING_DIM
        )
        qvec = resp.data[0].embedding
        results = rag_index.query(vector=qvec, top_k=top_k, include_metadata=True)
        chunks = []
        total_chars = 0
        for m in results["matches"]:
            txt = m["metadata"].get("text", "").strip()
            refs = m["metadata"].get("rule_refs", [])
            cat = m["metadata"].get("category", "")
            score = m["score"]
            ref_str = ",".join(refs) if refs else "no_id"
            entry = f"[{ref_str}|{cat}|{score:.2f}] {txt}"
            if total_chars + len(entry) > max_chars:
                break
            chunks.append(entry)
            total_chars += len(entry)
        return "\n".join(chunks)
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return ""


def predict(message, history):
    """Stream a response from the vLLM server with RAG-augmented context."""
    # 1. Retrieve relevant KP book passages
    rag_context = _retrieve_rag_context(message, top_k=args.top_k, max_chars=900)

    # 2. Build system prompt with or without RAG context
    if rag_context:
        sys_content = f"{SYSTEM_BASE}\n\nKP Book Excerpts:\n{rag_context}"
    else:
        sys_content = SYSTEM_NO_RAG

    # 3. Parse history into pairs of (user_msg, assistant_msg)
    pairs = []
    if history:
        pending_user = None
        for h in history:
            if isinstance(h, dict):
                if h["role"] == "user":
                    pending_user = h["content"]
                elif h["role"] == "assistant" and pending_user is not None:
                    pairs.append((pending_user, h["content"]))
                    pending_user = None
            elif isinstance(h, (list, tuple)) and len(h) == 2:
                if h[0] and h[1]:
                    pairs.append((h[0], h[1]))

    # 4. Keep only the most recent turn
    pairs = pairs[-MAX_HISTORY_TURNS:]

    # 5. Build messages list
    messages = [{"role": "system", "content": sys_content}]
    for u, a in pairs:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    # 6. Token budget: compute available output
    input_est = _est_msgs_tok(messages)
    available = MAX_MODEL_LEN - input_est
    max_tokens = max(MIN_OUTPUT_TOKENS, min(DEFAULT_OUTPUT_TOKENS, available))

    # 7. If too tight, drop history
    while available < MIN_OUTPUT_TOKENS and pairs:
        pairs.pop(0)
        messages = [{"role": "system", "content": sys_content}]
        for u, a in pairs:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": message})
        input_est = _est_msgs_tok(messages)
        available = MAX_MODEL_LEN - input_est
        max_tokens = max(MIN_OUTPUT_TOKENS, min(DEFAULT_OUTPUT_TOKENS, available))

    # 8. Final guard
    if available < MIN_OUTPUT_TOKENS:
        yield ("Your message is too long for the model's context window (2048 tokens). "
               "Please ask a shorter or more specific question.")
        return

    try:
        stream = client.chat.completions.create(
            model="kp-astrology-llama",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.15,
            stream=True,
        )
        partial = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                partial += delta
                yield partial
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
