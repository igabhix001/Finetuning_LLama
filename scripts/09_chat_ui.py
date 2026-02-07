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
import json
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
        # Truncate to ~500 chars to stay within OpenAI embedding 8192 token limit
        embed_text = question[:500]
        resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL, input=embed_text, dimensions=EMBEDDING_DIM
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


MAX_CHART_CHARS = 1800  # keep chart data compact to fit in 4096-token context


def _compact_chart_data(raw: str) -> str:
    """Parse computation-engine JSON and return a compact KP-essential summary.

    Keeps: birthDetails, planetKP (sign/nak/sub), cuspKP (sign/nak/sub),
           significators, planetSignifications, dashaBalance.
    Discards: raw positions, debug, full dasha tree (antardashas/pratyantars).
    If input is not valid JSON, truncate to MAX_CHART_CHARS.
    """
    raw = raw.strip()
    if not raw:
        return ""

    # ── Try JSON parse ────────────────────────────────────────────────────
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Not JSON — just truncate plain text
        if len(raw) > MAX_CHART_CHARS:
            return raw[:MAX_CHART_CHARS] + "\n[...chart data truncated]"
        return raw

    lines = []

    # 1. Birth details
    bd = d.get("birthDetails", {})
    name = d.get("name", "Unknown")
    gender = d.get("gender", "")
    lines.append(f"Native: {name}, {gender}")
    lines.append(f"DOB: {bd.get('date','?')}, TOB: {bd.get('time','?')}, "
                 f"Place: {bd.get('place','?')}")
    lines.append(f"Lagna: {bd.get('lagna','?')} ({bd.get('lagnaLord','?')}), "
                 f"Rasi: {bd.get('rasi','?')} ({bd.get('rasiLord','?')}), "
                 f"Nakshatra: {bd.get('nakshatra','?')} ({bd.get('nakshatraLord','?')})")

    # 2. Planet KP table (compact: sign, nak lord, sub lord, houses)
    pkp = d.get("planetKP", {})
    psig = d.get("planetSignifications", {})
    if pkp:
        lines.append("\nPLANET KP TABLE:")
        for planet in ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Rahu","Ketu"]:
            p = pkp.get(planet, {})
            if not p:
                continue
            deg = p.get("degree", "?")
            if isinstance(deg, float):
                deg = f"{deg:.2f}"
            houses = psig.get(planet, [])
            h_str = ",".join(str(h) for h in houses) if houses else "?"
            lines.append(f"  {planet}: {deg}° {p.get('rashi','?')}, "
                         f"Nak={p.get('nakshatra','?')}({p.get('nakshatraLord','?')}), "
                         f"Sub={p.get('sub','?')}, houses=[{h_str}]")

    # 3. Cusp KP table (compact)
    ckp = d.get("cuspKP", {})
    if ckp:
        lines.append("\nCUSP KP TABLE:")
        for c in [str(i) for i in range(1, 13)]:
            cp = ckp.get(c, {})
            if not cp:
                continue
            deg = cp.get("degree", "?")
            if isinstance(deg, float):
                deg = f"{deg:.2f}"
            lines.append(f"  Cusp {c}: {deg}° {cp.get('rashi','?')}, "
                         f"Nak={cp.get('nakshatra','?')}({cp.get('nakshatraLord','?')}), "
                         f"Sub={cp.get('sub','?')}")

    # 4. House significators
    sig = d.get("significators", {})
    if sig:
        lines.append("\nHOUSE SIGNIFICATORS:")
        for h in [str(i) for i in range(1, 13)]:
            planets = sig.get(h, [])
            if planets:
                lines.append(f"  House {h}: {','.join(planets)}")

    # 5. Dasha balance (top-level only, no nested tree)
    dashas = d.get("dashas", {})
    db = dashas.get("dashaBalance", {})
    if db:
        lines.append(f"\nDASHA BALANCE: {db.get('lord','?')} "
                     f"{db.get('years',0)}Y {db.get('months',0)}M {db.get('days',0)}D")
    # Top-level dasha periods (lord + dates only, no antar/pratyantars)
    dlist = dashas.get("dashas", [])
    if dlist:
        lines.append("MAHADASHA PERIODS:")
        for dd in dlist:
            lines.append(f"  {dd.get('lord','?')}: "
                         f"{dd.get('startDate','?')[:10]} to {dd.get('endDate','?')[:10]} "
                         f"({dd.get('period','?')})")

    result = "\n".join(lines)
    # Final safety truncation
    if len(result) > MAX_CHART_CHARS:
        result = result[:MAX_CHART_CHARS] + "\n[...chart data truncated]"
    return result


def predict(message, history, chart_data):
    """Stream a response from the vLLM server with RAG-augmented context + chart data."""
    # 0. Compact chart data (auto-parse large JSON from computation engine)
    chart_data = _compact_chart_data(chart_data or "")
    if chart_data:
        full_question = f"Chart Data from computation engine:\n{chart_data}\n\nQuestion: {message}"
    else:
        full_question = message

    # 1. Retrieve RAG chunks (search using original question for better retrieval)
    rag_chunks = _retrieve_rag_chunks(message, top_k=args.top_k)

    # 2. Build prompt with adaptive RAG trimming to fit character budget
    fixed_chars = len(SYSTEM_BASE) + len(full_question) + 30
    rag_budget = MAX_INPUT_CHARS - fixed_chars

    selected_chunks = []
    used_chars = 0
    for chunk in rag_chunks:
        if used_chars + len(chunk) + 1 > rag_budget:
            break
        selected_chunks.append(chunk)
        used_chars += len(chunk) + 1

    if selected_chunks:
        rag_text = "\n".join(selected_chunks)
        sys_content = f"{SYSTEM_BASE}\n\nKP Book Excerpts:\n{rag_text}"
    else:
        sys_content = SYSTEM_NO_RAG

    # 3. Build messages (no history — every question gets fresh RAG)
    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": full_question},
    ]

    # 4. Final safety: compute actual char total and adjust output tokens
    total_chars = sum(len(m["content"]) for m in messages)
    est_input_tokens = int(total_chars / 0.78) + 100
    available = MAX_MODEL_LEN - est_input_tokens
    max_tokens = max(64, min(OUTPUT_TOKENS, available))

    if max_tokens < 64:
        yield (f"Your message is too long for the model's {MAX_MODEL_LEN}-token context. "
               "Please shorten the chart data or question.")
        return

    # 5. Product recommendations
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
        if product_text:
            partial += "\n" + product_text
            yield _postprocess(partial)
    except Exception as e:
        yield f"Error: {e}\n\nMake sure vLLM is running: python scripts/08_serve_vllm.py"


# ── Sample chart template ─────────────────────────────────────────────────────
SAMPLE_CHART = """{
  "native": "TestUser",
  "dob": "01-01-1990",
  "tob": "10:00",
  "pob": "Mumbai",
  "lagna": "Aquarius",
  "cusps": {
    "7": {"degree": "122-12-49", "sign": "Leo", "sub_lord": "VEN", "nak_lord": "MON"},
    "2": {"degree": "42-15-30", "sign": "Taurus", "sub_lord": "JUP"},
    "11": {"degree": "312-45-10", "sign": "Aquarius", "sub_lord": "MAR"}
  },
  "planets": {
    "VEN": {"degree": "282-37-46", "sign": "Aquarius", "nak": "Dhanishta", "sub": "MAR", "houses_signified": [1,4,6,9,12]},
    "SUN": {"degree": "256-30-00", "sign": "Sagittarius", "houses_signified": [4,7,9,11,12]},
    "MER": {"degree": "270-15-20", "sign": "Capricorn", "houses_signified": [5,7,8,9,11,12]},
    "JUP": {"degree": "85-40-10", "sign": "Gemini", "houses_signified": [2,5,11,12]},
    "MAR": {"degree": "195-20-30", "sign": "Libra", "houses_signified": [1,3,10,11,12]}
  },
  "dasha": {"maha": "Jupiter", "antar": "Ketu", "balance": "MAR 0Y 7M 23D"},
  "house_significators": {
    "2": ["JUP","SUN"],
    "7": ["MER","SUN"],
    "11": ["JUP","MAR","MER","SAT","SUN"]
  }
}"""

EXAMPLE_QUESTIONS = [
    "Will marriage happen for this native? Analyze the 7th cusp sub-lord.",
    "What is the best dasha period for marriage in this chart?",
    "Analyze financial gains — check 11th cusp sub-lord significance.",
    "Is the current Jupiter-Ketu dasha favorable for career?",
    "What does Venus signify for relationships in this chart?",
]

# ── Build Gradio UI with Chart Data panel ─────────────────────────────────────
rag_status = "with RAG (Pinecone + OpenAI)" if rag_index else "without RAG"

with gr.Blocks(title="KP Astrology AI Assistant") as demo:
    gr.Markdown(
        f"# KP Astrology AI Assistant\n"
        f"**Powered by fine-tuned Llama 3.1 8B** — {rag_status}\n\n"
        "Paste your computation engine output (chart data) on the left, "
        "then ask questions on the right. The model will analyze the specific chart."
    )

    with gr.Row():
        # ── Left panel: Chart Data Input ──────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Chart Data (from Computation Engine)")
            chart_input = gr.Textbox(
                label="Paste full chart JSON here",
                placeholder="Paste the FULL JSON from your computation engine.\n"
                            "Large JSON is auto-compacted — only KP-essential fields "
                            "(planets, cusps, significators, dasha balance) are kept.",
                lines=20,
                max_lines=30,
            )
            with gr.Row():
                load_sample_btn = gr.Button("Load Sample Chart")
                clear_chart_btn = gr.Button("Clear Chart")

            gr.Markdown(
                "**How to use:**\n"
                "1. Your computation engine outputs chart data (planets, cusps, dashas)\n"
                "2. Paste the **full JSON** here — large files are auto-compacted\n"
                "3. Ask any KP astrology question on the right\n"
                "4. The model analyzes YOUR specific chart using KP rules"
            )

        # ── Right panel: Chat ─────────────────────────────────────────────
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="KP Astrology Chat",
                height=500,
            )
            msg_input = gr.Textbox(
                label="Ask a question about the chart",
                placeholder="e.g. Will marriage happen? Analyze 7th cusp sub-lord...",
                lines=2,
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")

            gr.Markdown("**Example questions** (click to fill):")
            example_btns = []
            with gr.Row():
                for eq in EXAMPLE_QUESTIONS[:3]:
                    b = gr.Button(eq, variant="secondary")
                    example_btns.append((b, eq))
            with gr.Row():
                for eq in EXAMPLE_QUESTIONS[3:]:
                    b = gr.Button(eq, variant="secondary")
                    example_btns.append((b, eq))

    # ── Event handlers ────────────────────────────────────────────────────
    def load_sample():
        return SAMPLE_CHART

    def clear_chart():
        return ""

    def user_submit(message, history, chart_data):
        """Add user message to history and stream bot response."""
        if not message.strip():
            yield history, ""
            return
        history = history + [{"role": "user", "content": message}]
        yield history, ""
        # Stream bot response
        partial_response = ""
        for chunk in predict(message, history, chart_data):
            partial_response = chunk
            # Replace or append assistant message
            if history and history[-1].get("role") == "assistant":
                history[-1]["content"] = partial_response
            else:
                history = history + [{"role": "assistant", "content": partial_response}]
            yield history, ""

    load_sample_btn.click(fn=load_sample, outputs=chart_input)
    clear_chart_btn.click(fn=clear_chart, outputs=chart_input)
    clear_btn.click(fn=lambda: [], outputs=chatbot)

    # Wire example question buttons
    for btn, question_text in example_btns:
        btn.click(fn=lambda q=question_text: q, outputs=msg_input)

    msg_input.submit(
        fn=user_submit,
        inputs=[msg_input, chatbot, chart_input],
        outputs=[chatbot, msg_input],
    )
    send_btn.click(
        fn=user_submit,
        inputs=[msg_input, chatbot, chart_input],
        outputs=[chatbot, msg_input],
    )

print(f"\n{'='*60}")
print(f"  KP Astrology Chat UI")
print(f"  Local:  http://0.0.0.0:{args.port}")
if args.share:
    print(f"  Public: will be shown after launch")
print(f"  vLLM:   {args.vllm_url}")
print(f"{'='*60}\n")

launch_kwargs = dict(
    server_name="0.0.0.0",
    server_port=args.port,
    share=args.share,
    show_error=True,
)
# Gradio 6+ moved theme to launch(); older versions use Blocks(theme=...)
try:
    demo.launch(**launch_kwargs, theme=gr.themes.Soft())
except TypeError:
    demo.launch(**launch_kwargs)
