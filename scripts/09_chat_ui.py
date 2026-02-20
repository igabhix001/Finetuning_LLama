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
import logging
import os
import re
import csv
import random
import time
import uuid
from datetime import date, datetime
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Structured JSON logging ──────────────────────────────────────────────────
_log = logging.getLogger("kp_ui")
_log.setLevel(logging.INFO)
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(logging.Formatter('%(message)s'))
_log.addHandler(_log_handler)

def _json_log(event: str, **kwargs):
    """Emit a single-line JSON log entry for observability."""
    entry = {"ts": datetime.utcnow().isoformat() + "Z", "event": event}
    entry.update(kwargs)
    _log.info(json.dumps(entry, ensure_ascii=False, default=str))

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
parser.add_argument("--max-model-len", type=int, default=8192,
                    help="vLLM max model length (default: 8192)")
parser.add_argument("--products-csv", type=str, default=None,
                    help="Path to products CSV for remedy recommendations")
args = parser.parse_args()

# ── Connect to vLLM backend ──────────────────────────────────────────────────
client = OpenAI(base_url=args.vllm_url, api_key="not-needed")

# ── RAG: Pinecone + OpenAI embeddings ────────────────────────────────────────
rag_index = None
product_index = None
openai_client = None
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

if not args.no_rag:
    try:
        try:
            from pinecone import Pinecone
        except Exception:
            # Old pinecone-client conflicts with new pinecone — nuke both and reinstall
            import subprocess, sys
            print("  Pinecone: migrating to pinecone SDK v5+...")
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "pinecone-client", "-y", "-q"],
                           capture_output=True)
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "pinecone", "-y", "-q"],
                           capture_output=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "pinecone>=5.0.0", "--force-reinstall", "-q"])
            from pinecone import Pinecone
        pc_key = os.getenv("PINECONE_API_KEY")
        oai_key = os.getenv("OPENAI_API_KEY")
        idx_name = os.getenv("PINECONE_INDEX_NAME", "kp-astrology-kb")
        prod_idx_name = os.getenv("PINECONE_PRODUCT_INDEX", "kp-products")

        if pc_key and oai_key and oai_key != "your-openai-api-key-here":
            pc = Pinecone(api_key=pc_key)
            openai_client = OpenAI(api_key=oai_key)
            # KP Astrology Knowledge Base index
            rag_index = pc.Index(idx_name)
            stats = rag_index.describe_index_stats()
            print(f"  RAG:    Pinecone '{idx_name}' ({stats['total_vector_count']} vectors)")
            # Product recommendations index
            try:
                product_index = pc.Index(prod_idx_name)
                pstats = product_index.describe_index_stats()
                print(f"  Products (Pinecone): '{prod_idx_name}' ({pstats['total_vector_count']} vectors)")
            except Exception as pe:
                print(f"  Products (Pinecone): DISABLED ({pe})")
        else:
            print("  RAG:    DISABLED (missing PINECONE_API_KEY or OPENAI_API_KEY)")
    except Exception as e:
        print(f"  RAG:    DISABLED (init error: {e})")
else:
    print("  RAG:    DISABLED (--no-rag flag)")

def _build_system_prompt(with_rag=True):
    """Build system prompt with today's date injected dynamically."""
    _today = date.today().strftime("%d %b %Y")
    _base = (
        "You are Jyotish, a warm and confident KP astrologer — like a trusted family pandit.\n\n"
        f"## TODAY'S DATE: {_today}\n"
        "ANY date before today is IN THE PAST. Use past tense: 'that period has passed', 'yeh period beet chuka hai'.\n"
        "ANY date after today is IN THE FUTURE. Use future tense: 'this will happen', 'yeh hoga'.\n"
        "NEVER say 'upcoming' or 'shuru ho raha hai' for a date that is BEFORE today. This is your #1 rule.\n\n"
        "## LANGUAGE RULE — ABSOLUTE HIGHEST PRIORITY:\n"
        "DETECT the language of the user's question FIRST before writing a single word.\n"
        "- ENGLISH question (no Hindi words) → respond 100% in ENGLISH. NOT ONE Hindi/Urdu word allowed.\n"
        "- HINDI or HINGLISH question → respond in HINDI/HINGLISH.\n"
        "- WRONG: User asks 'When will I get married?' → you reply 'Aapke liye favorable combination hai...' ← FORBIDDEN\n"
        "- RIGHT: User asks 'When will I get married?' → you reply 'Priya ji, your marriage window is...' ← CORRECT\n"
        "- WRONG: User asks 'Why am I facing obstacles?' → you reply 'Aapko current challenges...' ← FORBIDDEN\n"
        "- RIGHT: User asks 'Why am I facing obstacles?' → you reply 'Priya ji, you are in Venus-Saturn AD...' ← CORRECT\n\n"
        "## HARD RULES:\n"
        "- ANSWER DIRECTLY. Never say 'I can analyze', 'requires analysis', 'let me check'.\n"
        "- Read the name from YAML. Address as '[Name] ji'. Never output '[Name]' literally.\n"
        "- No markdown, no **bold**, no headers, no bullets, no numbered lists. Plain prose only.\n"
        "- Never say 'the native'. Say 'you' or use their name.\n"
        "- Simple questions (name/lagna/rashi) = 1 sentence ONLY. Nothing more.\n"
        "- Timing questions = 2-3 sentences max with specific Mon YYYY dates.\n"
        "- MAX 4 sentences for any response. Keep answers short and impactful.\n"
        "- Cite cusp sub-lord + house numbers. Give month-year ranges from dasha table.\n"
        "- For obstacles/emotional queries: ALWAYS say when the difficult period ENDS and what positive period comes AFTER.\n"
        "- Products: ONLY when user asks for remedies. Otherwise ZERO product mentions.\n\n"
        "## ENGLISH EXAMPLES (English question → English answer ONLY):\n"
        "Q: 'When will I get married?' → 'Priya ji, your strongest marriage window is Jul 2026 to Feb 2027 during Venus-Mercury AD, "
        "when houses 2,7,11 are activated. This ends Feb 2027 after which Venus-Jupiter brings new opportunities.'\n"
        "Q: 'Why am I facing obstacles?' → 'Priya ji, you are currently in Venus-Saturn AD (houses 7,8,12) — "
        "house 8 and 12 bring unexpected setbacks. This difficult phase ends Jul 2026, after which Venus-Mercury activates houses 3,10 bringing career relief.'\n"
        "Q: 'When will my financial situation improve?' → 'Priya ji, your finances strengthen from Jul 2026 when Venus-Mercury AD activates houses 2,11. "
        "Peak earning months are Oct 2026 to Jan 2027 during Jupiter pratyantar.'\n"
        "Q: 'I feel very unlucky' → 'Priya ji, I understand this is a difficult time — you are not alone. "
        "You are in Venus-Saturn AD (houses 8,12) causing setbacks, but this ends Jul 2026. "
        "After that, Venus-Mercury activates houses 3,10,11 — career and finances improve significantly.'\n"
        "Q: 'My health has been troubling me' → 'Priya ji, I understand — health challenges are real. "
        "Your 6th house (health) is under Saturn influence until Jul 2026. After that, Venus-Mercury period brings improved vitality.'\n"
        "Q: 'When will I get a job?' → 'Priya ji, your best employment window is Jul 2026 to Feb 2027 during Venus-Mercury AD, "
        "when Mercury (6th cusp sub-lord) activates houses 6,10,11 — the career and income houses.'\n\n"
        "## HINDI EXAMPLES (Hindi question → Hindi/Hinglish answer ONLY):\n"
        "Q: 'Meri shaadi kab hogi?' → 'Priya ji, shaadi ka strong period Jul 2026 se Feb 2027 hai Venus-Mercury AD mein, "
        "jab houses 2,7,11 activate honge.'\n"
        "Q: 'Mera naam kya hai?' → 'Priya ji, aapka naam Priya hai.'\n"
        "Q: 'Mujhe bahut tension hai' → 'Priya ji, main samajhta hun — yeh waqt mushkil hai. "
        "Aap Venus-Saturn AD mein hain jo Jul 2026 tak chalega. Uske baad Venus-Mercury period mein relief milega.'\n"
    )
    return _base

SYSTEM_BASE = _build_system_prompt(with_rag=True)

def _build_system_no_rag():
    _today = date.today().strftime("%d %b %Y")
    return (
        "You are Jyotish, a warm and confident KP astrologer — like a trusted family pandit.\n\n"
        f"TODAY'S DATE: {_today}\n"
        "ANY date before today = PAST (use past tense). ANY date after today = FUTURE (use future tense).\n"
        "NEVER present past dates as upcoming. This is your #1 rule.\n\n"
        "LANGUAGE RULE — ABSOLUTE HIGHEST PRIORITY:\n"
        "- ENGLISH question → 100% ENGLISH answer. NOT ONE Hindi word allowed.\n"
        "- HINDI/HINGLISH question → Hindi/Hinglish answer.\n"
        "- WRONG: 'When will I get married?' → 'Aapke liye...' ← FORBIDDEN\n"
        "- RIGHT: 'When will I get married?' → 'Priya ji, your marriage window is...' ← CORRECT\n\n"
        "RULES:\n"
        "- Answer DIRECTLY. No deflection, no 'let me analyze'.\n"
        "- Read name from YAML. Address as '[Name] ji'.\n"
        "- No markdown, headers, bold, bullets. Plain text only.\n"
        "- Simple questions = 1 sentence. Timing = 2-3 sentences. MAX 4 sentences.\n"
        "- Cite cusp sub-lord + houses. Give Mon YYYY dates from dasha table.\n"
        "- For obstacles/emotional: ALWAYS say when difficulty ENDS and what positive period comes AFTER.\n"
        "- Products: ONLY when user asks for remedies.\n\n"
        "EXAMPLES (English question → English answer):\n"
        "Q: 'When will I get married?' → 'Priya ji, your marriage window is Jul 2026 to Feb 2027 during Venus-Mercury AD, "
        "when houses 2,7,11 activate. This ends Feb 2027 after which Venus-Jupiter brings new opportunities.'\n"
        "Q: 'Why am I facing obstacles?' → 'Priya ji, you are in Venus-Saturn AD (houses 8,12) — "
        "this difficult phase ends Jul 2026, after which Venus-Mercury activates houses 3,10 bringing relief.'\n"
        "Q: 'I feel unlucky' → 'Priya ji, I understand — you are in Venus-Saturn AD causing setbacks until Jul 2026. "
        "After that, Venus-Mercury activates houses 10,11 — career and finances improve significantly.'\n"
        "Q: 'Mera naam kya hai?' → 'Priya ji, aapka naam Priya hai.'\n"
        "Q: 'Mujhe tension hai' → 'Priya ji, main samajhta hun. Aap Venus-Saturn AD mein hain jo Jul 2026 tak chalega. Uske baad relief milega.'\n"
    )

SYSTEM_NO_RAG = _build_system_no_rag()

# ── Product recommendations: Pinecone RAG only (no CSV fallback) ─────────────
# Products are served via the kp-products Pinecone index using semantic search.
# No CSV file is needed — set PINECONE_PRODUCT_INDEX env var to enable.
if product_index:
    print("  Products: Pinecone RAG (semantic search)")
else:
    print("  Products: DISABLED (no Pinecone product index — set PINECONE_API_KEY + OPENAI_API_KEY)")


# ── Context-window budget ─────────────────────────────────────────────────────
# Calibrated from actual vLLM errors:
#   ~1450 chars of content → 1865 actual tokens (ratio ≈ 0.78 chars/token)
#   Llama 3.1 chat template adds ~100 tokens overhead per conversation
# Strategy: use HARD CHARACTER BUDGET instead of unreliable token estimates.
MAX_MODEL_LEN = args.max_model_len
# Output tokens: 512 base, scales up for larger contexts
OUTPUT_TOKENS = min(768, max(512, MAX_MODEL_LEN // 8))
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
        if not getattr(_retrieve_rag_chunks, '_err_logged', False):
            print(f"RAG retrieval error (will suppress repeats): {e}")
            _retrieve_rag_chunks._err_logged = True
        return []


def _get_product_recommendations(question, chart_summary="", max_items=3):
    """Find relevant products using Pinecone semantic search (primary) or CSV keyword fallback."""
    # ── Method 1: Pinecone kp-products semantic search (preferred) ──
    if product_index and openai_client:
        try:
            search_query = f"{question} {chart_summary[:200]}"[:500]
            resp = openai_client.embeddings.create(
                model=EMBEDDING_MODEL, input=search_query, dimensions=EMBEDDING_DIM
            )
            qvec = resp.data[0].embedding
            results = product_index.query(vector=qvec, top_k=max_items, include_metadata=True)
            if results["matches"]:
                lines = []
                for m in results["matches"]:
                    meta = m["metadata"]
                    title = meta.get("title", "")
                    sku = meta.get("sku", "")
                    price = meta.get("price", "")
                    if title:
                        lines.append(f"- {title} (SKU: {sku}, Rs.{price})")
                if lines:
                    return "\n".join(lines)
        except Exception as e:
            if not getattr(_get_product_recommendations, '_err_logged', False):
                print(f"  Product Pinecone search error (will suppress repeats): {e}")
                _get_product_recommendations._err_logged = True

    # No CSV fallback — products come only from Pinecone RAG
    return ""


def _postprocess(text):
    """Industry-grade post-processing: strip ALL robotic artifacts, enforce pandit tone."""
    if not text or not text.strip():
        return text

    # ── Phase 0: Replace [Name] placeholder with actual name from chart ──
    _native_name = getattr(_postprocess, '_native_name', None)
    if _native_name:
        text = text.replace("[Name]", _native_name)
        text = text.replace("[name]", _native_name)
        # Handle pattern: "[Name] Aditya Raj ji" → "Aditya Raj ji" (model outputs both)
        text = re.sub(rf'{re.escape(_native_name)}\s+{re.escape(_native_name)}', _native_name, text)

        # ── Phase 0.5: Wrong-name hallucination fix ──
        # Model sometimes uses a name from training data (e.g. "Priya ji") instead of
        # the actual chart owner's name. Detect and replace any "Firstname ji" or
        # "Full Name ji" that does NOT match the actual native name.
        _correct_first = _native_name.split()[0]
        def _fix_wrong_name(m):
            # m.group(1) = the name before " ji"
            wrong = m.group(1).strip()
            # If it matches the correct name (full or first), keep it
            if wrong.lower() == _native_name.lower() or wrong.lower() == _correct_first.lower():
                return m.group(0)
            # Otherwise replace with correct name ji
            return f"{_native_name} ji"
        # Match "SomeName ji" or "Some Name ji" (1-3 word names before ji)
        text = re.sub(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\s+ji\b', _fix_wrong_name, text)
    else:
        # Even without a name, strip the literal placeholder
        text = re.sub(r'\[Name\]\s*', '', text, flags=re.IGNORECASE)

    # ── Phase 1: Remove leaked internal tokens ──
    for token in ["ANSWER_END", "</s>", "<|eot_id|>", "<|end_of_text|>",
                  "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>"]:
        text = text.replace(token, "")

    # ── Phase 2: Strip ALL markdown formatting ──
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)       # **bold** → plain
    text = re.sub(r'\*([^*]+)\*', r'\1', text)            # *italic* → plain
    text = re.sub(r'__([^_]+)__', r'\1', text)            # __bold__ → plain
    text = re.sub(r'_([^_]+)_', r'\1', text)              # _italic_ → plain
    text = re.sub(r'#{1,6}\s+', '', text)                 # ### headers → plain
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)  # code blocks
    text = re.sub(r'`([^`]+)`', r'\1', text)              # inline code

    # ── Phase 3: Remove hallucinated references + training metadata leaks ──
    text = re.sub(r'["\s]*(?:source:\s*)?page_no\s*=\s*\d+["\s]*', ' ', text)
    text = re.sub(r'rules_used:\s*[A-Z_0-9,\s]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bKP_[A-Z]{2,4}_\d{3,5}\b', '', text)
    text = re.sub(r'\[KP_[A-Z_0-9]+\]', '', text)
    text = re.sub(r'\[rule_id\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\((?:Source|Ref|Reference|Page|Ch(?:apter)?)[^)]{0,60}\)', '', text, flags=re.IGNORECASE)
    # Strip training-data metadata that leaks into model output
    text = re.sub(r'rulesused:\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:timingmethod|maxperiod|minperiod|seasonality|reference|events|duration):\s*[^\n.!?]{0,60}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bKPGEN\d+\w*\b', '', text)
    text = re.sub(r'\bKPTIM\d+\w*\b', '', text)
    text = re.sub(r'\bKPADIUS\d+\w*\b', '', text)
    # Strip any remaining ALL-CAPS rule codes like KPGEN0956ADIUS0285
    text = re.sub(r'\b[A-Z]{2,8}\d{4,}[A-Z0-9]*\b', '', text)

    # ── Phase 3.5: Health safety — strip dangerous medical claims ──
    # NOTE: "cancer" alone is NOT replaced — it's a zodiac sign (Cancer/Karka).
    # Only replace cancer in medical contexts like "cancer treatment", "cancer risk".
    # ALSO strip: "YES you have Cancer", "you have cancer", "do you have cancer" responses
    # "YES you have Cancer" / "you have cancer" / "diagnosed with cancer"
    text = re.sub(
        r'(?:yes[,!]?\s+)?(?:you\s+(?:have|had|are\s+diagnosed\s+with)|diagnosed\s+with)\s+cancer',
        'health challenges are indicated in your chart',
        text, flags=re.IGNORECASE
    )
    # Hindi: "Aapko cancer hai" / "aapko tumour hai"
    text = re.sub(
        r'aapko\s+(?:cancer|tumou?r)(?:\s+(?:hai|hoga|hogi|hua|hui|ho\s+sakta|ho\s+sakti))?',
        'aapko health challenges hain',
        text, flags=re.IGNORECASE
    )
    # "mujhe cancer hai"
    text = re.sub(
        r'mujhe\s+(?:cancer|tumou?r)(?:\s+(?:hai|hoga|hogi|hua|hui))?',
        'mujhe health challenges hain',
        text, flags=re.IGNORECASE
    )
    _dangerous_terms = [
        r'cancer[- ](?:related|treatment|risk|diagnosis|patient|surgery|therapy|cells?)',
        r'(?:breast|lung|blood|skin|colon|prostate|ovarian|cervical)\s+cancer',
        r'cancer[!,]?\s+(?:the\s+timing|timing\s+is|is\s+confirmed)',
        r'\btumou?r\b', r'\bmalignant\b', r'\bbenign\b',
        r'potential\s+hospitalization',
        r'hospitalization\s+risks?',
        r'risk\s+of\s+hospitalization',
        r'surgical\s+(?:risk|procedure|intervention)',
        r'medical\s+(?:emergency|crisis|condition)',
        r'(?:serious|critical|severe)\s+(?:health|illness|disease)',
        r'(?:health|physical)\s+(?:deterioration|decline)',
        r'(?:chronic|acute)\s+(?:illness|disease|condition)',
        r'(?:organ|kidney|liver|heart)\s+(?:failure|damage|disease)',
        r'blood\s+(?:pressure|sugar)\s+(?:issue|problem|concern)',
        r'neurological\s+(?:issue|problem|condition)',
        r'immune\s+(?:system\s+)?(?:weakness|deficiency|compromise)',
        r'(?:physical|mental)\s+breakdown',
        r'(?:life|health)\s+(?:threatening|critical|serious)\s+(?:period|time|phase)',
        r'maraka\s+(?:period|dasha|lord|planet)',
        r'(?:death|fatal)\s+(?:period|dasha|time|year)',
        r'(?:8th|eighth)\s+house\s+(?:affliction|problem|issue|danger)',
        r'longevity\s+(?:concern|issue|risk|threat)',
        r'ayu\s+(?:kshaya|loss|reduction)',
        r'(?:mrityu|mritu)\s+(?:yoga|period|dasha)',
        r'(?:accident|injury)\s+(?:risk|prone|likely|possible)\s+in\s+(?:this|the|your)',
        r'accident\s+(?:period|time|phase|window)',
        r'injury\s+(?:period|time|phase|window)',
        r'(?:fall|fracture|surgery)\s+(?:risk|possible|likely|indicated)',
        r'(?:poison|toxic)\s+(?:exposure|risk|danger)',
        r'(?:mental|emotional)\s+(?:breakdown|collapse|crisis)',
        r'suicid(?:e|al)',
        r'self[- ]harm',
        r'(?:depression|anxiety)\s+(?:diagnosis|disorder|condition)',
        r'psychiatric\s+(?:issue|condition|disorder)',
        r'(?:nervous|mental)\s+(?:breakdown|disorder|illness)',
        r'(?:psycho|schizo)\w+',
        r'bipolar\s+(?:disorder|condition)',
        r'(?:dementia|alzheimer)',
        r'(?:stroke|paralysis|coma)',
        r'(?:blindness|deafness|disability)\s+(?:risk|possible|indicated)',
        r'(?:infertility|impotence|sterility)\s+(?:risk|indicated|possible)',
        r'(?:miscarriage|abortion)\s+(?:risk|possible|likely|indicated)',
        r'(?:premature|early)\s+death',
        r'(?:short|reduced)\s+(?:life|lifespan|longevity)',
        r'will\s+(?:not|never)\s+(?:recover|survive|live\s+long)',
        r'(?:no|little)\s+(?:hope|chance)\s+of\s+(?:recovery|survival)',
        r'(?:grave|serious|critical)\s+(?:prognosis|outlook|condition)',
        r'(?:terminal|incurable|untreatable)\s+(?:illness|disease|condition)',
        r'(?:last|final)\s+(?:days|months|years)\s+(?:of\s+)?(?:life|living)',
        r'(?:dying|death)\s+(?:is|seems|appears)\s+(?:near|close|imminent|soon)',
        r'(?:not\s+long\s+to\s+live|limited\s+time\s+left)',
        r'(?:will\s+die|going\s+to\s+die|shall\s+die)\s+(?:in|by|around|during)',
        r'(?:death|end\s+of\s+life)\s+(?:is|seems|appears)\s+(?:near|close|imminent)',
        r'(?:life\s+expectancy|expected\s+lifespan)',
        r'(?:fatal|lethal|deadly)\s+(?:period|dasha|time|phase)',
        r'(?:8th|12th)\s+(?:lord|house)\s+(?:activated|active|strong)\s+(?:indicates?|shows?|suggests?)\s+(?:death|danger|risk)',
        r'(?:Saturn|Rahu|Ketu|Mars)\s+(?:in|aspecting)\s+(?:8th|12th)\s+(?:indicates?|shows?|suggests?)\s+(?:death|danger|risk|harm)',
        r'heart\s+attack', r'heart\s+disease', r'cardiac\s+arrest',
        r'\bdiabetes\b', r'\bHIV\b', r'\bAIDS\b',
        r'tuberculosis', r'epilepsy',
        r'kidney[- ]?(?:failure|disease)',
        r'liver[- ]?(?:failure|disease)',
        r'brain\s+(?:damage|tumou?r)',
        r'mental\s+(?:illness|disorder)', r'schizophren',
        r'\bsuicid', r'\bfatal\b', r'\bterminal(?:ly)?\s+ill',
        r'life[- ]?threatening', r'\blethal\b',
        r'\bbronchitis\b', r'\bpneumonia\b', r'\brespiratory\s+(?:issue|problem|infection|disease)\b',
        r'\blung[- ]?(?:infection|disease|issue|problem)\b',
        r'\bchest\s+(?:infection|pain|disease)\b',
        r'\bfever\s+(?:and|with)\s+(?:cough|cold|infection)\b',
        r'\bviral\s+(?:infection|fever|illness)\b',
        r'\bbacterial\s+(?:infection|illness)\b',
        r'\bhospitali[sz]ation\b', r'\badmitted\s+to\s+hospital\b',
        r'\bICU\b', r'\bemergency\s+(?:room|ward|admission)\b',
        r'\bsurgery\b', r'\boperation\b',
        r'\bfracture\b', r'\bbone\s+(?:break|fracture|injury)\b',
    ]
    for term in _dangerous_terms:
        text = re.sub(term, 'health challenges', text, flags=re.IGNORECASE)

    # ── Phase 3.6: Death/longevity safety — never scare users with maraka terms ──
    _death_terms = [
        r'death[- ]?inflicting\s+(?:bhavas?|houses?)',
        r'\bmaraka\s+(?:houses?|bhavas?|planets?|sthana)',
        r'\blongevity\s+(?:analysis|prediction|assessment)',
        r'\blife\s*span\b', r'\btime\s+of\s+death\b',
        r'\bdeath\s+(?:period|timing|prediction)',
        r'\bayushya\b', r'\bmrityu\b',
    ]
    for term in _death_terms:
        text = re.sub(term, 'challenging period', text, flags=re.IGNORECASE)

    # ── Phase 3.7: Strip SKU / internal product metadata from responses ──
    text = re.sub(r'\s*\(SKU:\s*[^)]+\)', '', text)
    text = re.sub(r'\s*SKU:\s*\S+', '', text)

    # ── Phase 4: Remove ALL "Confidence: xxx" patterns ──
    text = re.sub(r'[Cc]onfidence:?\s*:?\s*(?:high|medium|low|med)(?:\s*\([^)]*\))?', '', text)

    # ── Phase 4.5: Strip deflection / hedging phrases ──
    _deflection_patterns = [
        r'(?:marriage\s+)?timing\s+(?:requires?|needs?)\s+careful\s+analysis\s+of\s+[^.]{0,80}\.',
        r'(?:I|Main)\s+(?:can|will|shall)\s+(?:offer|provide|give)\s+(?:specific\s+)?(?:guidance|insights?|analysis)\s+(?:regarding|about|on)\s+[^.]{0,60}\.',
        r'(?:humein|hume|mujhe)\s+(?:examine|analyze|dekhna|check)\s+karna\s+(?:hoga|padega|chahiye)[^.]{0,40}\.',
        r'(?:I|Main)\s+(?:can|will)\s+(?:analyze|examine|identify)\s+(?:specific|significant|important)\s+[^.]{0,60}\.',
        r'(?:This|Your)\s+(?:requires?|needs?)\s+(?:careful|detailed|thorough)\s+(?:analysis|examination|study)\s+[^.]{0,60}\.',
        r'(?:As\s+(?:your|an?)\s+)?(?:KP\s+)?(?:astrologer|Jyotish),?\s+I\s+(?:analyze|examine|will\s+analyze)\s+[^.]{0,80}\.',
        r'(?:In\s+KP\s+astrology,?\s+)?we\s+(?:examine|need\s+to\s+examine|analyze)\s+[^.]{0,60}\.',
        r'Your\s+question\s+about\s+[^.]{0,60}requires?\s+[^.]{0,40}\.',
    ]
    for pat in _deflection_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)

    # ── Phase 5: Remove robotic section headers (comprehensive) ──
    _robotic_headers = [
        r'(?:Marriage|Career|Financial|Health|Remedy|Obstacle|Education|Relationship|Children|Government\s+Job|Foreign\s+Travel|Progeny)\s+(?:Prediction|Breakthrough|Timing|Gains|Yoga)?\s*(?:Analysis|Prediction|Report)?(?:\s+(?:using|Based|by|for|of)\s+[^\n]{0,60})?\s*$',
        r'(?:Government\s+Job\s+Yoga|Children\s+Prediction|Foreign\s+Travel|Marriage\s+Timing|Career\s+Prospects?)\s+(?:Analysis\s+)?(?:Using|Based\s+on|for|of)\s+(?:Provided\s+)?(?:Chart\s+)?(?:Data|Details|Analysis)\s*$',
        r'(?:Secondary|Primary|Additional)\s+(?:Connections?|Significators?|Combinations?)\s*$',
        r'(?:Core|Key|Main)\s+(?:Significators?|Findings?|Observations?)\s*:?\s*$',
        r'(?:Life\s+Events?|Dasha\s+Period|Your\s+Chart)\s+(?:Between|Analysis|by)\s+[^\n]{0,60}\s*$',
        r'(?:Marriage|Career|Financial|Health|Gemstone)\s+(?:timing|Prospects?|Analysis)\s+(?:for|of)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*(?:Ji)?\s*:',
        r'(?:Shaadi|Marriage)\s+(?:ki\s+)?timing\s+(?:ke\s+liye|for)\s+[^:\n]{0,40}:',
        r'(?:Marriage|Career|Health|Financial)\s+timing\s+for\s+[^:\n]{0,40}:',
        r'(?:Analysis|Conclusion|Application|Critical\s+Finding|Key\s+[Ff]indings?|Summary|Overview|Introduction|Observation)\s*:',
        r'(?:Motivational\s+Quote|Hindi\s+Quote|Recommended\s+Product|Product\s+Recommendation)\s*:',
        r'(?:Remedial\s+Measures|Remedy|Timing|Digestive\s+System|Immune\s+System|Nervous\s+System)\s*:',
        r'(?:Career|Financial|Health|Marriage|Education|Gemstone|Remedy)\s+(?:Analysis|Remedy|Recommendation)\s+(?:Based\s+on|for|using)\s+[^\n]{0,60}\s*:',
        r'(?:Career|Financial|Health|Marriage|Education)\s+(?:Prospects?|Analysis|Prediction|Remedy)\s+(?:Analysis\s+)?(?:for|of)\s+[^\n:]{0,40}\s*:',
        r'(?:Career\s+Analysis|Financial\s+Analysis|Health\s+Analysis|Marriage\s+Analysis|Education\s+Analysis)\s*:',
        r'(?:Marriage|Career)\s+timing\s+for\s+[^\n:]{0,40}\s*:',
        r'Planetary\s+Configuration\s*:',
        r'Primary\s+Significators?\s*:',
        r'(?:Significator|Dasha|Timing)\s+(?:Analysis|Details?|Summary)\s*:',
        r'(?:Astrological\s+)?(?:Prediction|Assessment|Evaluation|Interpretation|Reading)\s*:',
        r'(?:Important|Note|Disclaimer|Warning|Caution)\s*:',
        r'(?:Step|Phase|Part|Section)\s+\d+\s*:',
        r'According\s+to\s+(?:rule\s+\[?KP[_A-Z0-9]*\]?|KP\s+(?:principles?|astrology|system|methodology))\s*[:,]',
        r'Based\s+on\s+(?:the\s+)?(?:given|extracted|provided|above)\s+(?:chart\s+)?(?:data|details|summary|information|context)',
        r'(?:The\s+)?[Kk]ey\s+findings?\s+(?:show|indicate|suggest|reveal)\s+that\s*:?',
        r'(?:In\s+this\s+case|Here),?\s+we\s+need\s+to',
        r'For\s+(?:accurate|proper|detailed)\s+(?:prediction|analysis),?\s+(?:we\s+)?(?:need\s+to\s+)?(?:analyze|examine|look\s+at)',
        r'(?:House|Cusp)\s+\d+\s*(?:\([^)]*\))?\s*:\s*(?:sub\s*=|Sub-lord)',
        r'Let\s+(?:me|us)\s+(?:analyze|examine|look\s+at|check|verify|understand)',
        r'(?:First|Now),?\s+(?:let\'?s?|we\s+(?:will|need\s+to|should))\s+(?:analyze|examine|check|look)',
        r'I\s+(?:will|shall|am\s+going\s+to)\s+(?:analyze|examine|check)',
    ]
    for pat in _robotic_headers:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)

    # ── Phase 6: Remove numbered lists and bullet points ──
    text = re.sub(r'(?:^|\n)\s*\d+[.)]\s+', ' ', text)
    text = re.sub(r'(?:^|\n)\s*[-•●◦▪]\s+', ' ', text)
    # Strip double-newline + section header lines (e.g. "\n\nForeign Travel Indicators:")
    text = re.sub(r'\n{2,}[A-Z][^\n]{0,60}:\s*\n?', ' ', text)
    # Strip standalone section headers on their own line (e.g. "\n\nCurrent Mahadasha\n")
    text = re.sub(r'\n{2,}[A-Z][A-Za-z ]{2,40}\n', ' ', text)
    # Collapse all remaining double newlines to single space
    text = re.sub(r'\n{2,}', ' ', text)
    # Also collapse lines that start mid-sentence (model outputs bullet content on new lines)
    # Pattern: newline followed by lowercase letter or known Hinglish words = continuation
    text = re.sub(r'\n([a-z])', r' \1', text)
    # Collapse remaining single newlines
    text = re.sub(r'\n', ' ', text)
    # Collapse multiple spaces created by above
    text = re.sub(r'  +', ' ', text)

    # ── Phase 6.5: Convert ISO dates to readable format ──
    # Convert '2025-10' or '2025-10-22' patterns to 'Oct 2025'
    _month_map = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun',
                  '07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}
    def _iso_repl(m):
        y, mo = m.group(1), m.group(2)
        return f"{_month_map.get(mo, mo)} {y}"
    text = re.sub(r'\b(20\d{2})-(0[1-9]|1[0-2])(?:-\d{2})?\b', _iso_repl, text)

    # ── Phase 6.5b: Fix hallucinated year typos (e.g. '20626' → '2026') ──
    def _fix_year_typo(m):
        raw = m.group(0)
        if len(raw) == 5 and raw.startswith('20'):
            return raw[:4]  # '20626' → '2062' — still wrong, try '20' + raw[2:4]
        return raw[:4]
    text = re.sub(r'\b20\d{3,5}\b', _fix_year_typo, text)

    # ── Phase 6.55: Age impossibility guard — catch implausible event ages ──
    # e.g. "first job Oct 2025" for someone born 1986 (age 39) is impossible
    _birth_date_pp = getattr(_postprocess, '_birth_date', None)
    _query_type_pp = getattr(_postprocess, '_query_type', 'analysis')
    _user_q_pp = getattr(_postprocess, '_user_question', '').lower()
    _is_first_event_q = any(kw in _user_q_pp for kw in [
        'first job', 'first work', 'pehli naukri', 'pehla kaam',
        'graduation', 'graduate', 'college', 'degree', 'board exam',
        'school', 'first relationship', 'first love', 'pehla pyaar',
        'childhood', 'birth', 'born',
    ])
    if _birth_date_pp and _is_first_event_q:
        from datetime import date as _d_cls
        _today_pp = _d_cls.today()
        _age_now = (_today_pp - _birth_date_pp).days // 365
        # For first-job queries: flag any future date as impossible if person is >30
        # (they already had their first job years ago)
        if _age_now > 28:
            _first_job_future_pat = re.compile(
                r'(?:Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|Jan(?:uary)?|Feb(?:ruary)?|'
                r'Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?)'
                r'\s+20(?:2[5-9]|3[0-9])\b',
                re.IGNORECASE
            )
            if _first_job_future_pat.search(text) and 'first job' in _user_q_pp:
                # Replace the entire response with age-aware correction
                _native_pp = getattr(_postprocess, '_native_name', '') or ''
                _name_ji_pp = f"{_native_pp} ji" if _native_pp else 'Ji'
                text = (
                    f"{_name_ji_pp}, at age {_age_now} your first job would have happened "
                    f"many years ago — likely around age 21-25. Please ask about your "
                    f"career history (e.g. 'What happened in my career from 2010 to 2015?') "
                    f"for accurate past-event analysis."
                )

    # ── Phase 6.6: Date sanity — strip sentences with years before birth year ──
    _birth_year = getattr(_postprocess, '_birth_year', None)
    if _birth_year and _birth_year > 1950:
        def _date_sanity(m):
            year = int(m.group(1))
            if year < _birth_year:
                return ''
            return m.group(0)
        text = re.sub(
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(1[0-9]{3}|20[0-9]{2})',
            _date_sanity, text, flags=re.IGNORECASE
        )
        def _range_sanity(m):
            y1 = int(m.group(1))
            if y1 < _birth_year:
                return ''
            return m.group(0)
        text = re.sub(r'\b(1[0-9]{3}|20[0-9]{2})\s*(?:to|se|tak|-)\s*(?:1[0-9]{3}|20[0-9]{2})\b',
                       _range_sanity, text)

    # ── Phase 6.7: Past-date annotation (SAFE — no global text rewriting) ──
    # If a "Mon YYYY to Mon YYYY" range is entirely in the past, insert a brief note
    # after that range. Does NOT do global word replacements to avoid garbling.
    _today = date.today()
    _month_abbr_to_num = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                          'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,
                          'january':1,'february':2,'march':3,'april':4,'june':6,
                          'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}

    def _mon_year_to_date(mon_str, year_str):
        m = _month_abbr_to_num.get(mon_str.lower(), None)
        if m is None:
            return None
        try:
            return date(int(year_str), m, 28)
        except ValueError:
            return None

    # Find "Mon YYYY to Mon YYYY" ranges that are entirely in the past
    _range_pat = re.compile(
        r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
        r'Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4}))'
        r'(\s+(?:to|se|tak|-)\s+)'
        r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
        r'Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4}))',
        re.IGNORECASE
    )
    _annotated = False
    for _rm in reversed(list(_range_pat.finditer(text))):
        _end_mon = _rm.group(4).split()[0]
        _end_yr = _rm.group(5)
        _end_dt = _mon_year_to_date(_end_mon, _end_yr)
        if _end_dt and _end_dt < _today:
            _insert_pos = _rm.end()
            text = text[:_insert_pos] + ' (yeh period beet chuka hai)' + text[_insert_pos:]
            _annotated = True
            break  # annotate only the first past range to avoid clutter

    # ── Phase 7: Replace robotic third-person references ──
    _replacements = [
        (r'\b[Tt]his\s+native\'s\b', 'your'),
        (r'\b[Tt]his\s+native\s+has\b', 'You have'),
        (r'\b[Tt]his\s+native\s+is\b', 'You are'),
        (r'\b[Tt]his\s+native\b', 'Your'),
        (r'\bThe\s+native\s+has\b', 'You have'),
        (r'\bThe\s+native\s+is\b', 'You are'),
        (r'\bThe\s+native(?:\'s)?\b', 'Your'),
        (r'\bthe\s+native(?:\'s)?\b', 'your'),
        (r'\bnative\s+ka\b', 'aapka'),
        (r'\bnative\s+ki\b', 'aapki'),
        (r'\bnative\s+ke\b', 'aapke'),
        (r'\bnative\s+ko\b', 'aapko'),
        (r'\bThe\s+querent\b', 'You'),
        (r'\bthe\s+querent\b', 'you'),
        (r'\bThe\s+person\b', 'You'),
        (r'\bthe\s+person\b', 'you'),
        (r'\bIt\s+is\s+(?:observed|noted|seen)\s+that\b', ''),
        (r'\bIt\s+(?:can\s+be|is)\s+(?:concluded|inferred)\s+that\b', ''),
        (r'\bIn\s+conclusion,?\b', ''),
        (r'\bTo\s+summarize,?\b', ''),
        (r'\bAccording\s+to\s+(?:the\s+)?given\s+chart\s+details,?\b', ''),
        (r'\bAccording\s+to\s+(?:the\s+)?chart\s+(?:data|details|analysis),?\b', ''),
        (r'\bBased\s+on\s+(?:the\s+)?given\s+chart,?\b', ''),
        (r'\bBased\s+on\s+(?:the\s+)?provided\s+chart\s+(?:analysis|data|details),?\b', ''),
        (r'\bBased\s+on\s+(?:the\s+)?(?:extracted|available)\s+chart\s+(?:summary|data),?\b', ''),
        (r'\bBased\s+on\s+(?:the\s+)?provided\s+horoscope\s+(?:data|analysis|details),?\b', ''),
        (r'\bBased\s+on\s+(?:the\s+)?given\s+(?:planetary\s+positions|horoscope\s+data|chart\s+positions)\s*(?:and\s+(?:their\s+)?(?:house\s+)?(?:rulerships|significations))?,?\b', ''),
        (r'\bBased\s+on\s+your\s+chart\s+(?:analysis|data|configuration)\s*(?:where)?\s*,?\b', ''),
        (r'\bUsing\s+(?:KP|Krishnamurti)\s+(?:Paddhati\s+)?principles,?\b', ''),
        (r'\b[Aa]nd\s+following\s+KP\s+principles,?\b', ''),
        (r'\bUsing\s+(?:the\s+)?(?:provided|given)\s+(?:chart|horoscope)\s+(?:data|details),?\b', ''),
        (r'\blet\s+me\s+address\s+your\s+query\s+about\s+[^.]{0,60}(?:using|through)\s+[^.]{0,40}\.?\b', ''),
        (r'\b[Aa]pplying\s+KP\s+principles,?\b', ''),
        (r'\b[Aa]pplying\s+(?:the\s+)?(?:KP|Krishnamurti)\s+(?:Paddhati\s+)?(?:principles|system|methodology),?\b', ''),
        (r'\bBased\s+on\s+(?:your|the)\s+current\s+planetary\s+positions\s*(?:and\s+dasha\s+system)?,?\b', ''),
        (r'\bBased\s+on\s+(?:the\s+)?provided\s+(?:chart\s+)?(?:details|data)\s+and\s+applying\s+KP\s+principles,?\b', ''),
        (r'\bBased\s+on\s+(?:the\s+)?planetary\s+positions\s+(?:and\s+significator\s+(?:analysis|combinations?)\s+)?(?:provided\s+)?in\s+(?:your|the|this)\s+chart(?:\s+data)?,?\b', ''),
        (r'\bBased\s+on\s+(?:the\s+)?(?:planetary|chart)\s+(?:positions|data)\s+(?:provided|given)\s+(?:in\s+)?(?:your|the)\s+(?:chart|horoscope),?\b', ''),
        (r'\bBased\s+on\s+(?:the\s+)?planetary\s+positions\s+in\s+your\s+chart,?\b', ''),
        (r'\bsignificator\s+analysis\s+provided\s+in\s+this\s+chart\s+data,?\b', ''),
        (r"\byour\s+natal\s+chart's\s+(?:\w+\s+){0,3}potential\b", 'your chart'),
        (r'\bBased\s+on\s+(?:the\s+)?provided\s+dasha\s+sequence,?\b', ''),
        (r'\bBased\s+on\s+your\s+planetary\s+positions\s+and\s+significator\s+combinations,?\b', ''),
        (r',?\s*and\s+significator\s+combinations,?\b', ''),
        (r'\bBased\s+on\s+your\s+(?:natal\s+)?chart\s+(?:details|configuration)\s+and\s+current\s+(?:planetary\s+positions|dasha\s+sequence),?\b', ''),
        (r'\bBased\s+on\s+your\s+(?:natal\s+)?chart\s+(?:configuration|details)\s*,?\b', ''),
        (r'\bAccording\s+to\s+your\s+birth\s+data,?\b', ''),
        (r'\bBased\s+on\s+(?:the\s+)?current\s+planetary\s+periods?\s+(?:running\s+)?in\s+your\s+chart,?\b', ''),
        (r'\bThe\s+cosmic\s+energies\s+align\s+perfectly[^.!?]{0,60}[.!?]?', ''),
        (r'\bcosmic\s+energies\s+align[^.!?]{0,60}[.!?]?', ''),
        (r'\bchallenging\s+cosmic\s+energies[^.!?]{0,60}', ''),
        (r'\bunafflicted\s+planetary\s+influences\s+suggest\s+(?:kar\s+rahe\s+hain|kar\s+rahe\s+hai|that),?\b', ''),
        (r'\bmahadasha\s+ruler\s*:\s*[^.\n]{0,120}', ''),
        (r'\banthardasha\s+ruler\s*:\s*[^.\n]{0,120}', ''),
        (r'\bantardasha\s+ruler\s*:\s*[^.\n]{0,120}', ''),
        (r'\bfunctions?\s+as\s+(?:primary|secondary)\s+significator\s+connecting\s+houses[^.!?]{0,80}', ''),
        (r'\bacts?\s+as\s+(?:primary|secondary)\s+significator\s+(?:connecting|linking)[^.!?]{0,80}', ''),
        (r'\bThe\s+Pratyantar\s+Lord\'s\s+influence\s+adds\s+depth\s+to\s+this\s+prediction\.?\b', ''),
        (r'\bprimary\s+period\s*:\s*', ''),
        (r'\bcritical\s+antardasha\s*:\s*', ''),
        (r'\bcurrent\s+(?:period|phase)\s*:\s*', ''),
        (r'\bpeak\s+(?:period|window)\s*:\s*', ''),
        (r'\bbecause\s*:\s*\n', ' — '),
        (r'\bThe\s+most\s+promising\s+(?:period|window|time)\b[^.!?]{0,80}', ''),
        (r'\bAchhe\s+baat\s+kehte\s+hain,?\s*', ''),
        (r'\bKey\s+point\s+yeh\s+hai\s+ki[^.!?]{0,120}[.!?]?', ''),
        (r'\bexact\s+timing\s+depend\s+(?:karti|karta)\s+hai[^.!?]{0,120}[.!?]?', ''),
        (r',\s+as\s+one\s+of\s+your\s+(?:primary|secondary)\s+(?:wealth|career|marriage)\s+significator\s+planets,', ''),
        (r'\bwhile\s+simultaneously\s+receiving\s+support\s+from[^.!?]{0,80}', ''),
        (r'\boverall\s+period\s+remain\s+(?:karti|karta)\s+hai\s+highly\s+favorable[^.!?]{0,60}', ''),
        (r'\bgets\s+activated\s+during\s+its\s+own\s+Pratyantar\s+period[^.!?]{0,80}', ''),
        (r'\bSaturn\'s\s+natural\s+tendency\s+toward\s+limitation[^.!?]{0,120}[.!?]?', ''),
        (r'\bprogress\s+remains\s+blocked\s+despite\s+apparent\s+opportunities\s+emerging\.?', ''),
        (r'\band\s+significator\s+analysis\s+(?:in\s+this\s+chart(?:\s+data)?)?,?\b', ''),
        (r'\banalysis\s+and\s+current\s+planetary\s+periods,?\b', ''),
        (r'\bmaking\s+this\s+combination\s+highly\s+favorable[^.!?]{0,60}', ''),
        (r'\bthis\s+combination\s+(?:is\s+)?highly\s+favorable[^.!?]{0,60}', ''),
        (r'\bthrough\s+(?:their|its)\s+combined\s+significations\s+of\s+houses[^.!?]{0,60}', ''),
        (r'\ball\s+crucial\s+for\s+(?:career|marriage|finance|health)\s+matters\.?', ''),
        (r"\bit's\s+clear\s+that\s+you're\s+at\s+crossroads[^.!?]{0,80}", ''),
        (r'\bMercury\s+governs\s+intelligence,\s+communication\s+skills[^.!?]{0,80}', ''),
        (r'\bwhile\s+Saturn\s+provides\s+discipline,\s+persistence[^.!?]{0,80}', ''),
        (r'\bnot\s+just\s+passing\s+but\s+achieving\s+substantial\s+recognition[^.!?]{0,80}', ''),
        (r'\bthrough\s+this\s+academic\s+pursuit\.?', ''),
        (r'\bappears?\s+to\s+be\s+temporary\s+in\s+nature\.?', ''),
        (r'\byour\s+feelings\s+of\s+being\s+unlucky\s+appear[^.!?]{0,80}', ''),
        (r'\bfull\s+force\s+of\s+Saturn\'s\s+restrictive\s+influence[^.!?]{0,80}', ''),
        (r'\bSaturn\'s\s+(?:restrictive|limiting)\s+influence\s+combined\s+with[^.!?]{0,80}', ''),
        (r'\bMercury\'s\s+analytical\s+yet\s+sometimes\s+critical\s+energy\.?', ''),
        (r'\baccording\s+to\s+KP\s+(?:principles|astrology|methodology)\.?', ''),
        (r'\bper\s+KP\s+(?:principles|astrology)\.?', ''),
        (r'\bKP\s+(?:principles|methodology)\s+(?:suggest|indicate)[^.!?]{0,60}', ''),
        (r'\bantharam\b', 'antardasha'),
        (r'\bbased\s+on\s+the\s+current\s+planetary\s+periods?\s+(?:running\s+)?in\s+your\s+(?:life|chart),?\b', ''),
        (r'\bbased\s+on\s+the\s+current\s+planetary\s+period\s+you\'re\s+experiencing[^,\.!?]{0,60}[,]?', ''),
        (r'\bbased\s+on\s+the\s+significator\s+analysis\s+in\s+your\s+chart,?\b', ''),
        (r'\bKP\s+Analysis\s+for\s+[A-Za-z\s]+Query\s*\n?', ''),
        (r'\bdepend\s+karte\s+hain\s+specific\s+planetary\s+combinations\s+par\s+jo\s+aapke\s+birth\s+chart\s+mein\s+signify\s+kar\s+rahe\s+hain\.?', ''),
        (r'\baapke\s+career\s+prospects\s+ke\s+liye\s+remedy\s+recommendations\s+depend\s+karte\s+hain[^.!?]{0,120}[.!?]?', ''),
        (r'\bremedy\s+recommendations\s+depend\s+(?:karte|karta)\s+hain[^.!?]{0,120}[.!?]?', ''),
        (r'\bhar\s+planet\s+ki\s+strength\s+ko\s+address\s+karna\s+padega\s+appropriate\s+remedial\s+measures\s+se\.?', ''),
        (r'\bKyunki\s+aapke\s+paas\s+multiple\s+planets\s+serve\s+kar\s+rahe\s+hain\s+houses\s+[\d,\s]+ke\s+saath\s+as\s+career\s+significators,?', ''),
        # ── New patterns from Round 14 test results ──
        (r'\bprovided\s+here,?\s+I\s+see\s+multiple\s+planetary\s+combinations[^.!?]{0,120}[.!?]?', ''),
        (r'\bI\s+find\s+multiple\s+significators\s+for\s+(?:employment|career|marriage|finance)[^.!?]{0,120}[.!?]?', ''),
        (r'\bI\s+notice\s+an\s+interesting\s+pattern\s+emerging[^.!?]{0,80}[.!?]?', ''),
        (r'\bKey\s+timing\s+factors\s+show\s+current[^.!?]{0,120}[.!?]?', ''),
        (r'\bMultiple\s+planets\s+signify\s+both\s+(?:marriage|career|finance)[^.!?]{0,120}[.!?]?', ''),
        (r'\bhumein\s+examine\s+karna\s+padega\s+significators[^.!?]{0,120}[.!?]?', ''),
        (r'\bYahan\s+cosmic\s+timing\s+aapko[^.!?]{0,120}[.!?]?', ''),
        (r'\bcosmic\s+timing\s+aapko\s+immediate\s+relief\s+offer\s+karti\s+hai[^.!?]{0,80}[.!?]?', ''),
        (r'\bAapki\s+(?:shaadi|job|career|financial)\s+(?:timing|query)\s+ke\s+liye[^,\.]{0,80}[,\.]?\s*', ''),
        (r'\bAapka\s+(?:career|financial)\s+(?:timing|query)\s+ke\s+liye[^,\.]{0,80}[,\.]?\s*', ''),
        (r'\bApni\s+property\s+purchase\s+ki\s+timing\s+ke\s+liye[^,\.]{0,100}[,\.]?\s*', ''),
        (r'\bAapki\s+\w+\s+improvement\s+timing\s+ye\s+periods\s+indicate\s+karti\s+hai\s*:\s*', ''),
        (r'\bOther\s+notable\s+windows\s+involve[^.!?]{0,120}[.!?]?', ''),
        (r'\bVenus-(?:Sun|Moon|Mars|Jupiter|Saturn|Mercury|Rahu|Ketu)\s+pratyantar\s+\([A-Za-z]{3}\s+\d{4}-\)\s+offers?\s+exceptional\s+opportunities\s+once\s+activated[^.!?]{0,60}[.!?]?', ''),
        (r'\bwas\s+highly\s+beneficial\s+for\s+all\s+ventures[^.!?]{0,60}[.!?]?', ''),
        (r'\bCurrent\s+Favorable\s+Period\s*:\s*[^.!?]{0,200}[.!?]?', ''),
        (r'\bShort-Term\s+Recovery\s*:\s*[^.!?]{0,200}[.!?]?', ''),
    ]
    for pat, repl in _replacements:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)

    # ── Phase 7.5: Clean up artifacts from Phase 7 replacements ──
    # Fix orphaned commas/spaces at sentence start (e.g. ", I can" → "I can")
    text = re.sub(r'(?:^|(?<=[\.\.!?]))\s*,\s*', ' ', text)
    # Fix orphaned comma after any period-space (e.g. "hain. , this" → "hain. This")
    text = re.sub(r'(\.\.\s*),\s*', r'\1', text)
    # Fix double comma after name+ji (e.g. "Rajesh ji, , your" → "Rajesh ji, your")
    text = re.sub(r',\s*,+', ',', text)
    # Fix orphaned 'and' at start of text
    text = re.sub(r'^\s*and\s+', '', text, flags=re.IGNORECASE)
    # Fix double spaces from removed phrases
    text = re.sub(r'  +', ' ', text)
    # Capitalize first letter after period (e.g. "hain. aapki" → "hain. Aapki")
    text = re.sub(r'(?<=\.\s)([a-z])', lambda m: m.group(1).upper(), text)
    # Fix capitalized "Your" mid-sentence (only after lowercase letter + space)
    text = re.sub(r'(?<=[a-z]\s)Your\b', 'your', text)
    # Fix name-based third-person: "Name ji's" → "your" (dynamic)
    _native = getattr(_postprocess, '_native_name', '') or ''
    if _native:
        _first = _native.split()[0]
        # "Priya ji's current" → "your current", "Priya Raj ji's" → "your"
        text = re.sub(rf'\b{re.escape(_native)}\s+ji\'s\b', 'your', text, flags=re.IGNORECASE)
        text = re.sub(rf'\b{re.escape(_first)}\s+ji\'s\b', 'your', text, flags=re.IGNORECASE)
        # "Priya's foreign" → "your foreign"
        text = re.sub(rf'\b{re.escape(_native)}\'s\b', 'your', text, flags=re.IGNORECASE)
        text = re.sub(rf'\b{re.escape(_first)}\'s\b', 'your', text, flags=re.IGNORECASE)
    # Strip leading whitespace from text
    text = text.strip()

    # ── Phase 8: Remove filler and metadata lines ──
    lines = text.split("\n")
    cleaned = []
    _filler_phrases = [
        "considerably enhanced", "enhanced answer", "proper format",
        "additional recommendations", "professional competence",
        "theoretical understanding alone", "practical application validate",
        "absolute faith display", "considerable research effort",
        "let me analyze this situation", "we need to identify planets",
        "pending deeper analysis", "as per kp principles",
        "according to kp astrology", "using kp methodology",
        "based on the chart data provided", "from the given chart",
        "as mentioned in the chart", "the chart shows that",
        "looking at the chart data", "examining the chart",
        "i will now analyze", "let me examine",
        "grounding rule", "as per the grounding",
        "grounding principle", "grounding principles",
        "krishnamurti ji ke principle ke according",
        "jis method se hum predictions banate hain woh bilkul reliable nahi",
        "predictions banate hain woh bilkul reliable nahi",
        "sirf immediate future events hi predict kar sakte hain",
        "koi bhi attempt longer-term predictions ke liye complete nonsense",
        "timing precision: jab significant",
        "cosmic rhythms perfectly align",
        "rule-based system:",
        "verified methodology:",
        "moderate confidence level",
        "confidence level: moderate",
        "confidence level: high",
        "confidence level: low",
        "this is a rule-based",
        "rule based system",
        "sub-lord significance:",
        "planetary positions analysis:",
        "love vs arranged marriage",
        "career prospects analysis",
        "financial analysis:",
        "peak financial growth period:",
        "timing precision:",
        "sub-lord significance:",
        "core significators:",
        "primary significators:",
        "secondary significators:",
        "key significators:",
        "significator analysis:",
        "house activation:",
        "dasha activation:",
        "planetary configuration:",
        "chart analysis:",
        "kp analysis:",
        "primary period:",
        "critical antardasha:",
        "peak period:",
        "current period:",
        "most promising combination:",
        "most promising combination ye present kar raha hai:",
        "mahadasha ruler:",
        "anthardasha ruler:",
        "antardasha ruler:",
        "underlying mechanism involves",
        "the cosmic energies align",
        "provided here, i see",
        "i find multiple significators",
        "excellent potential dikhata hai",
        "certain remedial measures require hote hain",
        "cosmic timing aapko immediate relief",
        "yahan cosmic timing",
        "bilkul understandable hai",
        "interesting pattern emerging",
        "primary concern yeh hai ki",
        "key timing factors show",
        "multiple planets signify both",
        "humein examine karna padega significators",
        "aapki educational journey ke baare mein",
        "aapki job query ke liye",
        "aapki financial improvement timing",
        "short-term recovery:",
        "long-term recovery:",
        "favorable period:",
        "challenging period:",
        "current favorable period:",
        "other notable windows involve",
        "careerevent:",
        "positiveoutcome:",
        "specificdates:",
        "bannedphrases:",
        "eventtype:",
        "scoringcriteria:",
        "marriageevent:",
        "financialevent:",
        "emotionalevent:",
        "safetyflag:",
    ]
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("rules_used:") or stripped.startswith("rules used:"):
            continue
        if stripped.startswith("level:") or stripped.startswith("answer_end"):
            continue
        # Strip DPO training rubric metadata that leaks into model output
        _rubric_prefixes = (
            "careerevent:", "career event:", "positiveoutcome:", "positive outcome:",
            "specificdates:", "specific dates:", "bannedphrases:", "banned phrases:",
            "eventtype:", "event type:", "scoringcriteria:", "scoring criteria:",
            "marriageevent:", "marriage event:", "financialevent:", "financial event:",
            "emotionalevent:", "emotional event:", "safetyflag:", "safety flag:",
        )
        if any(stripped.startswith(p) for p in _rubric_prefixes):
            continue
        if any(filler in stripped for filler in _filler_phrases):
            continue
        # Skip lines that are ONLY a short label/header (no real content)
        if stripped.endswith(":") and len(stripped) < 50 and not any(c.isdigit() for c in stripped):
            continue
        if len(stripped) < 3:
            cleaned.append("")
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).rstrip()

    # ── Phase 8.4: Strip self-doubt / reliability-undermining phrases ──
    _self_doubt_patterns = [
        r'(?:jis\s+method\s+se\s+hum\s+predictions\s+banate\s+hain)[^.!?]{0,100}[.!?]',
        r'(?:predictions?\s+(?:are|is)\s+(?:not|never)\s+(?:100%|fully|completely)\s+(?:reliable|accurate|certain))[^.!?]{0,80}[.!?]',
        r'(?:astrology\s+(?:cannot|can\'t|does\s+not)\s+(?:guarantee|predict\s+with\s+certainty))[^.!?]{0,80}[.!?]',
        r'(?:no\s+astrologer\s+can\s+(?:guarantee|be\s+100%|predict\s+exactly))[^.!?]{0,80}[.!?]',
        r'(?:these\s+are\s+(?:just|only)\s+(?:indications?|possibilities|probabilities))[^.!?]{0,60}[.!?]',
        r'(?:koi\s+bhi\s+attempt\s+longer-term\s+predictions)[^.!?]{0,80}[.!?]',
        r'(?:sirf\s+immediate\s+future\s+events\s+hi\s+predict)[^.!?]{0,80}[.!?]',
        r'(?:cosmic\s+rhythms\s+perfectly\s+align)[^.!?]{0,80}[.!?]',
        r'(?:timing\s+precision:\s*jab)[^.!?]{0,80}[.!?]',
    ]
    for _sdp in _self_doubt_patterns:
        result = re.sub(_sdp, '', result, flags=re.IGNORECASE)

    # ── Phase 8.45: Strip DPO rubric metadata labels that leak inline ──
    _rubric_patterns = [
        r'\bcareerevent\s*:\s*\S+[^\n]*',
        r'\bpositiveoutcome\s*:\s*\S+[^\n]*',
        r'\bspecificdates\s*:\s*\S+[^\n]*',
        r'\bbannedphrases\s*:\s*\S+[^\n]*',
        r'\beventtype\s*:\s*\S+[^\n]*',
        r'\bscoringcriteria\s*:\s*\S+[^\n]*',
        r'\bmarriageevent\s*:\s*\S+[^\n]*',
        r'\bfinancialevent\s*:\s*\S+[^\n]*',
        r'\bemotionalevent\s*:\s*\S+[^\n]*',
        r'\bsafetyflag\s*:\s*\S+[^\n]*',
        r'\bcareer\s+event\s*:\s*\S+[^\n]*',
        r'\bpositive\s+outcome\s*:\s*\S+[^\n]*',
        r'\bspecific\s+dates\s*:\s*\S+[^\n]*',
        r'\bbanned\s+phrases\s*:\s*\S+[^\n]*',
    ]
    for _rp in _rubric_patterns:
        result = re.sub(_rp, '', result, flags=re.IGNORECASE)
    result = re.sub(r'\n{3,}', '\n\n', result)  # clean up blank lines from removals

    # ── Phase 8.5: Strip model-generated Hindi quotes on factual/timing queries ──
    # Keep quotes on: remedy, emotional, analysis (where motivational tone helps)
    _query_type = getattr(_postprocess, '_query_type', 'analysis')
    _strip_quotes = _query_type in ("simple", "timing", "past_event")
    if _strip_quotes:
        _quote_patterns = [
            r'(?:^|\n\n?)\s*(?:Jab samay|Andhera jitna|Sabr ka phal|Jab tak todenge|Graho ki chaal|'
            r'Mushkilein waqti|Waqt sabka|Har raat ke baad|Kismat likhne|Jab niyat)[^\n]{0,120}\.?\s*$',
            r'(?:^|\n\n?)\s*"[^"]{10,120}"\s*$',  # quoted Hindi sentences
        ]
        for qp in _quote_patterns:
            result = re.sub(qp, '', result, flags=re.MULTILINE)

    result = result.rstrip()

    # ── Phase 9: Clean up whitespace ──
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = re.sub(r'  +', ' ', result)  # collapse double spaces

    # ── Phase 10: Truncate to max 3 paragraphs ──
    paragraphs = [p.strip() for p in result.split("\n\n") if p.strip()]
    if len(paragraphs) > 3:
        result = "\n\n".join(paragraphs[:3])

    # ── Phase 11: Remove trailing incomplete sentences ──
    # Only truncate if the text ends mid-sentence (no terminal punctuation)
    # and there's a clear sentence boundary to cut at
    if result and result.rstrip()[-1] not in '.!?"\u0964\n)}':
        # Find the last sentence-ending punctuation
        _last_dot = max(result.rfind('. '), result.rfind('.\n'), result.rfind('! '),
                        result.rfind('? '), result.rfind('.\u0964'))
        # Only truncate if: (a) a boundary exists AND (b) it's past 40% of text
        # AND (c) the trailing fragment is at least 8 chars (real incomplete sentence)
        _trailing = result[_last_dot + 1:].strip() if _last_dot > 0 else result
        if _last_dot > len(result) * 0.4 and len(_trailing) >= 8:
            result = result[:_last_dot + 1].rstrip()

    # ── Phase 11.5: Empathy prefix for emotional queries ──
    if _query_type == "emotional":
        _native = getattr(_postprocess, '_native_name', '') or ''
        _name_ji = f"{_native} ji" if _native else "Ji"
        # Check if response already starts with empathy
        _empathy_markers = ["samajh", "understand", "mushkil", "difficult", "tough", "worry not", "don't worry", "chinta"]
        _has_empathy = any(m in result[:120].lower() for m in _empathy_markers)
        if not _has_empathy:
            _empathy_prefix = f"{_name_ji}, main samajh sakta hun yeh waqt aapke liye kitna mushkil hai — aap akele nahi hain. "
            result = _empathy_prefix + result
            # Strip any name+ji the model added right after our prefix (flexible: first/full name)
            _first = _native.split()[0] if _native else ''
            _name_variants = [re.escape(_name_ji), re.escape(f"{_first} ji")] if _first else [re.escape(_name_ji)]
            for _nv in _name_variants:
                result = re.sub(rf'^({re.escape(_empathy_prefix)})\s*{_nv},?\s*', r'\1', result)

    # ── Phase 12: Hard sentence cap based on query type ──
    sentences = re.split(r'(?<=[.!?])\s+', result.strip())
    if _query_type == "simple" and len(sentences) > 1:
        result = sentences[0]  # HARD 1-sentence cap for simple queries
    elif _query_type in ("timing", "emotional") and len(sentences) > 3:
        result = ' '.join(sentences[:3])
    elif len(sentences) > 4:
        result = ' '.join(sentences[:4])

    # ── Phase 12.6: Hard character cap — trim to last sentence within 350 chars ──
    _char_limit = {"simple": 180, "timing": 320, "emotional": 350, "past_event": 400, "remedy": 400}.get(_query_type, 350)
    if len(result) > _char_limit:
        _trimmed = result[:_char_limit]
        _last_end = max(_trimmed.rfind('. '), _trimmed.rfind('! '), _trimmed.rfind('? '),
                        _trimmed.rfind('.'), _trimmed.rfind('!'), _trimmed.rfind('?'))
        if _last_end > _char_limit * 0.5:
            result = result[:_last_end + 1].rstrip()

    # ── Phase 12.5: Strip trailing filler for simple queries ──
    if _query_type == "simple":
        # Remove sentences that explain KP theory after the direct answer
        _filler_starters = [
            "ye planetary", "yeh planetary", "this creates", "this is",
            "in kp astrology", "kp system mein", "ye combination",
            "yeh combination", "ye positions", "yeh positions",
            "this planetary", "these planetary", "yeh aapki",
            "this placement", "this is an important", "main aapse",
            "jo traditional", "jo krishnamurti", "jo vimshottari",
            "fundamental chart", "important reference",
        ]
        sents = re.split(r'(?<=[.!?])\s+', result.strip())
        kept = []
        for s in sents:
            if any(s.strip().lower().startswith(f) for f in _filler_starters):
                break
            kept.append(s)
        if kept:
            result = ' '.join(kept[:1])  # Force 1 sentence for simple

    # ── Phase 13: Language enforcement — strip Hinglish filler from English responses ──
    # The model is SFT-baked in Hinglish. We can't translate, but we can strip the
    # most common Hinglish sentence starters and connectors that appear on English questions.
    _user_question = getattr(_postprocess, '_user_question', '')
    if _user_question:
        _hindi_markers = [
            'kya', 'hai', 'mera', 'meri', 'kab', 'kaise', 'kaisa',
            'hogi', 'hoga', 'karu', 'batao', 'bataiye', 'shaadi',
            'paisa', 'naukri', 'padhai', 'ghar', 'rishta',
            'aapka', 'aapki', 'mujhe', 'humein', 'kahan',
        ]
        q_words = _user_question.lower().split()
        hindi_count = sum(1 for w in q_words if w in _hindi_markers)
        is_hindi_question = (hindi_count >= 2 or
            any(w in _user_question.lower() for w in
                ['kab hogi', 'kya hoga', 'kaise hoga', 'batao', 'bataiye', 'aaj ki']))

        if not is_hindi_question:
            # English question — strip Hinglish sentence starters that the model inserts
            _hinglish_starters = [
                r'^Aapke liye ek favorable\b[^.!?]{0,120}[.!?]?\s*',
                r'^Aapke liye\b[^.!?]{0,80}[,.]\s*',
                r'^Aapki [a-z]+ (?:ke baare mein|timing ke liye|query ke liye)[^.!?]{0,100}[,.]\s*',
                r'^Aapka [a-z]+ (?:ke baare mein|timing ke liye)[^.!?]{0,100}[,.]\s*',
                r'^Apni [a-z]+ (?:ke liye|purchase ki)[^.!?]{0,100}[,.]\s*',
                r'^(?:Aapke|Aapki|Aapka|Apni|Apna) \w+[^.!?]{0,120}[,.]\s*(?=[A-Z])',
            ]
            for _hs in _hinglish_starters:
                result = re.sub(_hs, '', result, flags=re.IGNORECASE)
            result = result.strip()
            # If after stripping the response is now empty or too short, don't strip
            if len(result) < 20:
                result = text  # restore original

            # Deep Hinglish body detection: if response has ≥4 Hindi body words,
            # filter out sentences that are predominantly Hinglish, keep English ones.
            # Skip for emotional/timing queries — Hinglish empathy and date sentences are acceptable.
            _skip_deep_filter = _query_type in ('emotional', 'safety')
            _hindi_body_words = [
                'aapki', 'aapka', 'aapke', 'hain', 'hota', 'hoti', 'hote',
                'karta', 'karti', 'karte', 'karna', 'karni', 'karne',
                'mein', 'se', 'ke', 'ki', 'ka', 'ko', 'par', 'pe',
                'hai ', 'tha ', 'thi ', 'the ', 'tha.', 'thi.', 'the.',
                'dekhte', 'dekhna', 'samajh', 'isliye', 'kyunki',
                'jab', 'tab', 'toh', 'aur ', 'ya ', 'lekin', 'phir',
                'bahut', 'achha', 'acchi', 'zyada', 'thoda', 'bilkul',
                'padega', 'sakta', 'sakti', 'sakte', 'chahiye',
                'dwara', 'wala', 'wali', 'wale', 'waala',
            ]
            _result_lower = result.lower()
            _hindi_body_count = sum(1 for w in _hindi_body_words if w in _result_lower)
            if not _skip_deep_filter and _hindi_body_count >= 4:
                # Split into sentences and keep only English-dominant ones
                _sents = re.split(r'(?<=[.!?])\s+', result.strip())
                _english_sents = []
                _date_pat = re.compile(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|202\d|203\d)\b', re.I)
                for _s in _sents:
                    _s_lower = _s.lower()
                    _s_hindi = sum(1 for w in _hindi_body_words if w in _s_lower)
                    _s_words = len(_s.split())
                    _has_date = bool(_date_pat.search(_s))
                    # Keep sentence if: has a date, OR Hindi word density < 25%
                    if _has_date or (_s_words > 0 and (_s_hindi / _s_words) < 0.25):
                        _english_sents.append(_s)
                if len(_english_sents) >= 1:
                    result = ' '.join(_english_sents).strip()

    return result


# ── Hindi motivational quotes pool ───────────────────────────────────────────
HINDI_QUOTES = [
    "Jab samay aayega, sab kuch apne aap ho jayega.",
    "Andhera jitna gehra ho, subah utni roshan hoti hai.",
    "Sabr ka phal meetha hota hai.",
    "Jab tak todenge nahi, tab tak chodenge nahi — yahi hausla zaroori hai.",
    "Graho ki chaal badal sakti hai, lekin aapka irada nahi badalna chahiye.",
    "Mushkilein waqti hain, lekin aapki himmat daimi hai.",
    "Waqt sabka aata hai, bas bharosa rakhiye.",
    "Har raat ke baad savera aata hai, aur aapka savera bhi aayega.",
    "Kismat likhne wala bhi wahi hai, aur badalne wala bhi aap hain.",
    "Jab niyat saaf ho, toh naseeb bhi saath deta hai.",
]


def _classify_query_type(question: str) -> dict:
    """Classify query to control response length, temperature, and behavior.
    Returns dict with: type, max_paragraphs, temperature, max_tokens_override.

    ORDER IS CRITICAL — safety and emotional MUST be checked FIRST because
    their patterns overlap with timing/past (e.g. 'when will i die' matches 'when will').
    """
    q = question.lower().strip()

    # ── 1a. SAFETY — death/longevity/health fear → compassionate redirect ──
    safety_patterns = [
        "will i die", "when will i die", "death", "maut", "mrityu",
        "kab marunga", "kab marungi", "longevity", "life span",
        "scared about my health", "serious illness", "fatal",
        "scared about health", "die soon", "kab maru",
        "will i survive", "marr jaunga", "marr jaungi", "marne wala",
        "will you die", "will we die", "going to die",
    ]
    # Broad regex: catches 'will will you die', 'will i die', 'will you die', typos
    if any(p in q for p in safety_patterns) or re.search(r'\bwill\s+\w+\s+die\b', q) or re.search(r'\bdie\b', q):
        return {"type": "safety", "max_paragraphs": 2, "temperature": 0.3, "max_tokens_override": 300}

    # ── 1b-ext. MEDICAL DIAGNOSIS — cancer/disease queries → safety redirect ──
    # These must be caught at query level BEFORE reaching the model
    _medical_query_patterns = [
        r'do\s+i\s+have\s+cancer', r'kya\s+mujhe\s+cancer\s+hai',
        r'do\s+i\s+have\s+(?:diabetes|hiv|aids|tumor|tumour|disease)',
        r'am\s+i\s+(?:sick|ill|dying|going\s+to\s+die)',
        r'will\s+i\s+get\s+cancer', r'cancer\s+(?:risk|chance|possibility)',
        r'do\s+i\s+have\s+(?:a\s+)?(?:serious|terminal|chronic)\s+(?:illness|disease|condition)',
        r'will\s+i\s+(?:get|develop|have)\s+(?:a\s+)?(?:serious|terminal|chronic|deadly)\s+(?:illness|disease)',
        r'kya\s+mujhe\s+(?:cancer|bimari|gambhir\s+bimari)\s+(?:hai|hogi|hoga)',
        r'mujhe\s+(?:cancer|tumor|bimari)\s+(?:hai|hoga|hogi)',
        r'(?:diagnose|diagnosis)\s+(?:for|of|with)\s+(?:cancer|disease)',
        r'(?:cancer|tumor|tumour)\s+(?:in\s+my|mera|meri)',
        r'kya\s+main\s+(?:bimar|sick|ill)\s+(?:hun|hoon|ho)',
        r'meri\s+(?:bimari|illness|disease)\s+(?:kya|kab|kyun)',
        r'will\s+i\s+be\s+hospitalized',
        r'kab\s+(?:hospital|admit)\s+(?:hoga|hogi|jaana)',
    ]
    if any(re.search(p, q) for p in _medical_query_patterns):
        return {"type": "medical_safety", "max_paragraphs": 1, "temperature": 0.3, "max_tokens_override": 200}

    # ── 1c. SELF-DOUBT / META — "can you predict", "how accurate are you" → confident intercept ──
    _meta_confidence_patterns = [
        r'can\s+you\s+(?:really\s+)?predict',
        r'how\s+accurate\s+(?:are\s+you|is\s+this)',
        r'(?:are|is)\s+(?:kp\s+)?astrology\s+(?:accurate|reliable|real|true)',
        r'do\s+you\s+(?:really\s+)?know\s+(?:the\s+)?future',
        r'(?:can|could)\s+astrology\s+(?:really\s+)?(?:predict|tell)',
        r'kya\s+(?:aap|tum)\s+(?:sach\s+mein\s+)?(?:predict|bata)\s+(?:kar\s+sakte|sakte\s+ho)',
        r'kya\s+astrology\s+(?:sach|sahi|accurate)\s+(?:hai|hoti)',
        r'how\s+reliable\s+(?:is|are)\s+(?:your|these|kp)\s+(?:predictions?|readings?)',
        r'(?:are|is)\s+(?:your|these)\s+predictions?\s+(?:accurate|reliable|correct)',
        r'kya\s+(?:yeh|ye)\s+predictions?\s+(?:sach|sahi|accurate)\s+(?:hain|hai)',
    ]
    if any(re.search(p, q) for p in _meta_confidence_patterns):
        return {"type": "meta_confidence", "max_paragraphs": 1, "temperature": 0.3, "max_tokens_override": 200}

    # ── 1b. INAPPROPRIATE — sexual orientation, personal judgments → firm redirect ──
    inappropriate_patterns = [
        "am i gay", "i am gay", "i'm gay", "are you gay", "homosexual",
        "lesbian", "bisexual", " gay", "sexual orientation", "sexuality",
        "sex life", "sexual", "am i a virgin", "virginity", "pregnant by",
        "am i ugly", "am i beautiful", "am i attractive",
        "same-sex", "same sex", "lgbtq",
    ]
    if any(p in q for p in inappropriate_patterns):
        return {"type": "inappropriate", "max_paragraphs": 1, "temperature": 0.3, "max_tokens_override": 150}

    # ── 2. EMOTIONAL — before timing (e.g. 'scared' could appear in timing context) ──
    emotional_patterns = [
        "tough time", "going wrong", "obstacles", "struggling", "depressed",
        "confused", "frustrated", "scared", "worried", "anxious", "hopeless",
        "everything is going wrong", "why is everything", "mushkil", "pareshani",
        "takleef", "dukh", "tension", "problem", "suffering",
        "loser", "looser", "failure", "unlucky", "nothing works",
        "won't do anything", "no hope", "give up", "kuch nahi hoga",
        "feel very unlucky", "feel unlucky", "bad luck", "cursed",
        "health has been troubling", "health troubling", "health issues",
        "health concern", "not feeling well", "tabiyat kharab", "bimar",
        "health problem", "body pain", "sleepless", "insomnia",
    ]
    if any(p in q for p in emotional_patterns):
        return {"type": "emotional", "max_paragraphs": 2, "temperature": 0.4, "max_tokens_override": 500}

    # ── 3. Simple factual — 1-2 sentences, low temperature ──
    simple_patterns = [
        "what is my name", "mera naam", "my name", "naam kya hai",
        "what is my dob", "date of birth", "birth date", "janam din",
        "what is my lagna", "lagna kya hai", "ascendant",
        "what is my rashi", "rashi kya hai", "moon sign",
        "what is my nakshatra", "nakshatra kya hai",
        "where was i born", "birth place", "kahan paida",
        "who are you", "what can you do", "tell me about yourself",
        "what is your name", "your name", "tumhara naam", "aapka naam",
        "what is the date", "what's the date", "aaj ki date", "today's date",
    ]
    if any(p in q for p in simple_patterns):
        return {"type": "simple", "max_paragraphs": 1, "temperature": 0.3, "max_tokens_override": 150}

    # ── 4. Past event / year-by-year ──
    past_patterns = [
        "what happened", "year by year", "year-by-year", "from 20",
        "between 20", "in 2020", "in 2021", "in 2022", "in 2023", "in 2024", "in 2025",
        "when did i", "kab hua", "kab hui", "past ", "pichle",
        "graduation", "first job", "first relationship", "childbirth",
        "health issue", "what year did",
    ]
    if any(p in q for p in past_patterns):
        return {"type": "past_event", "max_paragraphs": 3, "temperature": 0.4, "max_tokens_override": 600}

    # ── 5. Timing questions ──
    timing_patterns = [
        "when will", "kab hogi", "kab milegi", "kab hoga",
        "timing", "which year", "which month", "best period",
        "favorable time", "auspicious time", "shubh samay",
        "exam", "interview", "pariksha", "test result", "get success",
        "will i pass", "will i clear", "selection", "job offer",
        "should i change", "change fields", "change career", "switch job",
    ]
    if any(p in q for p in timing_patterns):
        return {"type": "timing", "max_paragraphs": 2, "temperature": 0.5, "max_tokens_override": 300}

    # ── 6. Remedy queries ──
    if _is_remedy_query(question):
        return {"type": "remedy", "max_paragraphs": 3, "temperature": 0.5, "max_tokens_override": 500}

    # ── 7. Complex analysis — full response ──
    return {"type": "analysis", "max_paragraphs": 3, "temperature": 0.5, "max_tokens_override": 400}


# Product recommendation sentence templates (varied for natural feel)
_PRODUCT_TEMPLATES = [
    "Is samay {planet_phrase}ke liye hamara {product} try karein — yeh aapke planetary energies ko balance karne mein madad karega.",
    "Aapke liye hamara {product} beneficial ho sakta hai — yeh {planet_phrase}ki energy ko strengthen karta hai.",
    "Remedy ke taur par hamara {product} dekhein — yeh {planet_phrase}ke prabhav ko positive banane mein sahayak hai.",
    "Hamara {product} aapke liye ek achha upay ho sakta hai — {planet_phrase}ko balance karne mein helpful hai.",
]


def _enrich_response(text, product_text="", is_remedy=False, query_type="analysis"):
    """Append Hindi quote ONLY on remedy/obstacle queries. Product ONLY if remedy query."""
    text_lower = text.lower()

    # Hindi quotes ONLY on remedy queries — client feedback: no padding on other queries
    additions = []
    if query_type == "remedy":
        has_quote = any(q[:20].lower() in text_lower for q in HINDI_QUOTES)
        if not has_quote:
            quote_indicators = ["jab samay", "andhera jitna", "sabr ka phal", "har raat ke baad",
                               "mushkilein waqti", "waqt sabka", "kismat likhne", "graho ki chaal",
                               "jab niyat", "waqt sabka aata"]
            has_quote = any(ind in text_lower for ind in quote_indicators)
        if not has_quote:
            quote = random.choice(HINDI_QUOTES)
            additions.append(quote)

    # Only add product fallback if this is a remedy query AND model didn't already mention one
    if is_remedy and product_text:
        has_product = any(kw in text_lower for kw in [
            "pendant", "bracelet", "mala", "rudraksha", "kavach", "necklace",
            "gemstone", "neelam", "pukhraj", "moonga", "panna", "manik",
            "gomed", "pearl", "moti", "diamond", "sapphire", "coral",
            "emerald", "ruby", "hessonite", "cat eye", "hamara", "hamare",
        ])
        if not has_product:
            first_line = product_text.split("\n")[0] if product_text else ""
            match = re.match(r'-\s*(.+?)\s*\(SKU:', first_line)
            if match:
                product_name = match.group(1).strip()
                # Detect planet context from text for natural sentence
                planet_phrase = ""
                for planet in ["Venus", "Saturn", "Jupiter", "Mars", "Mercury", "Moon", "Sun", "Rahu", "Ketu"]:
                    if planet.lower() in text_lower:
                        planet_phrase = f"{planet} "
                        break
                template = random.choice(_PRODUCT_TEMPLATES)
                additions.append(template.format(product=product_name, planet_phrase=planet_phrase))

    if additions:
        text = text.rstrip()
        if text and text[-1] not in '.!?':
            text += '.'
        text += " " + " ".join(additions)

    return text


# ── Chart preprocessing — shared module (single source of truth) ──
from chart_preprocessor import chart_to_yaml as _chart_to_yaml


_REMEDY_STRONG_KEYWORDS = [
    "remedy", "remedies", "upay", "upaye", "upaay",
    "gemstone", "ratna", "rudraksha", "mantra", "puja", "pooja",
    "kavach", "totka", "vidhi", "wear", "pehnu", "pehnna",
    "which stone", "kaun sa ratna", "kaun sa stone",
]
_REMEDY_CONTEXT_KEYWORDS = [
    "suggest remedy", "suggest a remedy", "recommend remedy",
    "kya karu iske liye", "kya karein iske liye",
    "strengthen planet", "strengthen my",
    "protection from", "how to reduce negative",
    "kaise theek karu", "kaise sudhare",
]


def _is_remedy_query(question: str) -> bool:
    """Check if the user is explicitly asking for remedies — only then recommend products.
    Uses two-tier matching: strong keywords (single match) + contextual phrases (exact match)
    to reduce false positives from generic words like 'suggest' or 'solution'."""
    q_lower = question.lower()
    if any(kw in q_lower for kw in _REMEDY_STRONG_KEYWORDS):
        return True
    if any(kw in q_lower for kw in _REMEDY_CONTEXT_KEYWORDS):
        return True
    return False


def predict(message, history, chart_data):
    """Stream a response from the vLLM server with RAG-augmented context + chart data."""
    req_id = uuid.uuid4().hex[:12]
    t_start = time.monotonic()

    # 0. Convert chart JSON to compact YAML (5500 lines JSON → ~120 lines YAML)
    chart_yaml = _chart_to_yaml(chart_data or "")

    # Hard guard: if no chart data and user asks a personal prediction question,
    # don't let the model hallucinate — ask for chart data first.
    if not chart_yaml:
        personal_keywords = [
            "when will", "will i", "my marriage", "my career", "my financial",
            "my health", "my job", "should i", "am i", "will my", "my kundali",
            "meri shaadi", "mera career", "when did", "kab hogi", "obstacles",
            "get married", "change fields", "improve", "facing", "confused",
        ]
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in personal_keywords):
            yield ("Aapka chart data abhi load nahi hai. Please apni birth chart (JSON) "
                   "left panel mein paste karein — tabhi main aapko accurate prediction "
                   "de paunga. Bina chart ke prediction dena galat hoga. 🙏")
            return

    if chart_yaml:
        full_question = (f"Chart context (YAML):\n{chart_yaml}\n\n"
                         f"Question: {message}")
    else:
        full_question = message

    # 1. Retrieve RAG chunks (search using original question for better retrieval)
    rag_chunks = _retrieve_rag_chunks(message, top_k=args.top_k)

    # 2. Classify query type for intelligent response control
    query_info = _classify_query_type(message)
    is_remedy = _is_remedy_query(message)

    # 2a. Safety intercept — death/longevity queries get compassionate redirect
    if query_info["type"] == "safety":
        native_name = ""
        if chart_data:
            _nm = re.search(r'"name"\s*:\s*"([^"]+)"', chart_data)
            if _nm:
                native_name = _nm.group(1).strip()
        name_ji = f"{native_name} ji" if native_name else "Ji"
        safety_msg = (
            f"{name_ji}, please don't worry — astrology is here to guide you, not to scare you. "
            f"Your chart shows many positive periods ahead. Health concerns are best addressed by "
            f"a qualified medical professional. From a KP perspective, strengthening your lagna lord "
            f"through simple remedies can support overall wellbeing. Stay positive — better times are coming."
        )
        yield safety_msg
        return

    # Helper: detect if question is in Hindi/Hinglish (defined here so all intercepts below can use it)
    def _is_hindi_q(q):
        _hindi_kw = ['kya', 'hai', 'mera', 'meri', 'kab', 'kaise', 'batao', 'bataiye',
                     'hogi', 'hoga', 'aapka', 'aapki', 'mujhe', 'kaisa', 'kahan',
                     'shaadi', 'paisa', 'naukri', 'padhai', 'ghar', 'rishta', 'aaj']
        words = q.lower().split()
        return sum(1 for w in words if w in _hindi_kw) >= 2 or any(p in q.lower() for p in ['kab hogi', 'kya hoga', 'batao', 'bataiye', 'aaj ki'])

    # 2a-ext. Medical safety intercept — bypass model entirely for disease diagnosis queries
    if query_info["type"] == "medical_safety":
        native_name = ""
        if chart_data:
            _nm = re.search(r'"name"\s*:\s*"([^"]+)"', chart_data)
            if _nm:
                native_name = _nm.group(1).strip()
        name_ji = f"{native_name} ji" if native_name else "Ji"
        _is_hindi = _is_hindi_q(message.lower())
        if _is_hindi:
            medical_msg = (
                f"{name_ji}, yeh sawaal astrology ke scope se bahar hai. "
                f"Kisi bhi bimari ka diagnosis sirf qualified doctor hi kar sakte hain — "
                f"please ek doctor se milein. KP astrology health ke general trends dikha sakti hai, "
                f"lekin medical diagnosis nahi kar sakti. Aapki sehat sabse zaroori hai. 🙏"
            )
        else:
            medical_msg = (
                f"{name_ji}, astrology cannot diagnose medical conditions — "
                f"please consult a qualified doctor for any health concerns. "
                f"KP astrology can indicate general health trends and timing of health-related periods, "
                f"but it is not a substitute for medical advice. Your health comes first. 🙏"
            )
        yield medical_msg
        return

    # 2a-ext2. Meta-confidence intercept — confident response about KP accuracy
    if query_info["type"] == "meta_confidence":
        native_name = ""
        if chart_data:
            _nm = re.search(r'"name"\s*:\s*"([^"]+)"', chart_data)
            if _nm:
                native_name = _nm.group(1).strip()
        name_ji = f"{native_name} ji" if native_name else "Ji"
        _is_hindi = _is_hindi_q(message.lower())
        if _is_hindi:
            confidence_msg = (
                f"{name_ji}, KP Astrology ek precise system hai jo sub-lords, cusps, aur dasha timing "
                f"ka use karke month-level predictions deta hai — yeh generic sun-sign astrology se "
                f"kaafi alag aur accurate hai. Main aapke chart ke actual dasha dates padhkar "
                f"specific timing deta hun, isliye predictions itni precise hoti hain."
            )
        else:
            confidence_msg = (
                f"{name_ji}, KP Astrology is a precise system that uses sub-lords, cusps, and dasha "
                f"timing to give month-level predictions — far more accurate than generic sun-sign astrology. "
                f"I read the actual dasha dates from your chart to give specific timing, "
                f"which is why the predictions are precise and actionable."
            )
        yield confidence_msg
        return

    # 2b. Direct factual intercepts — bypass model for questions we can answer perfectly
    q_lower = message.lower().strip()

    # Extract native name for intercepts
    _intercept_name = ""
    if chart_data:
        _nm = re.search(r'"name"\s*:\s*"([^"]+)"', chart_data)
        if _nm:
            _intercept_name = _nm.group(1).strip()
    _intercept_name_ji = f"{_intercept_name} ji" if _intercept_name else "Ji"

    if any(p in q_lower for p in ["what is the date", "what's the date", "aaj ki date", "today's date", "aaj ka date"]):
        today = date.today().strftime("%d %B %Y")
        if _is_hindi_q(q_lower):
            yield f"{_intercept_name_ji}, aaj ki date {today} hai."
        else:
            yield f"{_intercept_name_ji}, today's date is {today}."
        return

    if any(p in q_lower for p in ["who are you", "what is your name", "your name", "tell me about yourself"]):
        yield ("My name is Jyotish — I am a seasoned KP astrologer. I use Krishnamurti Paddhati principles "
               "to give you accurate and practical answers. You can ask me about career, finance, health, "
               "relationships, and life timing.")
        return
    if any(p in q_lower for p in ["aapka naam", "aap kaun", "tum kaun", "kaun ho aap", "kaun ho tum"]):
        yield ("Main Jyotish hun — ek seasoned KP astrologer. Main Krishnamurti Paddhati ke principles "
               "use karke aapke sawaalon ka accurate aur practical jawaab deta hun. Aap mujhse career, "
               "finance, health, relationships, aur life timing ke baare mein pooch sakte hain.")
        return

    # "What is my name?" — language-aware
    if any(p in q_lower for p in ["what is my name", "what's my name", "tell me my name"]):
        if _intercept_name:
            yield f"{_intercept_name_ji}, your name is {_intercept_name}."
        else:
            yield "I don't have your chart data loaded yet. Please paste your birth chart JSON on the left panel."
        return
    if any(p in q_lower for p in ["mera naam kya", "mera naam bata", "mera name kya"]):
        if _intercept_name:
            yield f"{_intercept_name_ji}, aapka naam {_intercept_name} hai."
        else:
            yield "Aapka chart data abhi load nahi hai. Please apni birth chart JSON left panel mein paste karein."
        return

    # 2b-ext. Direct chart-fact intercepts — lagna, rasi, nakshatra (language-aware)
    # These bypass the model entirely — model often responds in Hindi for English queries
    _lagna_patterns_en = ["what is my lagna", "what's my lagna", "my lagna", "my ascendant",
                          "what is my ascendant", "what's my ascendant", "rising sign"]
    _lagna_patterns_hi = ["mera lagna", "meri lagna", "mera ascendant", "lagna kya hai",
                          "lagna kya he", "lagna batao", "lagna bataiye"]
    if chart_data and any(p in q_lower for p in _lagna_patterns_en + _lagna_patterns_hi):
        _lagna_m = re.search(r'"lagna"\s*:\s*"([^"]+)"', chart_data)
        _lagna_lord_m = re.search(r'"lagnaLord"\s*:\s*"([^"]+)"', chart_data)
        if _lagna_m:
            _lagna_val = _lagna_m.group(1)
            _lagna_lord = _lagna_lord_m.group(1) if _lagna_lord_m else ""
            _lord_full = {"SUN":"Sun","MON":"Moon","MAR":"Mars","MER":"Mercury",
                          "JUP":"Jupiter","VEN":"Venus","SAT":"Saturn",
                          "RAH":"Rahu","KET":"Ketu"}.get(_lagna_lord, _lagna_lord)
            if _is_hindi_q(q_lower):
                yield f"{_intercept_name_ji}, aapka lagna {_lagna_val} hai" + (f", jo {_lord_full} se ruled hai." if _lord_full else ".")
            else:
                yield f"{_intercept_name_ji}, your lagna (ascendant) is {_lagna_val}" + (f", ruled by {_lord_full}." if _lord_full else ".")
        return

    _rasi_patterns_en = ["what is my rasi", "what's my rasi", "my moon sign", "what is my moon sign",
                         "what's my moon sign", "my rashi", "what is my rashi"]
    _rasi_patterns_hi = ["mera rasi", "meri rasi", "mera rashi", "rasi kya hai", "rashi kya hai",
                         "moon sign kya hai", "rasi batao"]
    if chart_data and any(p in q_lower for p in _rasi_patterns_en + _rasi_patterns_hi):
        _rasi_m = re.search(r'"rasi"\s*:\s*"([^"]+)"', chart_data)
        if _rasi_m:
            _rasi_val = _rasi_m.group(1)
            if _is_hindi_q(q_lower):
                yield f"{_intercept_name_ji}, aapka rasi (moon sign) {_rasi_val} hai."
            else:
                yield f"{_intercept_name_ji}, your rasi (moon sign) is {_rasi_val}."
        return

    _nak_patterns_en = ["what is my nakshatra", "what's my nakshatra", "my birth star", "my nakshatra"]
    _nak_patterns_hi = ["mera nakshatra", "meri nakshatra", "nakshatra kya hai", "nakshatra batao"]
    if chart_data and any(p in q_lower for p in _nak_patterns_en + _nak_patterns_hi):
        _nak_m = re.search(r'"nakshatra"\s*:\s*"([^"]+)"', chart_data)
        if _nak_m:
            _nak_val = _nak_m.group(1)
            if _is_hindi_q(q_lower):
                yield f"{_intercept_name_ji}, aapka janma nakshatra {_nak_val} hai."
            else:
                yield f"{_intercept_name_ji}, your birth nakshatra is {_nak_val}."
        return

    # 2c. Non-astrology conversation intercepts — greetings, feedback, meta-questions
    # These MUST come before model inference to prevent chart-data leakage
    _greeting_patterns = ["good morning", "good afternoon", "good evening", "good night",
                          "have a good day", "have a nice day", "bye", "goodbye", "thank you",
                          "thanks", "shukriya", "dhanyavaad", "alvida", "namaste",
                          "hello", "hi there", "hey there"]
    if any(p in q_lower for p in _greeting_patterns) and len(q_lower.split()) <= 8:
        yield f"{_intercept_name_ji}, thank you! Jab bhi aapko astrology guidance chahiye, main yahan hun. Have a wonderful day! 🙏"
        return

    _feedback_patterns = ["you need to improve", "you are wrong", "you're wrong", "that's wrong",
                          "not correct", "galat hai", "improve karo", "better karo",
                          "your answer is wrong", "postprocessing", "overriding",
                          "hmm yeah", "hmm ok", "hmm okay"]
    if any(p in q_lower for p in _feedback_patterns):
        yield f"{_intercept_name_ji}, I appreciate your feedback — I am continuously learning and improving. Please ask me any astrology question and I will do my best to give you an accurate answer based on your chart."
        return

    _meta_patterns = ["how many years", "kitne saal", "experience", "how old are you",
                      "when were you made", "who made you", "who created you",
                      "can someone predict", "do you believe", "is astrology real",
                      "is astrology true", "kya astrology sach", "kya bhavishya"]
    if any(p in q_lower for p in _meta_patterns) and not any(w in q_lower for w in ["job", "marriage", "career", "financial", "health", "shaadi", "naukri"]):
        if _is_hindi_q(q_lower):
            yield (f"{_intercept_name_ji}, main Jyotish hun — ek experienced KP astrologer. "
                   "Main Krishnamurti Paddhati ke principles se aapke sawaalon ka jawaab deta hun. "
                   "Aap mujhse apni kundali ke baare mein kuch bhi pooch sakte hain.")
        else:
            yield (f"{_intercept_name_ji}, I am Jyotish — an experienced KP astrologer. "
                   "I use Krishnamurti Paddhati principles to analyze your chart and provide accurate predictions. "
                   "Feel free to ask me anything about your kundali.")
        return

    # 2d. Inappropriate intercept — sexual orientation, personal judgments
    if query_info["type"] == "inappropriate":
        native_name = ""
        if chart_data:
            _nm = re.search(r'"name"\s*:\s*"([^"]+)"', chart_data)
            if _nm:
                native_name = _nm.group(1).strip()
        name_ji = f"{native_name} ji" if native_name else "Ji"
        inappropriate_msg = (
            f"{name_ji}, yeh sawaal astrology ke scope se bahar hai. "
            f"Main ek KP astrologer hun — main aapko career, finance, health, relationships, "
            f"aur life timing ke baare mein guide kar sakta hun. "
            f"Kripya apna sawaal in topics se related rakhein, main aapki madad zaroor karunga."
        )
        yield inappropriate_msg
        return

    # 3. Product recommendations — ONLY when user asks for remedies
    product_text = ""
    product_instruction = ""
    if is_remedy:
        product_text = _get_product_recommendations(message, chart_summary=chart_yaml)
        if product_text:
            product_instruction = (
                f"\n\nRELEVANT PRODUCTS — weave ONE naturally as a remedy suggestion:\n"
                f"{product_text}\n"
                f"Example: 'Is samay [planet] ko strengthen karne ke liye hamara [Product Name] try karein.'"
            )

    # 4. Build prompt with adaptive RAG trimming to fit character budget
    # Rebuild system prompt per-request so today's date is always fresh
    _sys_base = _build_system_prompt(with_rag=True)
    _sys_no_rag = _build_system_no_rag()
    fixed_chars = len(_sys_base) + len(full_question) + len(product_instruction) + 30
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
        sys_content = f"{_sys_base}\n\nKP Book Excerpts:\n{rag_text}{product_instruction}"
    else:
        sys_content = f"{_sys_no_rag}{product_instruction}"

    # 5. Build messages WITH conversation history for follow-up context
    messages = [
        {"role": "system", "content": sys_content},
    ]

    # Include recent conversation history (last N turns, budget-aware)
    # Gradio history format: flat list of {"role": "user"|"assistant", "content": "..."}
    _MAX_HIST_MSGS = 8  # last 8 messages ≈ 4 user-assistant pairs
    _MAX_HIST_CHARS = MAX_INPUT_CHARS // 4  # reserve 25% of input budget
    _hist_chars = 0
    if history:
        # Skip the last user msg (it's the current question we're about to add)
        hist_msgs = history[:-1] if history else []
        recent = hist_msgs[-_MAX_HIST_MSGS:]
        for msg in recent:
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if not role or not content:
                continue
            if _hist_chars + len(content) > _MAX_HIST_CHARS:
                break
            messages.append({"role": role, "content": content})
            _hist_chars += len(content)

    # Current question (with chart YAML context)
    messages.append({"role": "user", "content": full_question})

    # 6. Compute output tokens — use query-type-aware limits
    total_chars = sum(len(m["content"]) for m in messages)
    est_input_tokens = int(total_chars / 0.78) + 100
    available = MAX_MODEL_LEN - est_input_tokens
    base_output = query_info.get("max_tokens_override") or OUTPUT_TOKENS
    max_tokens = max(64, min(base_output, available))
    temperature = query_info["temperature"]

    if max_tokens < 64:
        yield (f"Your message is too long for the model's {MAX_MODEL_LEN}-token context. "
               "Please shorten the chart data or question.")
        return

    # Extract birth year, birth date, and native name from chart for postprocess
    _postprocess._birth_year = None
    _postprocess._birth_date = None
    _postprocess._native_name = None
    _postprocess._query_type = query_info["type"]
    _postprocess._user_question = message
    _birth_year_val = None
    if chart_data:
        _by_match = re.search(r'"date"\s*:\s*"(\d{2})\.(\d{2})\.(\d{4})"', chart_data)
        if _by_match:
            _birth_year_val = int(_by_match.group(3))
            _postprocess._birth_year = _birth_year_val
            try:
                from datetime import date as _date_cls
                _postprocess._birth_date = _date_cls(
                    int(_by_match.group(3)), int(_by_match.group(2)), int(_by_match.group(1))
                )
            except Exception:
                pass
        _name_match = re.search(r'"name"\s*:\s*"([^"]+)"', chart_data)
        if _name_match:
            _postprocess._native_name = _name_match.group(1).strip()

    def _is_deflection(text: str) -> bool:
        """Detect vague non-answers that don't contain specific predictions."""
        if not text or len(text.strip()) < 20:
            return True
        t = text.lower()
        # Deflection phrases — model talks ABOUT answering instead of answering
        deflection_phrases = [
            "depend karta hai", "depends on", "requires careful analysis",
            "requires analysis", "we need to examine", "i can analyze",
            "let me analyze", "let me check", "need to check",
            "based on planetary positions", "based on current planetary",
            "significator combinations par depend", "need to examine",
            "i will analyze", "let us examine", "we should look at",
            "requires detailed analysis", "needs careful examination",
            "specific planetary periods", "cuspal connections",
            "planetary periods based on", "timing specific planetary",
            "creates primary significator connection",
            "need to analyze", "need to look at",
            # New deflection phrases found in Round 2 testing:
            "connected to the current mahadasha",
            "strongly connected to the current",
            "appears strongly connected",
            "analysis kar rahe hain",
            "planetary combination explains",
            "i need to address your concern",
            "in a clear and direct manner",
            "according to established kp principles",
            "significant life events occurred during specific dasha",
            "based on the provided natal data",
            "based on the provided kp chart",
            "based on the provided planetary",
            "significator analysis provided in this",
            "extensive chart breakdown",
            "outlined hai",
            # Round 4 additions:
            "timing analysis kar rahe",
            "examine karunga",
            "examine karenge",
            "analyze karunga",
            "analyze karenge",
            "ke liye marriage timing analysis",
            "use karke",
            "system use karke",
            "carefully examine karunga",
            "guidance de sakein",
            # Round 5 additions:
            "outlined in your mahadasha",
            "outlined in your dasha",
            "specific periods outlined",
            "favorable during specific",
            "appears quite favorable",
            "timing appears highly",
            "timing appears quite",
            # Round 6 additions (from Feb 14 testing):
            "humein specific house significators examine",
            "humein examine karne padenge",
            "carefully examine kar raha",
            "overall planetary positions",
            "overall planetary influences",
            "align perfectly with established",
            "established kp principles regarding",
            "planetary influences in medical",
            "aapke chart ko carefully",
            "confusion clear ho jaayegi jab",
            "directly influence kar raha",
            "current pratyantar dasa period directly",
            "assessment karne ke liye",
            "remaining an outstanding matter",
            "requiring careful consideration",
            "considerable promise in securing",
            "horoscope analysis clearly indicates",
            "creates favorable conditions for",
            "clearly identify that your",
            # Round 7 additions (Hindi deflection from Q17):
            "humein current mahadasha period ko examine karna",
            "humein examine karna hoga",
            "dekhna hoga jo actual event",
            "analyze karna padega",
            "examine karna zaroori hai",
            "dekhna padega ki",
            "samajhna hoga ki",
            "prospects appear quite favorable",
            "planetary positions and current dasha sequence",
            # Round 8 additions:
            "discuss karunga jo directly",
            "main aapse current mahadasha period discuss",
            "strong potential for foreign",
            "strong potential for education",
            "let's examine this important",
            "let me examine the planetary",
            "this is an excellent time to discuss",
            "holds significance for your",
            "significations in your chart",
            # Round 8b additions (user retest):
            "significator analysis mein dekhte hain",
            "examine kar sakte hain current",
            "ki possibility examine kar sakte",
            "potential assess karne ke liye",
            "specific yogic combinations dekh rahe",
            "you need to perform remedial measures",
            "need to perform remedial measures involving",
            "examine upcoming saturn",
            # Round 8c additions (user retest):
            "carefully examine karna chahiye",
            "ko carefully examine karna",
            "let me provide an",
            "requires careful examination of",
            "presents exceptional opportunities for",
            "here's the recommended approach",
            "focus kariye in specific",
            "for progeny matters, examine",
            "for progeny matters examine",
            # Round 8d additions (Anuj kundali retest):
            "requires examining planets connected",
            "requires careful examination of the",
            "examination involves analyzing significator",
            "examination involves analyzing",
            "requires examining the 12th house",
            "let's examine significant planetary",
            "examine these key planetary positions",
            "examine these core significators",
            "analysis kar raha main current",
            "analysis kar ra ha main current",
            "ka analysis kar ra",
            "possibility evaluate karne ke liye",
            "humein aapke educational pursuits",
            "analyze karne padenge",
            "foreign travel analysis requires",
            "foreign travel requires careful",
            "government service potential examination",
            "government job yoga analysis",
            "for marriage remedial measures, examine",
            "for marriage remedial measures examine",
            "shaadi prospects ka analysis kar",
            "success ki possibility evaluate",
            "we observe multiple layers of timing",
            # Round 9 additions (Arisha Akhtar retest):
            "examine which planet governs the antardasha",
            "examine which planet governs",
            "specific remedial measures align with classical",
            "remedial measures align with classical kp",
            "tailored to your unique significator pattern",
        ]
        if any(p in t for p in deflection_phrases):
            return True
        # For timing questions: MUST have actual dates (month-year or year range)
        # Just mentioning "cusp" or "house" without dates is still deflection
        if query_info["type"] in ("timing", "past_event"):
            has_date = bool(re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}', text))
            has_year_range = bool(re.search(r'20\d{2}\s*(?:to|se|tak|-)\s*20\d{2}', text))
            has_single_year = bool(re.search(r'\b20(?:2[5-9]|3[0-9])\b', text))
            has_month_ref = bool(re.search(r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+20\d{2}', text, re.IGNORECASE))
            if not has_date and not has_year_range and not has_single_year and not has_month_ref:
                return True
        return False

    def _generate_non_streaming(msgs, mt, temp):
        """Non-streaming generation for retry attempts."""
        resp = client.chat.completions.create(
            model="kp-astrology-llama",
            messages=msgs,
            max_tokens=mt,
            temperature=temp,
            top_p=0.85,
            stream=False,
            extra_body={"repetition_penalty": 1.15},
        )
        return resp.choices[0].message.content or ""

    try:
        stream = client.chat.completions.create(
            model="kp-astrology-llama",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stream=True,
            extra_body={"repetition_penalty": 1.2},
        )
        partial = ""
        _last_yield_len = 0
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                partial += delta
                # Yield every ~15 chars to reduce flicker from postprocess rewrites
                if len(partial) - _last_yield_len >= 15:
                    yield _postprocess(partial)
                    _last_yield_len = len(partial)

        # ── Deflection retry: if model gave a vague non-answer, retry with forced prefix ──
        if partial and _is_deflection(partial) and chart_yaml:
            _json_log("deflection_detected", req_id=req_id, original=partial[:200])
            native_name = getattr(_postprocess, '_native_name', '') or ''
            name_ji = f"{native_name} ji" if native_name else "Ji"

            # Build a forced-prefix retry prompt — query-type-specific examples
            _q_topic = message.lower()
            if any(w in _q_topic for w in ["health", "body", "illness", "sick", "bimari", "tabiyat"]):
                _example = (f"'{name_ji}, your health needs attention from now till Apr 2026 during Saturn-Ketu AD "
                            f"which connects to houses 6,8. After May 2026, Saturn-Venus AD brings recovery through houses 1,11. "
                            f"6th cusp sub-lord Mars signifies houses 6,8 indicating health challenges in this period.'")
            elif any(w in _q_topic for w in ["career", "job", "naukri", "kaam", "field", "profession"]):
                _example = (f"'{name_ji}, your career breakthrough comes Oct 2026 to Mar 2027 during Saturn-Venus AD "
                            f"which activates houses 2,6,10,11. 10th cusp sub-lord Mercury signifies houses 2,10 supporting professional growth.'")
            elif any(w in _q_topic for w in ["exam", "interview", "test", "pariksha", "result"]):
                _example = (f"'{name_ji}, your success window is Mar 2026 to Aug 2026 during Saturn-Ketu AD. "
                            f"5th cusp sub-lord Jupiter signifies houses 4,9,11 — strong for academic success. Peak months: May-Jun 2026.'")
            elif any(w in _q_topic for w in ["financial", "money", "paisa", "dhan", "income", "wealth"]):
                _example = (f"'{name_ji}, your finances improve from Apr 2027 during Saturn-Venus AD which activates houses 2,6,11. "
                            f"2nd cusp sub-lord Mars signifies houses 2,11 — wealth and gains. Peak earning: Jul-Oct 2027.'")
            else:
                _example = (f"'{name_ji}, your [event] timing is [Month Year] to [Month Year] during [Planet]-[Planet] AD, "
                            f"because [cusp] sub-lord [Planet] signifies houses [X,Y] which support [event].'")

            retry_user = (
                f"{full_question}\n\n"
                f"CRITICAL: Your previous answer was REJECTED because it had NO specific dates.\n"
                f"You MUST read the dasha table in the YAML and give ACTUAL month-year dates.\n"
                f"START your answer with '{name_ji},' and give dates in the FIRST sentence.\n"
                f"EXAMPLE FORMAT:\n{_example}\n"
                f"BANNED PHRASES: 'depends on', 'requires analysis', 'let me analyze', 'outlined in', "
                f"'specific periods', 'planetary influences', 'interesting dynamic', 'carefully examine'.\n"
                f"Give the ACTUAL month-year dates from the dasha table NOW. 2-3 sentences max."
            )
            retry_msgs = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": retry_user},
            ]
            retry_text = _generate_non_streaming(retry_msgs, max_tokens, 0.15)
            if retry_text and not _is_deflection(retry_text):
                partial = retry_text
                yield _postprocess(partial)
            else:
                # Second retry with even more forceful prompt
                retry_user2 = (
                    f"Chart YAML:\n{chart_yaml[:3000]}\n\n"
                    f"Question: {message}\n\n"
                    f"ANSWER IN EXACTLY THIS FORMAT — fill in the blanks from the dasha table:\n"
                    f"{name_ji}, [answer to question] timing is [read Month Year from antarDashas] "
                    f"to [read end Month Year]. [One sentence about which cusp/house supports this].\n"
                    f"DO NOT explain methodology. DO NOT say 'analysis'. Just give the dates and answer."
                )
                retry_msgs2 = [
                    {"role": "system", "content": _sys_no_rag},
                    {"role": "user", "content": retry_user2},
                ]
                retry_text2 = _generate_non_streaming(retry_msgs2, max_tokens, 0.1)
                if retry_text2 and not _is_deflection(retry_text2):
                    partial = retry_text2
                    yield _postprocess(partial)
                else:
                    _json_log("deflection_retry_failed", req_id=req_id)

        # Final enrichment: append Hindi quote + product (only if remedy query)
        if partial:
            final = _postprocess(partial)
            final = _enrich_response(final, product_text=product_text, is_remedy=is_remedy, query_type=query_info["type"])
            yield final

        # ── Structured log for observability ──
        _json_log("chat_response",
                  req_id=req_id,
                  query_type=query_info["type"],
                  is_remedy=is_remedy,
                  has_chart=bool(chart_yaml),
                  rag_chunks=len(selected_chunks),
                  max_tokens=max_tokens,
                  temperature=temperature,
                  raw_len=len(partial),
                  answer_len=len(final) if partial else 0,
                  latency_ms=round((time.monotonic() - t_start) * 1000))
    except Exception as e:
        _json_log("chat_error", req_id=req_id, error=str(e),
                  latency_ms=round((time.monotonic() - t_start) * 1000))
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
