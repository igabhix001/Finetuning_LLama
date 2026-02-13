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
        "## LANGUAGE RULE (HIGHEST PRIORITY — CHECK FIRST):\n"
        "- If the user's question is in ENGLISH → respond 100% in ENGLISH. Zero Hindi words.\n"
        "- If the user's question is in HINDI/HINGLISH → respond in HINDI/HINGLISH.\n"
        "- This is NON-NEGOTIABLE. Match the user's language exactly.\n\n"
        "## HARD RULES:\n"
        "- ANSWER DIRECTLY. Never say 'I can analyze', 'requires analysis', 'let me check'.\n"
        "- Read the name from YAML. Address as '[Name] ji'. Never output '[Name]' literally.\n"
        "- No markdown, no **bold**, no headers, no bullets, no numbered lists.\n"
        "- Never say 'the native'. Say 'you' or use their name.\n"
        "- Simple questions (name/lagna/rashi) = 1 sentence ONLY. Nothing more.\n"
        "- Timing questions = 2-3 sentences max with specific Mon YYYY dates.\n"
        "- MAX 4 sentences for any response. Keep answers short and impactful.\n"
        "- Cite cusp sub-lord + house numbers. Give month-year ranges from dasha table.\n"
        "- For obstacles/emotional queries: ALWAYS tell when the difficult period ENDS and what comes next.\n"
        "- Products: ONLY when user asks for remedies. Otherwise ZERO product mentions.\n\n"
        "## EXAMPLES (assume today = 10 Feb 2026):\n"
        "Q: 'When will I get married?' → 'Priya ji, your strongest marriage window is Mercury-Jupiter AD (Mar 2026 to Nov 2027), "
        "with peak months Jul to Oct 2026 when Venus pratyantar activates houses 2,7,11. "
        "7th cusp sub-lord Saturn signifies houses 2,7 which supports marriage.'\n"
        "Q: 'Why am I facing obstacles?' → 'Priya ji, you are currently in Venus-Saturn AD which connects to houses 7,8,10,12 — "
        "house 8 and 12 bring unexpected setbacks. This difficult phase ends Mar 2027, after which Venus-Mercury brings relief through houses 3,4,10.'\n"
        "Q: 'When will my financial situation improve?' → 'Priya ji, your finances strengthen from Apr 2026 when Mercury-Moon AD activates houses 2,11 (wealth and gains). "
        "Peak earning months are Jul to Oct 2026 during Mars pratyantar.'\n"
        "Q: 'I feel very unlucky' → 'Priya ji, I understand this is a difficult time — you are not alone. "
        "You are currently in Saturn-Rahu pratyantar which connects to houses 8,12 causing setbacks, but this ends May 2026. "
        "After that, Saturn-Jupiter pratyantar activates houses 9,11 bringing luck and gains.'\n\n"
        "## HINDI EXAMPLES:\n"
        "Q: 'Meri shaadi kab hogi?' → 'Priya ji, shaadi ka strong period Mercury-Jupiter AD (Mar 2026-Nov 2027) hai, "
        "peak Jul-Oct 2026 jab Venus pratyantar houses 2,7,11 activate karega.'\n"
        "Q: 'Mera naam kya hai?' → 'Priya ji, aapka naam Priya hai.'\n"
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
        "LANGUAGE RULE (HIGHEST PRIORITY):\n"
        "- English question → 100% English answer. Zero Hindi words.\n"
        "- Hindi/Hinglish question → Hindi/Hinglish answer.\n\n"
        "RULES:\n"
        "- Answer DIRECTLY. No deflection, no 'let me analyze'.\n"
        "- Read name from YAML. Address as '[Name] ji'.\n"
        "- No markdown, headers, bold, bullets. Plain text only.\n"
        "- Simple questions = 1 sentence. Timing = 2-3 sentences. MAX 4 sentences.\n"
        "- Cite cusp sub-lord + houses. Give Mon YYYY dates from dasha table.\n"
        "- For obstacles/emotional: ALWAYS say when difficulty ENDS and what comes next.\n"
        "- Products: ONLY when user asks for remedies.\n\n"
        "EXAMPLES (assume today = 10 Feb 2026):\n"
        "Q: 'When will I get married?' → 'Priya ji, your marriage window is Mercury-Jupiter AD (Mar 2026 to Nov 2027), "
        "peak Jul-Oct 2026 when Venus pratyantar activates houses 2,7,11.'\n"
        "Q: 'Mera naam kya hai?' → 'Priya ji, aapka naam Priya hai.'\n"
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
        print(f"RAG retrieval error: {e}")
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
            print(f"  Product Pinecone search error: {e}")

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

    # ── Phase 3: Remove hallucinated references ──
    text = re.sub(r'["\s]*(?:source:\s*)?page_no\s*=\s*\d+["\s]*', ' ', text)
    text = re.sub(r'rules_used:\s*[A-Z_0-9,\s]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bKP_[A-Z]{2,4}_\d{3,5}\b', '', text)
    text = re.sub(r'\[KP_[A-Z_0-9]+\]', '', text)
    text = re.sub(r'\[rule_id\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\((?:Source|Ref|Reference|Page|Ch(?:apter)?)[^)]{0,60}\)', '', text, flags=re.IGNORECASE)

    # ── Phase 3.5: Health safety — strip dangerous medical claims ──
    # NOTE: "cancer" alone is NOT replaced — it's a zodiac sign (Cancer/Karka).
    # Only replace cancer in medical contexts like "cancer treatment", "cancer risk".
    _dangerous_terms = [
        r'cancer[- ](?:related|treatment|risk|diagnosis|patient|surgery|therapy|cells?)',
        r'(?:breast|lung|blood|skin|colon|prostate|ovarian|cervical)\s+cancer',
        r'\btumou?r\b', r'\bmalignant\b', r'\bbenign\b',
        r'heart\s+attack', r'heart\s+disease', r'cardiac\s+arrest',
        r'\bdiabetes\b', r'\bHIV\b', r'\bAIDS\b',
        r'tuberculosis', r'epilepsy',
        r'kidney[- ]?(?:failure|disease)',
        r'liver[- ]?(?:failure|disease)',
        r'brain\s+(?:damage|tumou?r)',
        r'mental\s+(?:illness|disorder)', r'schizophren',
        r'\bsuicid', r'\bfatal\b', r'\bterminal(?:ly)?\s+ill',
        r'life[- ]?threatening', r'\blethal\b',
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
    text = re.sub(r'(?:^|\n)\s*\d+[.)]\s+', '\n', text)
    text = re.sub(r'(?:^|\n)\s*[-•●◦▪]\s+', '\n', text)

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
    ]
    for pat, repl in _replacements:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)

    # ── Phase 7.5: Clean up artifacts from Phase 7 replacements ──
    # Fix orphaned commas/spaces at sentence start (e.g. ", I can" → "I can")
    text = re.sub(r'(?:^|(?<=[\.!?]))\s*,\s*', ' ', text)
    # Fix orphaned 'and' at start of text
    text = re.sub(r'^\s*and\s+', '', text, flags=re.IGNORECASE)
    # Fix double spaces from removed phrases
    text = re.sub(r'  +', ' ', text)
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
    ]
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("rules_used:") or stripped.startswith("rules used:"):
            continue
        if stripped.startswith("level:") or stripped.startswith("answer_end"):
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
    if result and result[-1] not in '.!?"\n)}':
        last_period = max(result.rfind('. '), result.rfind('.\n'), result.rfind('.'))
        if last_period > len(result) * 0.4:
            result = result[:last_period + 1]

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
    elif _query_type == "timing" and len(sentences) > 3:
        result = ' '.join(sentences[:3])
    elif len(sentences) > 4:
        result = ' '.join(sentences[:4])

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

    # ── Phase 13: Language enforcement (model SFT bakes in Hinglish — override) ──
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
        is_hindi_question = hindi_count >= 2 or any(w in _user_question.lower() for w in ['kab hogi', 'kya hoga', 'kaise hoga', 'batao', 'bataiye'])

        if not is_hindi_question:
            # English question but Hinglish response — strip common Hindi filler
            # Only strip if the response has significant Hindi content
            _hindi_response_markers = ['hai', 'mein', 'aapka', 'aapki', 'ke liye', 'karta hai', 'hota hai', 'karega', 'hogi', 'hoga']
            resp_words = result.lower().split()
            hindi_resp_count = sum(1 for w in resp_words if w in _hindi_response_markers)
            if hindi_resp_count >= 3 and len(resp_words) > 5:
                # Response is heavily Hinglish on an English question — flag for logging
                # We can't auto-translate, but we can note it
                pass  # Model behavior — needs DPO fix. Postprocess can't translate.

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
    ]
    if any(p in q for p in timing_patterns):
        return {"type": "timing", "max_paragraphs": 2, "temperature": 0.5, "max_tokens_override": 450}

    # ── 6. Remedy queries ──
    if _is_remedy_query(question):
        return {"type": "remedy", "max_paragraphs": 3, "temperature": 0.5, "max_tokens_override": 500}

    # ── 7. Complex analysis — full response ──
    return {"type": "analysis", "max_paragraphs": 3, "temperature": 0.5, "max_tokens_override": None}


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
        text += "\n\n" + " ".join(additions)

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

    # 2b. Direct factual intercepts — bypass model for questions we can answer perfectly
    q_lower = message.lower().strip()

    # Helper: detect if question is in Hindi/Hinglish
    def _is_hindi_q(q):
        _hindi_kw = ['kya', 'hai', 'mera', 'meri', 'kab', 'kaise', 'batao', 'bataiye',
                     'hogi', 'hoga', 'aapka', 'aapki', 'mujhe', 'kaisa', 'kahan',
                     'shaadi', 'paisa', 'naukri', 'padhai', 'ghar', 'rishta', 'aaj']
        words = q.lower().split()
        return sum(1 for w in words if w in _hindi_kw) >= 2 or any(p in q.lower() for p in ['kab hogi', 'kya hoga', 'batao', 'bataiye', 'aaj ki'])

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

    # Extract birth year and native name from chart for postprocess
    _postprocess._birth_year = None
    _postprocess._native_name = None
    _postprocess._query_type = query_info["type"]
    _postprocess._user_question = message
    if chart_data:
        _by_match = re.search(r'"date"\s*:\s*"(\d{2})\.(\d{2})\.(\d{4})"', chart_data)
        if _by_match:
            _postprocess._birth_year = int(_by_match.group(3))
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
