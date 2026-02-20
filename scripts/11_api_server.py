"""
REST API for KP Astrology LLM — returns structured JSON for app integration.

Response format:
{
  "answer": "conversational astrology text",
  "prediction": "specific date/time prediction if any",
  "product_reco": {"sku": "...", "title": "...", "price": "..."} or null
}

Usage:
  # Start vLLM server first:
  python scripts/08_serve_vllm.py

  # Then start this API:
  python scripts/11_api_server.py --products-csv /workspace/products_export_2026-02-03.csv

  # API will be at http://0.0.0.0:8080
  # POST /chat  — main endpoint
  # GET  /health — health check
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
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# ── Structured JSON logging ──────────────────────────────────────────────────
_log = logging.getLogger("kp_api")
_log.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(message)s'))
_log.addHandler(_handler)

def _json_log(event: str, **kwargs):
    """Emit a single-line JSON log entry for observability."""
    entry = {"ts": datetime.utcnow().isoformat() + "Z", "event": event}
    entry.update(kwargs)
    _log.info(json.dumps(entry, ensure_ascii=False, default=str))

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="KP Astrology REST API")
parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                    help="vLLM server URL")
parser.add_argument("--port", type=int, default=8080, help="API port")
parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
parser.add_argument("--no-rag", action="store_true",
                    help="Disable Pinecone RAG retrieval")
parser.add_argument("--top-k", type=int, default=5,
                    help="Number of RAG chunks to retrieve")
parser.add_argument("--max-model-len", type=int, default=8192,
                    help="vLLM max model length (default: 8192)")
parser.add_argument("--products-csv", type=str, default=None,
                    help="Path to products CSV for remedy recommendations")
args = parser.parse_args()

# ── Connect to vLLM backend ──────────────────────────────────────────────────
llm_client = OpenAI(base_url=args.vllm_url, api_key="not-needed")

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
            import subprocess
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
            print("  RAG:    DISABLED (missing keys)")
    except Exception as e:
        print(f"  RAG:    DISABLED ({e})")
else:
    print("  RAG:    DISABLED (--no-rag)")

# ── Product recommendations: Pinecone RAG only (no CSV fallback) ─────────────
if product_index:
    print("  Products: Pinecone RAG (semantic search)")
else:
    print("  Products: DISABLED (no Pinecone product index)")

# ── System prompts (dynamic — inject today's date) ──────────────────────────
def _build_system_prompt(with_rag=True):
    """Build system prompt with today's date injected dynamically."""
    _today = date.today().strftime("%d %b %Y")
    return (
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

SYSTEM_BASE = _build_system_prompt(with_rag=True)
SYSTEM_NO_RAG = _build_system_no_rag()

# ── Context-window budget ────────────────────────────────────────────────────
MAX_MODEL_LEN = args.max_model_len
OUTPUT_TOKENS = min(768, max(512, MAX_MODEL_LEN // 8))
INPUT_TOKEN_BUDGET = MAX_MODEL_LEN - OUTPUT_TOKENS - 100
MAX_INPUT_CHARS = int(INPUT_TOKEN_BUDGET * 0.78)
print(f"  Budget:  max_model_len={MAX_MODEL_LEN}, output={OUTPUT_TOKENS}, input_chars≈{MAX_INPUT_CHARS}")


# ── Helper functions ─────────────────────────────────────────────────────────
def _retrieve_rag_chunks(question, top_k=5):
    if not rag_index or not openai_client:
        return []
    try:
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
                product_list = []
                prompt_lines = []
                for m in results["matches"]:
                    meta = m["metadata"]
                    title = meta.get("title", "")
                    sku = meta.get("sku", "")
                    price = meta.get("price", "")
                    if title:
                        product_list.append({"sku": sku, "title": title, "price": price})
                        prompt_lines.append(f"- {title} (SKU: {sku}, Rs.{price})")
                if product_list:
                    return product_list, "\n".join(prompt_lines)
        except Exception as e:
            if not getattr(_get_product_recommendations, '_err_logged', False):
                print(f"  Product Pinecone search error (will suppress repeats): {e}")
                _get_product_recommendations._err_logged = True

    # No CSV fallback — products come only from Pinecone RAG
    return [], ""


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

    # ── Phase 0.5: Correct hallucinated wrong names — replace any "X ji" that isn't the real name ──
    if _native_name:
        _first_name = _native_name.split()[0]
        # Replace "WrongName ji," or "WrongName ji" at sentence start with correct name
        def _fix_wrong_name(m):
            wrong = m.group(1).strip()
            # If the name in text doesn't match our known name at all, replace it
            if wrong.lower() != _native_name.lower() and wrong.lower() != _first_name.lower():
                return f"{_native_name} ji"
            return m.group(0)
        text = re.sub(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+ji\b', _fix_wrong_name, text)

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
        r'\bbronchitis\b', r'\bpneumonia\b', r'\brespiratory\s+(?:issue|problem|infection|disease)\b',
        r'\blung[- ]?(?:infection|disease|issue|problem)\b',
        r'\bchest\s+(?:infection|pain|disease)\b',
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
    # Collapse lines that start mid-sentence (model outputs bullet content on new lines)
    text = re.sub(r'\n([a-z])', r' \1', text)
    # Collapse remaining single newlines
    text = re.sub(r'\n', ' ', text)
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
            return raw[:4]
        return raw[:4]
    text = re.sub(r'\b20\d{3,5}\b', _fix_year_typo, text)

    # ── Phase 6.6: Date sanity — strip sentences with years before birth year ──
    # Extract birth year from the text context (set by _generate_response via chart YAML)
    _birth_year = getattr(_postprocess, '_birth_year', None)
    if _birth_year and _birth_year > 1950:
        # Remove any standalone year mentions before birth year
        # e.g., "September 1969", "October 1970", "1980 to 1990"
        def _date_sanity(m):
            year = int(m.group(1))
            if year < _birth_year:
                return ''  # strip the hallucinated date phrase
            return m.group(0)
        # Strip "Month YYYY" where YYYY < birth year
        text = re.sub(
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(1[0-9]{3}|20[0-9]{2})',
            _date_sanity, text, flags=re.IGNORECASE
        )
        # Strip standalone "YYYY to YYYY" or "YYYY-YYYY" where start < birth year
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
        (r'\band\s+significator\s+analysis\s+(?:in\s+this\s+chart(?:\s+data)?)?,?\b', ''),
        (r'\banalysis\s+and\s+current\s+planetary\s+periods,?\b', ''),
        (r'\bmaking\s+this\s+combination\s+highly\s+favorable[^.!?]{0,60}', ''),
        (r'\bthis\s+combination\s+(?:is\s+)?highly\s+favorable[^.!?]{0,60}', ''),
        (r'\bcrucially\s+house\s+\d+\s+along\s+with\s+house\s+\d+\s+for\s+\w+\s+-\s+', ''),
        (r'\bfor\s+monetary\s+gains\.?\s*$', '.'),
        (r'\bthrough\s+(?:their|its)\s+combined\s+significations\s+of\s+houses[^.!?]{0,60}', ''),
        (r'\ball\s+crucial\s+for\s+(?:career|marriage|finance|health)\s+matters\.?', ''),
        (r"\bit's\s+clear\s+that\s+you're\s+at\s+crossroads[^.!?]{0,80}", ''),
        (r'\bSaturn\s+influencing\s+Mercury\s+through\s+their\s+combined\s+significations[^.!?]{0,80}', ''),
        (r'\bMercury\s+governs\s+intelligence,\s+communication\s+skills,\s+and\s+analytical\s+abilities[^.!?]{0,80}', ''),
        (r'\bwhile\s+Saturn\s+provides\s+discipline,\s+persistence,\s+and\s+methodical\s+approach[^.!?]{0,80}', ''),
        (r'\bnot\s+just\s+passing\s+but\s+achieving\s+substantial\s+recognition[^.!?]{0,80}', ''),
        (r'\bthrough\s+this\s+academic\s+pursuit\.?', ''),
        (r'\bappear\s+to\s+be\s+temporary\s+in\s+nature\s+according\s+to\s+KP\s+principles\.?', ''),
        (r'\baccording\s+to\s+KP\s+principles\.?', ''),
        (r'\baccording\s+to\s+KP\s+astrology\.?', ''),
        (r'\bper\s+KP\s+principles\.?', ''),
        (r'\bper\s+KP\s+astrology\.?', ''),
        (r'\bKP\s+principles\s+suggest[^.!?]{0,60}', ''),
        (r'\bKP\s+methodology\s+indicates[^.!?]{0,60}', ''),
        (r'\bappears\s+to\s+be\s+temporary\s+in\s+nature\.?', ''),
        (r'\byour\s+feelings\s+of\s+being\s+unlucky\s+appear[^.!?]{0,80}', ''),
        (r'\bfull\s+force\s+of\s+Saturn\'s\s+restrictive\s+influence[^.!?]{0,80}', ''),
        (r'\bSaturn\'s\s+(?:restrictive|limiting)\s+influence\s+combined\s+with[^.!?]{0,80}', ''),
        (r'\bMercury\'s\s+analytical\s+yet\s+sometimes\s+critical\s+energy\.?', ''),
        (r'\bSaturn-Mercury\s+antharam\s+starting\b', 'Saturn-Mercury period from'),
        (r'\bantharam\b', 'antardasha'),
        (r'\banthardasha\b', 'antardasha'),
        (r'\bBased\s+on\s+your\s+(?:natal\s+)?chart\s+(?:details|configuration)\s+and\s+current\s+(?:planetary\s+positions|dasha\s+sequence),?\b', ''),
        (r'\bBased\s+on\s+your\s+(?:natal\s+)?chart\s+(?:configuration|details)\s*,?\b', ''),
        (r'\bAccording\s+to\s+your\s+birth\s+data,?\b', ''),
        (r'\bBased\s+on\s+(?:the\s+)?current\s+planetary\s+periods?\s+(?:running\s+)?in\s+your\s+chart,?\b', ''),
        (r'\bbased\s+on\s+the\s+current\s+planetary\s+periods?\s+(?:running\s+)?in\s+your\s+(?:life|chart),?\b', ''),
        (r'\bbased\s+on\s+the\s+current\s+planetary\s+period\s+you\'re\s+experiencing[^,\.!?]{0,60}[,]?', ''),
        (r'\bbased\s+on\s+the\s+significator\s+analysis\s+in\s+your\s+chart,?\b', ''),
        (r'\bdepend\s+karte\s+hain\s+specific\s+planetary\s+combinations\s+par\s+jo\s+aapke\s+birth\s+chart\s+mein\s+signify\s+kar\s+rahe\s+hain\.?', ''),
        (r'^KP\s+Analysis\s+for\s+[A-Za-z\s]+Query\s*$', '', ),
        (r'\bKP\s+Analysis\s+for\s+[A-Za-z\s]+Query\s*\n?', ''),
        (r'\bhar\s+planet\s+ki\s+strength\s+ko\s+address\s+karna\s+padega\s+appropriate\s+remedial\s+measures\s+se\.?', ''),
        (r'\bKyunki\s+aapke\s+paas\s+multiple\s+planets\s+serve\s+kar\s+rahe\s+hain\s+houses\s+[\d,\s]+ke\s+saath\s+as\s+career\s+significators,?', ''),
        (r'\baapke\s+career\s+prospects\s+ke\s+liye\s+remedy\s+recommendations\s+depend\s+karte\s+hain[^.!?]{0,120}[.!?]?', ''),
        (r'\bremedy\s+recommendations\s+depend\s+(?:karte|karta)\s+hain[^.!?]{0,120}[.!?]?', ''),
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
        (r"\bSaturn's\s+natural\s+tendency\s+toward\s+limitation[^.!?]{0,120}[.!?]?", ''),
        (r'\bprogress\s+remains\s+blocked\s+despite\s+apparent\s+opportunities\s+emerging\.?', ''),
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
        (r'\bwas\s+highly\s+beneficial\s+for\s+all\s+ventures[^.!?]{0,60}[.!?]?', ''),
        (r'\bCurrent\s+Favorable\s+Period\s*:\s*[^.!?]{0,200}[.!?]?', ''),
        (r'\bShort-Term\s+Recovery\s*:\s*[^.!?]{0,200}[.!?]?', ''),
    ]
    for pat, repl in _replacements:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)

    # ── Phase 7.5: Clean up artifacts from Phase 7 replacements ──
    text = re.sub(r'(?:^|(?<=[\.!?]))\s*,\s*', ' ', text)
    # Fix orphaned comma after any period-space (e.g. "hain. , this" → "hain. This")
    text = re.sub(r'(\.\s*),\s*', r'\1', text)
    # Fix double comma after name+ji (e.g. "Rajesh ji, , your" → "Rajesh ji, your")
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'^\s*and\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'  +', ' ', text)
    # Capitalize first letter after period (e.g. "hain. aapki" → "hain. Aapki")
    text = re.sub(r'(?<=\.\s)([a-z])', lambda m: m.group(1).upper(), text)
    text = re.sub(r'(?<=[a-z]\s)Your\b', 'your', text)
    _native = getattr(_postprocess, '_native_name', '') or ''
    if _native:
        _first = _native.split()[0]
        text = re.sub(rf'\b{re.escape(_native)}\s+ji\'s\b', 'your', text, flags=re.IGNORECASE)
        text = re.sub(rf'\b{re.escape(_first)}\s+ji\'s\b', 'your', text, flags=re.IGNORECASE)
        text = re.sub(rf'\b{re.escape(_native)}\'s\b', 'your', text, flags=re.IGNORECASE)
        text = re.sub(rf'\b{re.escape(_first)}\'s\b', 'your', text, flags=re.IGNORECASE)
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
        "rule-based system:", "verified methodology:",
        "moderate confidence level", "confidence level: moderate",
        "confidence level: high", "confidence level: low",
        "sub-lord significance:", "planetary positions analysis:",
        "core significators:", "primary significators:",
        "secondary significators:", "key significators:",
        "significator analysis:", "house activation:",
        "dasha activation:", "planetary configuration:",
        "chart analysis:", "kp analysis:",
        "primary period:", "critical antardasha:",
        "peak period:", "current period:",
        "most promising combination:",
        "mahadasha ruler:", "anthardasha ruler:", "antardasha ruler:",
        "underlying mechanism involves", "the cosmic energies align",
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
        "careerevent:", "positiveoutcome:", "specificdates:",
        "bannedphrases:", "eventtype:", "scoringcriteria:",
        "marriageevent:", "financialevent:", "emotionalevent:", "safetyflag:",
        "love vs arranged marriage",
        "career prospects analysis",
        "financial analysis:",
        "peak financial growth period:",
        "timing precision:",
        "most promising combination ye present kar raha hai:",
        "this is a rule-based",
        "rule based system",
        "kp analysis for",
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
    result = re.sub(r'\n{3,}', '\n\n', result)

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

    # ── Phase 12.6: Hard character cap — trim to last sentence within limit ──
    _char_limit = {"simple": 180, "timing": 320, "emotional": 350, "past_event": 400, "remedy": 400}.get(_query_type, 350)
    if len(result) > _char_limit:
        _trimmed = result[:_char_limit]
        _last_end = max(_trimmed.rfind('. '), _trimmed.rfind('! '), _trimmed.rfind('? '),
                        _trimmed.rfind('.'), _trimmed.rfind('!'), _trimmed.rfind('?'))
        if _last_end > _char_limit * 0.5:
            result = result[:_last_end + 1].rstrip()

    # ── Phase 12.5: Strip trailing filler for simple queries ──
    if _query_type == "simple":
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
            # Skip for emotional/safety queries — Hinglish empathy and date sentences are acceptable.
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
                    _hindi_density = (_s_hindi / _s_words) if _s_words > 0 else 0
                    # Keep sentence if: (has a date AND density < 40%), OR density < 25%
                    if (_has_date and _hindi_density < 0.40) or _hindi_density < 0.25:
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
        "frustrated", "scared", "worried", "anxious", "hopeless",
        "everything is going wrong", "why is everything", "mushkil", "pareshani",
        "takleef", "dukh", "tension", "suffering",
        "loser", "looser", "failure", "unlucky", "nothing works",
        "won't do anything", "no hope", "give up", "kuch nahi hoga",
        "feel very unlucky", "feel unlucky", "bad luck", "cursed",
        "health has been troubling", "health troubling",
        "not feeling well", "tabiyat kharab", "bimar",
        "body pain", "sleepless", "insomnia",
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


def _extract_prediction(answer: str) -> Optional[str]:
    """Extract specific date/time predictions from the answer text."""
    # Look for date patterns like "March 2026", "2026-2027", "April 2026 to August 2026"
    date_patterns = [
        r'(?:between|from|during|by|after|before|till|until)\s+\w+\s+\d{4}\s+(?:to|and|till|se|-)\s+\w+\s+\d{4}',
        r'(?:between|from|during|by|after|before|till|until)\s+\w+\s+\d{4}',
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+(?:to|and|till|se|-)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        r'\d{4}-\d{2}\s+(?:to|se|tak)\s+\d{4}-\d{2}',
        r'\d{4}-\d{2}-\d{2}\s+(?:to|se|tak)\s+\d{4}-\d{2}-\d{2}',
        r'\d{4}\s*(?:to|se|-)\s*\d{4}',
    ]
    predictions = []
    for pattern in date_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        predictions.extend(matches)
    if predictions:
        # Return the most specific prediction (longest match)
        return max(predictions, key=len).strip()
    return None


def _generate_response(question: str, chart_data: str = "", history: list = None):
    """Generate a complete (non-streaming) response and return structured output."""
    req_id = uuid.uuid4().hex[:12]
    t_start = time.monotonic()

    # Convert chart JSON to compact YAML (5500 lines JSON → ~120 lines YAML)
    chart_yaml = _chart_to_yaml(chart_data or "")

    # Hard guard: no chart data + personal prediction question = ask for chart
    if not chart_yaml:
        personal_keywords = [
            "when will", "will i", "my marriage", "my career", "my financial",
            "my health", "my job", "should i", "am i", "will my", "my kundali",
            "meri shaadi", "mera career", "when did", "kab hogi", "obstacles",
            "get married", "change fields", "improve", "facing", "confused",
        ]
        msg_lower = question.lower()
        if any(kw in msg_lower for kw in personal_keywords):
            return {
                "answer": ("Aapka chart data abhi load nahi hai. Please apni birth chart (JSON) "
                           "send karein — tabhi main aapko accurate prediction de paunga. "
                           "Bina chart ke prediction dena galat hoga."),
                "prediction": None,
                "product_reco": None,
            }

    if chart_yaml:
        full_question = (f"Chart context (YAML):\n{chart_yaml}\n\n"
                         f"Question: {question}")
    else:
        full_question = question

    rag_chunks = _retrieve_rag_chunks(question, top_k=args.top_k)

    # Classify query type for intelligent response control
    query_info = _classify_query_type(question)
    is_remedy = _is_remedy_query(question)

    # Safety intercept — death/longevity queries get compassionate redirect
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
        return {
            "answer": safety_msg,
            "prediction": None,
            "product_reco": None,
        }

    # Inappropriate intercept — sexual orientation, personal judgments
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
        return {
            "answer": inappropriate_msg,
            "prediction": None,
            "product_reco": None,
        }

    # Direct factual intercepts — bypass model for questions we can answer perfectly
    q_lower = question.lower().strip()

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
            ans = f"{_intercept_name_ji}, aaj ki date {today} hai."
        else:
            ans = f"{_intercept_name_ji}, today's date is {today}."
        return {"answer": ans, "prediction": None, "product_reco": None}

    if any(p in q_lower for p in ["who are you", "what is your name", "your name", "tell me about yourself"]):
        return {
            "answer": ("My name is Jyotish — I am a seasoned KP astrologer. I use Krishnamurti Paddhati principles "
                       "to give you accurate and practical answers. You can ask me about career, finance, health, "
                       "relationships, and life timing."),
            "prediction": None, "product_reco": None,
        }
    if any(p in q_lower for p in ["aapka naam", "aap kaun", "tum kaun", "kaun ho aap", "kaun ho tum"]):
        return {
            "answer": ("Main Jyotish hun — ek seasoned KP astrologer. Main Krishnamurti Paddhati ke principles "
                       "use karke aapke sawaalon ka accurate aur practical jawaab deta hun. Aap mujhse career, "
                       "finance, health, relationships, aur life timing ke baare mein pooch sakte hain."),
            "prediction": None, "product_reco": None,
        }

    # "What is my name?" — language-aware
    if any(p in q_lower for p in ["what is my name", "what's my name", "tell me my name"]):
        if _intercept_name:
            return {"answer": f"{_intercept_name_ji}, your name is {_intercept_name}.", "prediction": None, "product_reco": None}
        else:
            return {"answer": "I don't have your chart data loaded yet. Please provide your birth chart JSON.", "prediction": None, "product_reco": None}
    if any(p in q_lower for p in ["mera naam kya", "mera naam bata", "mera name kya"]):
        if _intercept_name:
            return {"answer": f"{_intercept_name_ji}, aapka naam {_intercept_name} hai.", "prediction": None, "product_reco": None}
        else:
            return {"answer": "Aapka chart data abhi load nahi hai. Please apni birth chart JSON provide karein.", "prediction": None, "product_reco": None}

    # Non-astrology conversation intercepts — greetings, feedback, meta-questions
    _greeting_patterns = ["good morning", "good afternoon", "good evening", "good night",
                          "have a good day", "have a nice day", "bye", "goodbye", "thank you",
                          "thanks", "shukriya", "dhanyavaad", "alvida", "namaste",
                          "hello", "hi there", "hey there"]
    if any(p in q_lower for p in _greeting_patterns) and len(q_lower.split()) <= 8:
        return {"answer": f"{_intercept_name_ji}, thank you! Jab bhi aapko astrology guidance chahiye, main yahan hun. Have a wonderful day! 🙏",
                "prediction": None, "product_reco": None}

    _feedback_patterns = ["you need to improve", "you are wrong", "you're wrong", "that's wrong",
                          "not correct", "galat hai", "improve karo", "better karo",
                          "your answer is wrong", "postprocessing", "overriding",
                          "hmm yeah", "hmm ok", "hmm okay"]
    if any(p in q_lower for p in _feedback_patterns):
        return {"answer": f"{_intercept_name_ji}, I appreciate your feedback — I am continuously learning and improving. Please ask me any astrology question and I will do my best to give you an accurate answer based on your chart.",
                "prediction": None, "product_reco": None}

    _meta_patterns = ["how many years", "kitne saal", "experience", "how old are you",
                      "when were you made", "who made you", "who created you",
                      "can someone predict", "do you believe", "is astrology real",
                      "is astrology true", "kya astrology sach", "kya bhavishya"]
    if any(p in q_lower for p in _meta_patterns) and not any(w in q_lower for w in ["job", "marriage", "career", "financial", "health", "shaadi", "naukri"]):
        if _is_hindi_q(q_lower):
            ans = (f"{_intercept_name_ji}, main Jyotish hun — ek experienced KP astrologer. "
                   "Main Krishnamurti Paddhati ke principles se aapke sawaalon ka jawaab deta hun. "
                   "Aap mujhse apni kundali ke baare mein kuch bhi pooch sakte hain.")
        else:
            ans = (f"{_intercept_name_ji}, I am Jyotish — an experienced KP astrologer. "
                   "I use Krishnamurti Paddhati principles to analyze your chart and provide accurate predictions. "
                   "Feel free to ask me anything about your kundali.")
        return {"answer": ans, "prediction": None, "product_reco": None}

    # Product recommendations — ONLY when user asks for remedies
    product_list, product_prompt_text = [], ""
    if is_remedy:
        product_list, product_prompt_text = _get_product_recommendations(question, chart_summary=chart_yaml)
    product_instruction = ""
    if product_prompt_text:
        product_instruction = (
            f"\n\nRELEVANT PRODUCTS — weave ONE naturally as a remedy suggestion:\n"
            f"{product_prompt_text}\n"
            f"Example: 'Is samay [planet] ko strengthen karne ke liye hamara [Product Name] try karein.'"
        )

    # Build prompt with adaptive RAG trimming
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

    # Build messages WITH conversation history for follow-up context
    messages = [
        {"role": "system", "content": sys_content},
    ]

    # Include recent conversation history (last N turns, budget-aware)
    MAX_HISTORY_TURNS = 4
    history_chars = 0
    history_budget = MAX_INPUT_CHARS // 4  # reserve 25% of input budget for history
    if history:
        recent = history[-MAX_HISTORY_TURNS:]
        for turn in recent:
            user_msg = turn.get("user", "") if isinstance(turn, dict) else (turn[0] if len(turn) > 0 else "")
            bot_msg = turn.get("assistant", "") if isinstance(turn, dict) else (turn[1] if len(turn) > 1 else "")
            if not user_msg:
                continue
            turn_chars = len(user_msg or '') + len(bot_msg or '')
            if history_chars + turn_chars > history_budget:
                break
            messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
            history_chars += turn_chars

    # Current question (with chart YAML context)
    messages.append({"role": "user", "content": full_question})

    # Compute output tokens — use query-type-aware limits
    total_chars = sum(len(m["content"]) for m in messages)
    est_input_tokens = int(total_chars / 0.78) + 100
    available = MAX_MODEL_LEN - est_input_tokens
    base_output = query_info.get("max_tokens_override") or OUTPUT_TOKENS
    max_tokens = max(64, min(base_output, available))
    temperature = query_info["temperature"]

    if max_tokens < 64:
        raise HTTPException(status_code=400, detail="Input too long for model context window.")

    # Extract birth year and native name from chart for postprocess (before generation for retry)
    _postprocess._birth_year = None
    _postprocess._native_name = None
    _postprocess._query_type = query_info["type"]
    _postprocess._user_question = question
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

    response = llm_client.chat.completions.create(
        model="kp-astrology-llama",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        stream=False,
        extra_body={"repetition_penalty": 1.2},
    )

    raw_answer = response.choices[0].message.content or ""

    # ── Deflection retry: if model gave a vague non-answer, retry with forced prefix ──
    if raw_answer and _is_deflection(raw_answer) and chart_yaml:
        _json_log("deflection_detected", req_id=req_id, original=raw_answer[:200])
        native_name = getattr(_postprocess, '_native_name', '') or ''
        name_ji = f"{native_name} ji" if native_name else "Ji"

        # Build query-type-specific example
        _q_topic = question.lower()
        if any(w in _q_topic for w in ["health", "body", "illness", "sick", "bimari", "tabiyat"]):
            _example = (f"'{name_ji}, your health needs attention from now till Apr 2026 during Saturn-Ketu AD "
                        f"which connects to houses 6,8. After May 2026, Saturn-Venus AD brings recovery through houses 1,11.'")
        elif any(w in _q_topic for w in ["career", "job", "naukri", "kaam", "field", "profession"]):
            _example = (f"'{name_ji}, your career breakthrough comes Oct 2026 to Mar 2027 during Saturn-Venus AD "
                        f"which activates houses 2,6,10,11. 10th cusp sub-lord Mercury signifies houses 2,10.'")
        elif any(w in _q_topic for w in ["exam", "interview", "test", "pariksha", "result"]):
            _example = (f"'{name_ji}, your success window is Mar 2026 to Aug 2026 during Saturn-Ketu AD. "
                        f"5th cusp sub-lord Jupiter signifies houses 4,9,11 — strong for academic success.'")
        elif any(w in _q_topic for w in ["financial", "money", "paisa", "dhan", "income", "wealth"]):
            _example = (f"'{name_ji}, your finances improve from Apr 2027 during Saturn-Venus AD which activates houses 2,6,11. "
                        f"2nd cusp sub-lord Mars signifies houses 2,11 — wealth and gains.'")
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
        retry_resp = llm_client.chat.completions.create(
            model="kp-astrology-llama",
            messages=retry_msgs,
            max_tokens=max_tokens,
            temperature=0.15,
            top_p=0.85,
            stream=False,
            extra_body={"repetition_penalty": 1.15},
        )
        retry_text = retry_resp.choices[0].message.content or ""
        if retry_text and not _is_deflection(retry_text):
            raw_answer = retry_text
        else:
            # Second retry with even more forceful prompt
            retry_user2 = (
                f"Chart YAML:\n{chart_yaml[:3000]}\n\n"
                f"Question: {question}\n\n"
                f"ANSWER IN EXACTLY THIS FORMAT — fill in the blanks from the dasha table:\n"
                f"{name_ji}, [answer to question] timing is [read Month Year from antarDashas] "
                f"to [read end Month Year]. [One sentence about which cusp/house supports this].\n"
                f"DO NOT explain methodology. DO NOT say 'analysis'. Just give the dates and answer."
            )
            retry_msgs2 = [
                {"role": "system", "content": _sys_no_rag},
                {"role": "user", "content": retry_user2},
            ]
            retry_resp2 = llm_client.chat.completions.create(
                model="kp-astrology-llama",
                messages=retry_msgs2,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.85,
                stream=False,
                extra_body={"repetition_penalty": 1.15},
            )
            retry_text2 = retry_resp2.choices[0].message.content or ""
            if retry_text2 and not _is_deflection(retry_text2):
                raw_answer = retry_text2
            else:
                _json_log("deflection_retry_failed", req_id=req_id)

    answer = _postprocess(raw_answer)

    # Enrich: append Hindi quote + product (only if remedy query)
    answer = _enrich_response(answer, product_text=product_prompt_text, is_remedy=is_remedy, query_type=query_info["type"])

    # Extract prediction
    prediction = _extract_prediction(answer)

    # Pick best product recommendation — only on remedy queries
    product_reco = product_list[0] if (is_remedy and product_list) else None

    # ── Structured log for observability ──
    _json_log("chat_response",
              req_id=req_id,
              query_type=query_info["type"],
              is_remedy=is_remedy,
              rag_chunks=len(selected_chunks),
              max_tokens=max_tokens,
              temperature=temperature,
              raw_len=len(raw_answer),
              answer_len=len(answer),
              has_prediction=prediction is not None,
              has_product=product_reco is not None,
              latency_ms=round((time.monotonic() - t_start) * 1000))

    return {
        "answer": answer,
        "prediction": prediction,
        "product_reco": product_reco,
    }


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="KP Astrology AI API",
    description="REST API for KP Astrology predictions with structured JSON output",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryTurn(BaseModel):
    user: str
    assistant: Optional[str] = None


class ChatRequest(BaseModel):
    question: str
    chart_data: Optional[str] = None
    history: Optional[List[HistoryTurn]] = None


class ProductReco(BaseModel):
    sku: str
    title: str
    price: str


class ChatResponse(BaseModel):
    answer: str
    prediction: Optional[str] = None
    product_reco: Optional[ProductReco] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_enabled": rag_index is not None,
        "products_loaded": len(PRODUCT_CATALOG),
        "max_model_len": MAX_MODEL_LEN,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main endpoint. Send a question + optional chart_data JSON.

    Returns:
    - answer: conversational astrology response
    - prediction: specific date/time prediction extracted (if any)
    - product_reco: recommended product {sku, title, price} (if any)
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        hist = [h.model_dump() for h in req.history] if req.history else None
        result = _generate_response(req.question, req.chart_data or "", history=hist)
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")


# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  KP Astrology REST API")
    print(f"  Endpoint: http://{args.host}:{args.port}/chat")
    print(f"  Docs:     http://{args.host}:{args.port}/docs")
    print(f"  vLLM:     {args.vllm_url}")
    print(f"{'='*60}\n")
    uvicorn.run(app, host=args.host, port=args.port)
