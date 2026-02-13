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
from typing import Optional
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

# ── System prompts ───────────────────────────────────────────────────────────
SYSTEM_BASE = (
    "You are Jyotish, a KP astrologer. You give DIRECT answers with SPECIFIC dates from the chart.\n\n"
    "## HARD RULES — violating ANY rule is a critical failure:\n"
    "- NEVER say 'I can analyze', 'requires careful analysis', 'we need to examine', 'let me check'. GIVE THE ANSWER DIRECTLY.\n"
    "- NEVER output '[Name]' literally. Read the name from YAML and use it.\n"
    "- NEVER write headers like 'Career Analysis:', 'Marriage Prediction:', 'Remedy Based on:'. Just answer.\n"
    "- NEVER use markdown: no **bold**, no bullets, no numbered lists, no headers.\n"
    "- NEVER say 'the native', 'the querent', 'the person'. Say 'you' or use their name.\n"
    "- NEVER exceed 4 sentences. Simple questions = 1 sentence only.\n"
    "- NEVER output years before the person's birth year (read dob from YAML).\n"
    "- ALWAYS cite the specific cusp sub-lord and house numbers when making predictions.\n"
    "- ALWAYS give specific month-year ranges from the dasha table, not vague statements.\n"
    "- Read today_date from YAML. Past dates = past tense. Future dates = future tense.\n\n"
    "## OUTPUT FORMAT — follow EXACTLY:\n"
    "SIMPLE (name/lagna/rashi/who are you): One sentence. Read value from YAML. Done.\n"
    "TIMING (when will X happen): '[Name] ji, [event] ka strong period [Planet]-[Planet] AD mein hai (Mon YYYY - Mon YYYY), "
    "peak [Mon-Mon YYYY] jab [pratyantar planet] houses [X,Y,Z] activate karega — tab aap [age] ke honge. "
    "[Cusp] sub-lord [Planet] houses [X,Y,Z] signify karta hai jo [event] ke liye positive hai.'\n"
    "PAST EVENT (what happened in year X): Match the dasha running in that period to house significations. "
    "State what likely happened and why, citing the specific AD/PD and houses.\n"
    "REMEDY: Give the astrological remedy first, then mention ONE product if available.\n\n"
    "## FEW-SHOT EXAMPLES (follow this style EXACTLY):\n"
    "Q: 'What is my name?' → 'Aditya Raj ji, aapka naam Aditya Raj hai.'\n"
    "Q: 'What is my lagna?' → 'Aditya Raj ji, aapka lagna Sagittarius hai.'\n"
    "Q: 'Who are you?' → 'Mera naam Jyotish hai, main KP astrology se aapke sawaalon ka jawaab deta hun.'\n"
    "Q: 'When will I get married?' → 'Aditya Raj ji, shaadi ka strong period Rahu-Venus AD mein hai "
    "(Jun 2028 - Sep 2030), peak Oct 2028 - Mar 2029 jab Mercury pratyantar houses 2,7,11 activate karega "
    "— tab aap 25 ke honge. 7th cusp sub-lord Venus houses 2,7,11 signify karta hai jo marriage ke liye positive hai.'\n"
    "Q: 'What happened in my career from 2020 to 2025?' → 'Aditya Raj ji, 2020-2023 mein Mars MD tha "
    "jismein Mars houses 6,10 signify karta hai — yeh service/job period tha. Dec 2023 se Rahu MD shuru hua "
    "jo houses 10,11 activate karta hai, isse career mein growth aur new opportunities aaye.'\n"
    "Q: 'Suggest a remedy for career' → 'Aditya Raj ji, career ke liye 10th cusp sub-lord Saturn ko "
    "strengthen karna chahiye — Saturday ko neela vastra pehnein aur Shani mantra ka jaap karein.'\n\n"
    "## LANGUAGE: Default English. If user writes Hindi/Hinglish, match their language.\n"
    "## PRODUCTS: Only if RELEVANT PRODUCTS section exists below. Otherwise NEVER mention products.\n"
)

SYSTEM_NO_RAG = (
    "You are Jyotish, a KP astrologer. You give DIRECT answers with SPECIFIC dates from the chart.\n\n"
    "HARD RULES:\n"
    "- NEVER deflect: no 'I can analyze', 'requires analysis', 'we need to examine'. ANSWER DIRECTLY.\n"
    "- NEVER output '[Name]'. Read name from YAML. Address as '[name] ji'.\n"
    "- NEVER write headers, markdown, bold, bullets, numbered lists.\n"
    "- NEVER say 'the native'. Say 'you' or use their name.\n"
    "- Simple questions = 1 sentence. Timing = 2-3 sentences. MAX 4 sentences.\n"
    "- ALWAYS cite cusp sub-lord + house numbers. ALWAYS give month-year ranges from dasha table.\n"
    "- Read today_date from YAML for correct tense. No years before birth year.\n\n"
    "FORMAT:\n"
    "SIMPLE: One sentence reading value from YAML.\n"
    "TIMING: '[Name] ji, [event] ka period [AD range Mon YYYY-Mon YYYY], peak [months] jab [planet] houses [X,Y] activate karega — aap [age] ke honge. [Cusp] sub-lord [planet] houses [X,Y] signify karta hai.'\n"
    "PAST: Match dasha to houses, state what happened and why.\n"
    "REMEDY: Astrological remedy + ONE product if available.\n\n"
    "EXAMPLES:\n"
    "Q: 'What is my name?' → 'Aditya Raj ji, aapka naam Aditya Raj hai.'\n"
    "Q: 'When will I get married?' → 'Aditya Raj ji, shaadi ka strong period Rahu-Venus AD (Jun 2028-Sep 2030), peak Oct 2028-Mar 2029 jab Mercury pratyantar houses 2,7,11 activate karega — tab aap 25 ke honge. 7th cusp sub-lord Venus houses 2,7,11 signify karta hai.'\n"
    "Q: 'Who are you?' → 'Mera naam Jyotish hai, main KP astrology se aapke sawaalon ka jawaab deta hun.'\n"
    "Language: English default. Match Hindi/Hinglish if user uses it.\n"
    "Products: Only if RELEVANT PRODUCTS section exists. Otherwise NONE.\n"
)

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
        print(f"RAG error: {e}")
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
            print(f"  Product Pinecone search error: {e}")

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
    _dangerous_terms = [
        r'cancer[- ]?related', r'\bcancer\b', r'tumor', r'malignant', r'benign',
        r'heart\s+attack', r'heart\s+disease', r'cardiac\s+arrest',
        r'\bstroke\b', r'\bdiabetes\b', r'\bHIV\b', r'\bAIDS\b',
        r'tuberculosis', r'epilepsy', r'paralysis',
        r'kidney[- ]?(?:related|failure|disease|issues|problems)',
        r'liver[- ]?(?:related|failure|disease|issues|problems)',
        r'brain\s+(?:damage|tumor|cancer)',
        r'mental\s+(?:illness|disorder|disease)', r'schizophren',
        r'bipolar', r'suicid', r'\bdeath\b', r'\bfatal\b', r'\bterminal\b',
        r'life[- ]?threatening', r'\blethal\b',
        r'immediate\s+(?:medical\s+)?attention',
        r'require[sd]?\s+(?:immediate|urgent)\s+(?:medical\s+)?(?:attention|treatment)',
        r'serious\s+(?:disease|illness|condition|complication)',
        r'chronic\s+(?:disease|illness|condition)',
    ]
    for term in _dangerous_terms:
        text = re.sub(term, 'health challenges', text, flags=re.IGNORECASE)

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
        r'(?:Marriage|Career|Financial|Health|Remedy|Obstacle|Education|Relationship)\s+(?:Prediction|Breakthrough|Timing|Gains)?\s*(?:Analysis|Prediction|Report)?(?:\s+(?:using|Based|by|for|of)\s+[^\n]{0,60})?\s*$',
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

    # ── Phase 7: Replace robotic third-person references ──
    _replacements = [
        (r'\bThe\s+native\s+has\b', 'You have'),
        (r'\bThe\s+native\s+is\b', 'You are'),
        (r'\bThe\s+native(?:\'s)?\b', 'Your'),
        (r'\bthe\s+native(?:\'s)?\b', 'your'),
        (r'\bThe\s+querent\b', 'You'),
        (r'\bthe\s+querent\b', 'you'),
        (r'\bThe\s+person\b', 'You'),
        (r'\bthe\s+person\b', 'you'),
        (r'\bIt\s+is\s+(?:observed|noted|seen)\s+that\b', ''),
        (r'\bIt\s+(?:can\s+be|is)\s+(?:concluded|inferred)\s+that\b', ''),
        (r'\bIn\s+conclusion,?\b', ''),
        (r'\bTo\s+summarize,?\b', ''),
    ]
    for pat, repl in _replacements:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)

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
    # ── Phase 8.5: Strip model-generated Hindi quotes on non-remedy queries ──
    _query_type = getattr(_postprocess, '_query_type', 'analysis')
    if _query_type != "remedy":
        _quote_patterns = [
            r'(?:^|\n\n?)\s*(?:Jab samay|Andhera jitna|Sabr ka phal|Jab tak todenge|Graho ki chaal|'
            r'Mushkilein waqti|Waqt sabka|Har raat ke baad|Kismat likhne|Jab niyat)[^\n]{0,120}\.?\s*$',
            r'(?:^|\n\n?)\s*"[^"]{10,120}"\s*$',  # quoted Hindi sentences
        ]
        for qp in _quote_patterns:
            result = re.sub(qp, '', "\n".join(cleaned), flags=re.MULTILINE)
            cleaned = [l for l in result.split("\n")]

    result = "\n".join(cleaned).rstrip()

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

    # ── Phase 12: Hard sentence cap based on query type ──
    sentences = re.split(r'(?<=[.!?])\s+', result.strip())
    if _query_type == "simple" and len(sentences) > 2:
        result = ' '.join(sentences[:2])
    elif _query_type == "timing" and len(sentences) > 4:
        result = ' '.join(sentences[:4])
    elif len(sentences) > 6:
        result = ' '.join(sentences[:6])

    # ── Phase 12.5: Strip trailing filler for simple queries ──
    if _query_type == "simple":
        _filler_starters = [
            "ye planetary", "yeh planetary", "this creates", "this is",
            "in kp astrology", "kp system mein", "ye combination",
            "yeh combination", "ye positions", "yeh positions",
            "this planetary", "these planetary",
        ]
        sents = re.split(r'(?<=[.!?])\s+', result.strip())
        kept = []
        for s in sents:
            if any(s.strip().lower().startswith(f) for f in _filler_starters):
                break
            kept.append(s)
        if kept:
            result = ' '.join(kept)

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
    Returns dict with: type, max_paragraphs, temperature, max_tokens_override."""
    q = question.lower().strip()

    # Simple factual — 1-2 sentences, low temperature
    simple_patterns = [
        "what is my name", "mera naam", "my name", "naam kya hai",
        "what is my dob", "date of birth", "birth date", "janam din",
        "what is my lagna", "lagna kya hai", "ascendant",
        "what is my rashi", "rashi kya hai", "moon sign",
        "what is my nakshatra", "nakshatra kya hai",
        "where was i born", "birth place", "kahan paida",
        "who are you", "what can you do", "tell me about yourself",
    ]
    if any(p in q for p in simple_patterns):
        return {"type": "simple", "max_paragraphs": 1, "temperature": 0.3, "max_tokens_override": 150}

    # Past event / year-by-year — needs past dasha data, higher token budget
    past_patterns = [
        "what happened", "year by year", "year-by-year", "from 20",
        "between 20", "in 2020", "in 2021", "in 2022", "in 2023", "in 2024", "in 2025",
        "when did i", "kab hua", "kab hui", "past ", "pichle",
        "graduation", "first job", "first relationship", "childbirth",
        "health issue", "what year did",
    ]
    if any(p in q for p in past_patterns):
        return {"type": "past_event", "max_paragraphs": 3, "temperature": 0.4, "max_tokens_override": 600}

    # Timing questions — 2-3 sentences, moderate temperature
    timing_patterns = [
        "when will", "kab hogi", "kab milegi", "kab hoga",
        "timing", "which year", "which month", "best period",
        "favorable time", "auspicious time", "shubh samay",
    ]
    if any(p in q for p in timing_patterns):
        return {"type": "timing", "max_paragraphs": 2, "temperature": 0.5, "max_tokens_override": 450}

    # Remedy queries — need product, moderate length
    if _is_remedy_query(question):
        return {"type": "remedy", "max_paragraphs": 3, "temperature": 0.5, "max_tokens_override": 500}

    # Complex analysis — full response
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


def _generate_response(question: str, chart_data: str = ""):
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
    fixed_chars = len(SYSTEM_BASE) + len(full_question) + len(product_instruction) + 30
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
        sys_content = f"{SYSTEM_BASE}\n\nKP Book Excerpts:\n{rag_text}{product_instruction}"
    else:
        sys_content = f"{SYSTEM_NO_RAG}{product_instruction}"

    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": full_question},
    ]

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
        ]
        if any(p in t for p in deflection_phrases):
            return True
        # For timing questions: MUST have actual dates (month-year or year range)
        # Just mentioning "cusp" or "house" without dates is still deflection
        if query_info["type"] in ("timing", "past_event"):
            has_date = bool(re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}', text))
            has_year_range = bool(re.search(r'20\d{2}\s*(?:to|se|tak|-)\s*20\d{2}', text))
            if not has_date and not has_year_range:
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
        retry_user = (
            f"{full_question}\n\n"
            f"IMPORTANT: You MUST give a SPECIFIC answer with dates from the dasha table. "
            f"Start your answer with '{name_ji},' and include specific month-year ranges. "
            f"Do NOT say 'depends on' or 'requires analysis'. Read the YAML and answer NOW."
        )
        retry_msgs = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": retry_user},
        ]
        retry_resp = llm_client.chat.completions.create(
            model="kp-astrology-llama",
            messages=retry_msgs,
            max_tokens=max_tokens,
            temperature=max(0.3, temperature - 0.2),
            top_p=0.85,
            stream=False,
            extra_body={"repetition_penalty": 1.15},
        )
        retry_text = retry_resp.choices[0].message.content or ""
        if retry_text and not _is_deflection(retry_text):
            raw_answer = retry_text
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


class ChatRequest(BaseModel):
    question: str
    chart_data: Optional[str] = None


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
        result = _generate_response(req.question, req.chart_data or "")
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
