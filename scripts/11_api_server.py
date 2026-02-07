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
import os
import re
import csv
import random
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn

load_dotenv()

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
parser.add_argument("--max-model-len", type=int, default=2048,
                    help="vLLM max model length")
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

# ── Product catalog ──────────────────────────────────────────────────────────
PRODUCT_CATALOG = []
_products_path = args.products_csv
if not _products_path:
    import glob
    _search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
        "/workspace",
    ]
    for d in _search_dirs:
        found = glob.glob(os.path.join(d, "products_export*.csv"))
        if found:
            _products_path = found[0]
            print(f"  Products: auto-discovered {_products_path}")
            break
if _products_path and os.path.isfile(_products_path):
    try:
        with open(_products_path, encoding="utf-8") as f:
            PRODUCT_CATALOG = list(csv.DictReader(f))
        print(f"  Products: {len(PRODUCT_CATALOG)} items loaded from {_products_path}")
    except Exception as e:
        print(f"  Products: FAILED ({e})")
else:
    print("  Products: NONE (no CSV found — pass --products-csv or place products_export*.csv nearby)")

# ── System prompts ───────────────────────────────────────────────────────────
SYSTEM_BASE = (
    "You are a warm, experienced KP astrologer speaking directly to the person sitting in front of you. "
    "Talk like a real astrologer — conversational, confident, compassionate. Use Hinglish naturally.\n\n"
    "MANDATORY OUTPUT FORMAT (violating ANY rule = failure):\n"
    "1. Write EXACTLY 3-4 short paragraphs. No bullet points. No numbered lists. No bold text. No headers.\n"
    "2. NEVER use **bold**, headers, labels, or section titles. No **Remedy:**, **Timing:**, **Analysis:**, "
    "**Digestive System:**, **Immune System:**, **Conclusion:**, **Confidence:**. ZERO markdown formatting.\n"
    "3. NEVER write 'Confidence: high/medium/low' anywhere.\n"
    "4. MUST include a specific Hindi/Hinglish motivational quote naturally in your response "
    "(e.g., 'Jab samay aayega, rishta khud chalkar aayega' or 'Andhera jitna gehra ho, subah utni roshan hoti hai').\n"
    "5. MUST mention specific months and years from the dasha/antardasha dates in the chart. "
    "Read the MAHADASHA and ANTARDASHA dates and quote them. Example: 'Venus bhukti from 2005-07 to 2006-09 is key.'\n"
    "6. If RELEVANT PRODUCTS are listed below, MUST weave exactly ONE product naturally into a sentence. "
    "Example: 'Venus ko strengthen karne ke liye hamara Shukra Kavach Pendant try karein.'\n\n"
    "CONTENT RULES:\n"
    "- Talk TO the person: 'Aapke 7th house mein...', 'Aapko...', NOT 'The native has...'\n"
    "- Read the Pre-extracted Chart Summary and use exact cusp sub-lords, planet significations, dasha dates.\n"
    "- Use KP Book Excerpts for rules. Reference [rule_id] briefly if needed.\n"
    "- NEVER invent page numbers or chapter numbers.\n\n"
    "EXAMPLE of ideal response (follow this style exactly):\n"
    "Aapke 7th house ka sub-lord Saturn hai jo houses 1,2,3,4,7,9 signify karta hai — yeh marriage ke liye "
    "positive indication hai. Venus bhukti 2005-07 se 2006-09 tak chal raha hai aur Venus directly 7th aur 11th "
    "house signify karta hai, toh yeh period marriage ke liye sabse strong window hai. Jab samay aayega, rishta "
    "khud chalkar aayega. Venus ko aur strengthen karne ke liye hamara Shukra Kavach Pendant try karein.\n"
)

SYSTEM_NO_RAG = (
    "You are a warm, experienced KP astrologer speaking directly to the person. "
    "Talk conversationally in Hinglish. Give SPECIFIC dates/months/years from dasha data.\n\n"
    "MANDATORY RULES:\n"
    "1. Write EXACTLY 3-4 short paragraphs. No bold text, no headers, no bullet points, no markdown.\n"
    "2. MUST include specific months/years from the antardasha dates in the chart summary.\n"
    "3. MUST include one Hindi motivational quote naturally woven in.\n"
    "4. If products are listed, MUST mention one product naturally as a remedy suggestion.\n"
    "5. Talk TO the person. NEVER write 'Confidence:' or use **bold** formatting.\n"
    "6. NEVER invent page numbers. If no chart data, ask them to share their birth chart.\n"
)

# ── Context-window budget ────────────────────────────────────────────────────
MAX_MODEL_LEN = args.max_model_len
OUTPUT_TOKENS = 400 if MAX_MODEL_LEN <= 4096 else min(768, MAX_MODEL_LEN // 6)
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

    # ── Method 2: CSV keyword fallback ──
    if not PRODUCT_CATALOG:
        return [], ""
    search_text = (question + " " + chart_summary).lower()
    planet_product_map = {
        "venus": ["diamond", "opal", "white", "zircon", "shukra", "venus"],
        "saturn": ["blue sapphire", "neelam", "karungali", "iron", "shani", "saturn"],
        "jupiter": ["yellow sapphire", "pukhraj", "topaz", "rudraksha", "guru", "jupiter"],
        "mars": ["coral", "moonga", "red", "mangal", "mars", "hanuman"],
        "mercury": ["emerald", "panna", "green", "budh", "mercury"],
        "moon": ["pearl", "moti", "chandra", "moon"],
        "sun": ["ruby", "manik", "surya", "sun"],
        "rahu": ["hessonite", "gomed", "garnet", "rahu"],
        "ketu": ["cat eye", "lehsunia", "vaidurya", "ketu"],
    }
    topic_product_map = {
        "marriage": ["venus", "shukra", "diamond", "opal", "love"],
        "career": ["ruby", "manik", "surya", "sun"],
        "financial": ["yellow sapphire", "pukhraj", "lakshmi"],
        "health": ["rudraksha", "healing", "chakra"],
        "obstacle": ["karungali", "shani", "protection", "kavach"],
        "luck": ["rudraksha", "navratna", "kavach", "sudarshan"],
        "protect": ["kavach", "evil eye", "protection", "karungali"],
    }
    keywords = set()
    for planet, terms in planet_product_map.items():
        if planet in search_text:
            keywords.update(terms)
    for topic, terms in topic_product_map.items():
        if topic in search_text:
            keywords.update(terms)
    if not keywords:
        keywords = {"rudraksha", "kavach", "chakra"}
    matches = []
    for p in PRODUCT_CATALOG:
        title_lower = p.get("Title", "").lower()
        if any(kw in title_lower for kw in keywords):
            matches.append(p)
    if not matches:
        return [], ""
    matches = matches[:max_items]
    product_list = [
        {"sku": p.get("SKU", ""), "title": p.get("Title", ""), "price": p.get("Sale Price", "")}
        for p in matches
    ]
    prompt_lines = []
    for p in matches:
        prompt_lines.append(f"- {p['Title']} (SKU: {p.get('SKU','')}, Rs.{p.get('Sale Price','')})")
    return product_list, "\n".join(prompt_lines)


MAX_CHART_CHARS = 8000


def _compact_chart_data(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        if len(raw) > MAX_CHART_CHARS:
            return raw[:MAX_CHART_CHARS] + "\n[...chart data truncated]"
        return raw
    slim = {}
    for key in ("name", "gender"):
        if key in d:
            slim[key] = d[key]
    if "birthDetails" in d:
        slim["birthDetails"] = d["birthDetails"]
    if "planetKP" in d:
        slim["planetKP"] = {}
        for planet, pdata in d["planetKP"].items():
            slim["planetKP"][planet] = {k: v for k, v in pdata.items() if k != "subSub"}
    if "cuspKP" in d:
        slim["cuspKP"] = {}
        for cusp, cdata in d["cuspKP"].items():
            slim["cuspKP"][cusp] = {k: v for k, v in cdata.items() if k != "subSub"}
    for key in ("significators", "planetSignifications"):
        if key in d:
            slim[key] = d[key]
    if "dashas" in d:
        slim["dashas"] = {}
        if "dashaBalance" in d["dashas"]:
            slim["dashas"]["dashaBalance"] = d["dashas"]["dashaBalance"]
        if "dashas" in d["dashas"]:
            slim["dashas"]["mahadashas"] = [
                {"lord": dd.get("lord"), "startDate": dd.get("startDate", "")[:10],
                 "endDate": dd.get("endDate", "")[:10], "period": dd.get("period")}
                for dd in d["dashas"]["dashas"]
            ]
    result = json.dumps(slim, indent=1, ensure_ascii=False)
    if len(result) > MAX_CHART_CHARS:
        result = result[:MAX_CHART_CHARS] + "\n...}"
    return result


def _chart_summary(raw: str) -> str:
    try:
        d = json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return ""
    lines = []
    name = d.get("name", "Unknown")
    bd = d.get("birthDetails", {})
    lines.append(f"Native: {name}, DOB: {bd.get('date','?')}, TOB: {bd.get('time','?')}, "
                 f"Lagna: {bd.get('lagna','?')} ({bd.get('lagnaLord','?')})")
    ckp = d.get("cuspKP", {})
    if ckp:
        lines.append("KEY CUSP SUB-LORDS:")
        for c in ["1", "2", "6", "7", "10", "11", "12"]:
            cp = ckp.get(c, {})
            if cp:
                lines.append(f"  Cusp {c}: sub={cp.get('sub','?')}, "
                             f"nak={cp.get('nakshatra','?')}({cp.get('nakshatraLord','?')}), "
                             f"rashi={cp.get('rashi','?')}, degree={cp.get('degree','?')}")
    psig = d.get("planetSignifications", {})
    if psig:
        lines.append("PLANET SIGNIFICATIONS (houses each planet signifies):")
        for planet in ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Rahu", "Ketu"]:
            houses = psig.get(planet, [])
            if houses:
                lines.append(f"  {planet}: houses {houses}")
    dashas = d.get("dashas", {})
    db = dashas.get("dashaBalance", {})
    if db:
        lines.append(f"DASHA BALANCE: {db.get('lord','?')} "
                     f"{db.get('years',0)}Y {db.get('months',0)}M {db.get('days',0)}D remaining")
    dlist = dashas.get("dashas", dashas.get("mahadashas", []))
    if dlist:
        lines.append("MAHADASHA PERIODS:")
        for dd in dlist[:5]:
            lines.append(f"  {dd.get('lord','?')}: {dd.get('startDate','?')[:10]} to "
                         f"{dd.get('endDate','?')[:10]} ({dd.get('period','?')})")
    return "\n".join(lines)


def _postprocess(text):
    """Strip ALL markdown formatting, robotic headers, confidence lines, leaked tokens, filler."""
    # 1. Remove leaked internal tokens
    for token in ["ANSWER_END", "</s>", "<|eot_id|>", "<|end_of_text|>"]:
        text = text.replace(token, "")
    # 2. Remove hallucinated page numbers
    text = re.sub(r'["\s]*(?:source:\s*)?page_no\s*=\s*\d+["\s]*', ' ', text)
    # 3. Strip ALL **bold** markdown — convert **text** to just text (universal fix)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # 4. Strip remaining * emphasis markers
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # 5. Remove rules_used / KP rule IDs ANYWHERE in text (not just line start)
    text = re.sub(r'rules_used:\s*[A-Z_0-9,\s]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bKP_[A-Z]{2,4}_\d{3,5}\b', '', text)
    # 5b. Remove [rule_id] references in brackets like [KP_TIM_0660]
    text = re.sub(r'\[KP_[A-Z_0-9]+\]', '', text)
    text = re.sub(r'\[rule_id\]', '', text, flags=re.IGNORECASE)
    # 6. Remove ALL "Confidence: xxx" lines/phrases entirely
    text = re.sub(r'[Cc]onfidence:?\s*:?\s*(?:high|medium|low|med)(?:\s*\([^)]*\))?', '', text)
    # 7. Remove robotic section headers the model generates (from client feedback)
    _robotic_headers = [
        r'Marriage\s+Timing\s+Analysis\s+using\s+KP\s+Principles',
        r'(?:Analysis|Conclusion|Application|Critical\s+Finding|Key\s+findings?|Summary)\s*:',
        r'(?:Motivational\s+Quote|Hindi\s+Quote|Recommended\s+Product|Product\s+Recommendation)\s*:',
        r'(?:Remedial\s+Measures|Remedy|Timing|Digestive\s+System|Immune\s+System)\s*:',
        r'According\s+to\s+rule\s+\[?KP[_A-Z0-9]*\]?\s*[:,]',
        r'Based\s+on\s+the\s+(?:given|extracted)\s+chart\s+(?:data|details|summary)',
        r'The\s+key\s+findings?\s+show\s+that\s*:',
        r'In\s+this\s+case,?\s+we\s+need\s+to',
        r'For\s+accurate\s+prediction,?\s+analyze',
        r'(?:House|Cusp)\s+\d+\s*(?:\([^)]*\))?\s*:\s*(?:sub\s*=|Sub-lord)',
    ]
    for pat in _robotic_headers:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
    # 8. Remove numbered list items (1. 2. 3.) and bullet points
    text = re.sub(r'(?:^|\n)\s*\d+\.\s+', '\n', text)
    text = re.sub(r'(?:^|\n)\s*[-•]\s+', '\n', text)
    # 9. Replace "The native has/is" with "Aap" for conversational tone
    text = re.sub(r'\bThe\s+native\s+has\b', 'Aapke paas', text, flags=re.IGNORECASE)
    text = re.sub(r'\bThe\s+native\s+is\b', 'Aap', text, flags=re.IGNORECASE)
    text = re.sub(r'\bThe\s+native\b', 'Aap', text, flags=re.IGNORECASE)
    text = re.sub(r'\bthe\s+native\b', 'aap', text)
    # 10. Remove metadata and filler lines
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip().lower()
        # Skip metadata lines
        if stripped.startswith("rules_used:") or stripped.startswith("rules used:"):
            continue
        if stripped.startswith("level:") or stripped.startswith("answer_end"):
            continue
        # Skip self-referential filler lines
        if any(filler in stripped for filler in [
            "considerably enhanced", "enhanced answer", "proper format",
            "additional recommendations", "professional competence",
            "theoretical understanding alone", "practical application validate",
            "absolute faith display", "considerable research effort",
            "let me analyze this situation systematically",
            "we need to identify planets signifying",
            "pending deeper analysis",
        ]):
            continue
        # Skip lines that are ONLY a short label/header (no real content)
        if stripped.endswith(":") and len(stripped) < 50 and not any(c.isdigit() for c in stripped):
            continue
        # Skip empty or near-empty lines after stripping
        if len(stripped) < 3:
            cleaned.append("")
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).rstrip()
    # 11. Clean up multiple blank lines
    result = re.sub(r'\n{3,}', '\n\n', result)
    # 12. Truncate to max ~3 paragraphs (enrichment adds 1 more with quote+product = 4 total)
    paragraphs = [p.strip() for p in result.split("\n\n") if p.strip()]
    if len(paragraphs) > 3:
        result = "\n\n".join(paragraphs[:3])
    # 13. Remove trailing incomplete sentences (cut off by token limit)
    if result and result[-1] not in '.!?"\n)}':
        last_period = max(result.rfind('. '), result.rfind('.\n'), result.rfind('.'))
        if last_period > len(result) * 0.4:  # trim if we keep >40%
            result = result[:last_period + 1]
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


def _enrich_response(text, product_text=""):
    """Append Hindi quote and product recommendation if model didn't include them."""
    text_lower = text.lower()

    # Check if model already included a Hindi/Hinglish quote-like phrase
    has_quote = any(q[:20].lower() in text_lower for q in HINDI_QUOTES)
    if not has_quote:
        quote_indicators = ["jab samay", "andhera jitna", "sabr ka phal", "har raat ke baad",
                           "mushkilein waqti", "waqt sabka", "kismat likhne"]
        has_quote = any(ind in text_lower for ind in quote_indicators)

    # Check if model already mentioned a product
    has_product = any(kw in text_lower for kw in [
        "pendant", "bracelet", "mala", "rudraksha", "kavach", "necklace",
        "gemstone", "neelam", "pukhraj", "moonga", "panna", "manik",
        "gomed", "pearl", "moti", "diamond", "sapphire", "coral",
        "emerald", "ruby", "hessonite", "cat eye", "hamara", "hamare",
    ])

    additions = []

    if not has_quote:
        quote = random.choice(HINDI_QUOTES)
        additions.append(quote)

    if not has_product and product_text:
        first_line = product_text.split("\n")[0] if product_text else ""
        match = re.match(r'-\s*(.+?)\s*\(SKU:', first_line)
        if match:
            product_name = match.group(1).strip()
            additions.append(
                f"Is samay ke liye hamara {product_name} try karein — yeh aapke planetary energies ko balance karne mein madad karega."
            )

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
    chart_data = _compact_chart_data(chart_data or "")
    summary = ""

    # Hard guard: no chart data + personal prediction question = ask for chart
    if not chart_data:
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

    if chart_data:
        summary = _chart_summary(chart_data)
        full_question = (f"Chart Data (JSON):\n{chart_data}\n\n"
                         f"Pre-extracted Chart Summary:\n{summary}\n\n"
                         f"Question: {question}")
    else:
        full_question = question

    rag_chunks = _retrieve_rag_chunks(question, top_k=args.top_k)

    # Product recommendations
    product_list, product_prompt_text = _get_product_recommendations(question, chart_summary=summary)
    product_instruction = ""
    if product_prompt_text:
        product_instruction = (
            f"\n\nRELEVANT PRODUCTS — YOU MUST MENTION EXACTLY ONE IN YOUR RESPONSE:\n"
            f"{product_prompt_text}\n"
            f"Pick the most relevant product and weave it into your last paragraph naturally. "
            f"Example: 'Is samay [planet] ko strengthen karne ke liye hamara [Product Name] bahut helpful hoga.'"
        )

    # Build prompt
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

    total_chars = sum(len(m["content"]) for m in messages)
    est_input_tokens = int(total_chars / 0.78) + 100
    available = MAX_MODEL_LEN - est_input_tokens
    max_tokens = max(64, min(OUTPUT_TOKENS, available))

    if max_tokens < 64:
        raise HTTPException(status_code=400, detail="Input too long for model context window.")

    response = llm_client.chat.completions.create(
        model="kp-astrology-llama",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=0.9,
        stream=False,
        extra_body={"repetition_penalty": 1.2},
    )

    raw_answer = response.choices[0].message.content or ""
    answer = _postprocess(raw_answer)

    # Enrich: append Hindi quote + product if model didn't include them
    answer = _enrich_response(answer, product_text=product_prompt_text)

    # Extract prediction
    prediction = _extract_prediction(answer)

    # Pick best product recommendation
    product_reco = product_list[0] if product_list else None

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
