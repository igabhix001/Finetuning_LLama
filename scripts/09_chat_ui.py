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
import random
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
            print("  RAG:    DISABLED (missing PINECONE_API_KEY or OPENAI_API_KEY)")
    except Exception as e:
        print(f"  RAG:    DISABLED (init error: {e})")
else:
    print("  RAG:    DISABLED (--no-rag flag)")

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

# ── Product catalog (for remedy recommendations) ─────────────────────────────
PRODUCT_CATALOG = []
_products_path = args.products_csv
if not _products_path:
    # Auto-discover product CSV in common locations
    import glob
    _search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),           # scripts/
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),  # Finetuning_LLama/
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),  # project root
        "/workspace",                                          # RunPod default
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


# ── Context-window budget ─────────────────────────────────────────────────────
# Calibrated from actual vLLM errors:
#   ~1450 chars of content → 1865 actual tokens (ratio ≈ 0.78 chars/token)
#   Llama 3.1 chat template adds ~100 tokens overhead per conversation
# Strategy: use HARD CHARACTER BUDGET instead of unreliable token estimates.
MAX_MODEL_LEN = args.max_model_len
# Output tokens: 400 for ≤4096, 768 for larger contexts (need enough for 3-4 paragraphs)
OUTPUT_TOKENS = 400 if MAX_MODEL_LEN <= 4096 else min(768, MAX_MODEL_LEN // 6)
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

    # ── Method 2: CSV keyword fallback ──
    if not PRODUCT_CATALOG:
        return ""
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
        return ""
    matches = matches[:max_items]
    lines = []
    for p in matches:
        lines.append(f"- {p['Title']} (SKU: {p.get('SKU','')}, Rs.{p.get('Sale Price','')})")
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
    # 12. Truncate to max ~4 paragraphs (split on double newline, keep first 4)
    paragraphs = [p.strip() for p in result.split("\n\n") if p.strip()]
    if len(paragraphs) > 4:
        result = "\n\n".join(paragraphs[:4])
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
        # Also check for common quote patterns the model might generate on its own
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
        # Extract first product name from product_text lines
        first_line = product_text.split("\n")[0] if product_text else ""
        # Parse "- Product Name (SKU: xxx, Rs.yyy)" format
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


MAX_CHART_CHARS = 12000  # safety cap for compacted JSON (includes antardasha for timing)


def _compact_chart_data(raw: str) -> str:
    """Parse computation-engine JSON and return a compact version preserving JSON structure.

    Keeps (in original JSON format): name, gender, birthDetails, planetKP, cuspKP,
        significators, planetSignifications, dashaBalance, top-level mahadasha periods.
    Discards: planetaryPositions, cuspalPositions, debug, ayanamsa, subSub fields,
        and the massive nested antarDasha/pratyantarDasha trees (~90% of file size).
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

    # Build a slim copy keeping only KP-essential fields in original JSON format
    slim = {}
    for key in ("name", "gender"):
        if key in d:
            slim[key] = d[key]

    if "birthDetails" in d:
        slim["birthDetails"] = d["birthDetails"]

    # planetKP — keep all fields except subSub (not needed for predictions)
    if "planetKP" in d:
        slim["planetKP"] = {}
        for planet, pdata in d["planetKP"].items():
            slim["planetKP"][planet] = {k: v for k, v in pdata.items() if k != "subSub"}

    # cuspKP — keep all fields except subSub
    if "cuspKP" in d:
        slim["cuspKP"] = {}
        for cusp, cdata in d["cuspKP"].items():
            slim["cuspKP"][cusp] = {k: v for k, v in cdata.items() if k != "subSub"}

    # significators and planetSignifications — keep as-is (small)
    for key in ("significators", "planetSignifications"):
        if key in d:
            slim[key] = d[key]

    # dashas — keep dashaBalance + mahadasha + antardasha (for month-level timing)
    if "dashas" in d:
        slim["dashas"] = {}
        if "dashaBalance" in d["dashas"]:
            slim["dashas"]["dashaBalance"] = d["dashas"]["dashaBalance"]
        if "dashas" in d["dashas"]:
            slim["dashas"]["mahadashas"] = []
            for dd in d["dashas"]["dashas"]:
                maha = {
                    "lord": dd.get("lord"),
                    "startDate": dd.get("startDate", "")[:10],
                    "endDate": dd.get("endDate", "")[:10],
                    "period": dd.get("period"),
                }
                # Include antardasha (bhukti) periods for specific timing
                if "antarDashas" in dd:
                    maha["antarDashas"] = [
                        {"lord": ad.get("lord"),
                         "startDate": ad.get("startDate", "")[:10],
                         "endDate": ad.get("endDate", "")[:10],
                         "period": ad.get("period")}
                        for ad in dd["antarDashas"]
                    ]
                slim["dashas"]["mahadashas"].append(maha)

    result = json.dumps(slim, indent=1, ensure_ascii=False)
    # Final safety truncation
    if len(result) > MAX_CHART_CHARS:
        result = result[:MAX_CHART_CHARS] + "\n...}"
    return result


def _chart_summary(raw: str) -> str:
    """Extract key KP values from chart JSON into plain-text summary for the model.

    The model struggles to parse raw JSON autonomously. This function pre-extracts
    the most important values so they appear as readable text in the prompt.
    """
    try:
        d = json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return ""

    lines = []
    name = d.get("name", "Unknown")
    bd = d.get("birthDetails", {})
    lines.append(f"Native: {name}, DOB: {bd.get('date','?')}, TOB: {bd.get('time','?')}, "
                 f"Lagna: {bd.get('lagna','?')} ({bd.get('lagnaLord','?')})")

    # Key cusp sub-lords (most asked about: 1,2,6,7,10,11)
    ckp = d.get("cuspKP", {})
    if ckp:
        lines.append("KEY CUSP SUB-LORDS:")
        for c in ["1", "2", "6", "7", "10", "11", "12"]:
            cp = ckp.get(c, {})
            if cp:
                lines.append(f"  Cusp {c}: sub={cp.get('sub','?')}, "
                             f"nak={cp.get('nakshatra','?')}({cp.get('nakshatraLord','?')}), "
                             f"rashi={cp.get('rashi','?')}, degree={cp.get('degree','?')}")

    # Planet significations (critical for analysis)
    psig = d.get("planetSignifications", {})
    if psig:
        lines.append("PLANET SIGNIFICATIONS (houses each planet signifies):")
        for planet in ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Rahu", "Ketu"]:
            houses = psig.get(planet, [])
            if houses:
                lines.append(f"  {planet}: houses {houses}")

    # Current dasha with antardasha for specific timing
    dashas = d.get("dashas", {})
    db = dashas.get("dashaBalance", {})
    if db:
        lines.append(f"DASHA BALANCE: {db.get('lord','?')} "
                     f"{db.get('years',0)}Y {db.get('months',0)}M {db.get('days',0)}D remaining")
    dlist = dashas.get("dashas", dashas.get("mahadashas", []))
    if dlist:
        lines.append("MAHADASHA & ANTARDASHA PERIODS (use these for specific date predictions):")
        for dd in dlist[:3]:  # first 3 mahadashas
            lines.append(f"  {dd.get('lord','?')} Mahadasha: {dd.get('startDate','?')[:10]} to "
                         f"{dd.get('endDate','?')[:10]} ({dd.get('period','?')})")
            # Include antardasha sub-periods for month-level timing
            antardashas = dd.get("antarDashas", [])
            if antardashas:
                for ad in antardashas:
                    lines.append(f"    → {dd.get('lord','?')}-{ad.get('lord','?')} bhukti: "
                                 f"{ad.get('startDate','?')[:10]} to {ad.get('endDate','?')[:10]} "
                                 f"({ad.get('period','?')})")

    return "\n".join(lines)


def predict(message, history, chart_data):
    """Stream a response from the vLLM server with RAG-augmented context + chart data."""
    # 0. Compact chart data (auto-parse large JSON from computation engine)
    chart_data = _compact_chart_data(chart_data or "")
    summary = ""

    # Hard guard: if no chart data and user asks a personal prediction question,
    # don't let the model hallucinate — ask for chart data first.
    if not chart_data:
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

    if chart_data:
        # Auto-surface key values so model doesn't need to parse JSON
        summary = _chart_summary(chart_data)
        full_question = (f"Chart Data (JSON):\n{chart_data}\n\n"
                         f"Pre-extracted Chart Summary:\n{summary}\n\n"
                         f"Question: {message}")
    else:
        full_question = message

    # 1. Retrieve RAG chunks (search using original question for better retrieval)
    rag_chunks = _retrieve_rag_chunks(message, top_k=args.top_k)

    # 2. Product recommendations — inject into system prompt so model weaves them naturally
    product_text = _get_product_recommendations(message, chart_summary=summary)
    product_instruction = ""
    if product_text:
        product_instruction = (
            f"\n\nRELEVANT PRODUCTS — YOU MUST MENTION EXACTLY ONE IN YOUR RESPONSE:\n"
            f"{product_text}\n"
            f"Pick the most relevant product and weave it into your last paragraph naturally. "
            f"Example: 'Is samay [planet] ko strengthen karne ke liye hamara [Product Name] bahut helpful hoga.'"
        )

    # 3. Build prompt with adaptive RAG trimming to fit character budget
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

    # 4. Build messages (no history — every question gets fresh RAG)
    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": full_question},
    ]

    # 5. Final safety: compute actual char total and adjust output tokens
    total_chars = sum(len(m["content"]) for m in messages)
    est_input_tokens = int(total_chars / 0.78) + 100
    available = MAX_MODEL_LEN - est_input_tokens
    max_tokens = max(64, min(OUTPUT_TOKENS, available))

    if max_tokens < 64:
        yield (f"Your message is too long for the model's {MAX_MODEL_LEN}-token context. "
               "Please shorten the chart data or question.")
        return

    try:
        stream = client.chat.completions.create(
            model="kp-astrology-llama",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.5,
            top_p=0.9,
            stream=True,
            extra_body={"repetition_penalty": 1.2},
        )
        partial = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                partial += delta
                yield _postprocess(partial)
        # Final enrichment: append Hindi quote + product if model didn't include them
        if partial:
            final = _postprocess(partial)
            final = _enrich_response(final, product_text=product_text)
            yield final
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
