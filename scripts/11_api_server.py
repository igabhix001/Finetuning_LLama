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
            print("  RAG:    DISABLED (missing keys)")
    except Exception as e:
        print(f"  RAG:    DISABLED ({e})")
else:
    print("  RAG:    DISABLED (--no-rag)")

# ── Product catalog ──────────────────────────────────────────────────────────
PRODUCT_CATALOG = []
if args.products_csv and os.path.isfile(args.products_csv):
    try:
        with open(args.products_csv, encoding="utf-8") as f:
            PRODUCT_CATALOG = list(csv.DictReader(f))
        print(f"  Products: {len(PRODUCT_CATALOG)} items loaded")
    except Exception as e:
        print(f"  Products: FAILED ({e})")

# ── System prompts ───────────────────────────────────────────────────────────
SYSTEM_BASE = (
    "You are a warm, experienced KP astrologer speaking directly to the person sitting in front of you. "
    "Talk like a real astrologer — conversational, confident, compassionate. Use Hinglish naturally. "
    "Sprinkle in motivational Hindi sayings/quotes where appropriate.\n\n"
    "STYLE RULES:\n"
    "- Talk TO the person: 'Your 7th house lord...', 'You will see...', NOT 'The native has...'\n"
    "- Give SPECIFIC time predictions: actual months, years, date ranges derived from dasha periods in the chart. "
    "Example: 'Marriage yoga is forming between March 2026 to August 2026.' NOT vague theory.\n"
    "- When chart has dasha data, CALCULATE which dasha/bhukti period covers the event and state the dates.\n"
    "- Keep it SHORT and punchy — 4-6 sentences max. No long academic paragraphs.\n"
    "- NEVER use robotic headers like 'Analysis:', 'Conclusion:', 'Confidence: medium'. "
    "Just speak naturally like a human astrologer.\n"
    "- If a remedy product is relevant, weave it naturally: 'To strengthen Venus, wear our Shukra Kavach Pendant.'\n"
    "- Add a Hindi/Hinglish quote naturally: 'Jab samay aayega, rishta khud chalkar aayega.'\n\n"
    "DATA RULES:\n"
    "1. If Chart Data is provided, READ the Pre-extracted Chart Summary carefully. "
    "Use exact values: cusp sub-lords, planet significations, dasha periods, degrees.\n"
    "2. Use KP Book Excerpts below for rules. You may reference [rule_id] briefly but don't make it the focus.\n"
    "3. NEVER invent page numbers or chapter numbers.\n"
    "4. If no chart data is provided, say 'Please share your birth chart so I can give you an accurate reading.'\n"
    "5. No repetition. Be direct and specific.\n"
)

SYSTEM_NO_RAG = (
    "You are a warm, experienced KP astrologer speaking directly to the person. "
    "Talk conversationally in Hinglish. Give SPECIFIC dates/months/years from dasha data.\n\n"
    "RULES:\n"
    "1. If Chart Data is provided, use exact values from the Pre-extracted Chart Summary.\n"
    "2. Give specific time predictions from dasha periods — not vague theory.\n"
    "3. Talk TO the person: 'Your Venus is strong...', 'You will see improvement by March 2026...'\n"
    "4. Keep it short, warm, and punchy. No robotic headers. Add Hindi quotes naturally.\n"
    "5. NEVER invent page numbers. If no chart data, ask them to share their birth chart.\n"
)

# ── Context-window budget ────────────────────────────────────────────────────
MAX_MODEL_LEN = args.max_model_len
OUTPUT_TOKENS = 400 if MAX_MODEL_LEN <= 4096 else min(512, MAX_MODEL_LEN // 8)
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
    # Return both structured list and text for prompt injection
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
    for token in ["ANSWER_END", "</s>", "<|eot_id|>", "<|end_of_text|>"]:
        text = text.replace(token, "")
    text = re.sub(r'["\s]*(?:source:\s*)?page_no\s*=\s*\d+["\s]*', ' ', text)
    seen_conf = False
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip().lower()
        is_conf = stripped.startswith("confidence:") or stripped.startswith("**confidence")
        is_rules = stripped.startswith("rules_used:") or stripped.startswith("rules used:")
        is_meta = stripped.startswith("level:") or stripped.startswith("answer_end")
        if is_conf or is_rules or is_meta:
            if seen_conf:
                continue
            seen_conf = True
        cleaned.append(line)
    result = "\n".join(cleaned).rstrip()
    if result and result[-1] not in '.!?"\n)}':
        last_period = max(result.rfind('. '), result.rfind('.\n'), result.rfind('.'))
        if last_period > len(result) * 0.7:
            result = result[:last_period + 1]
    return result


def _extract_prediction(answer: str) -> Optional[str]:
    """Extract specific date/time predictions from the answer text."""
    # Look for date patterns like "March 2026", "2026-2027", "April 2026 to August 2026"
    date_patterns = [
        r'(?:between|from|during|by|after|before|till|until)\s+\w+\s+\d{4}\s+(?:to|and|till|-)\s+\w+\s+\d{4}',
        r'(?:between|from|during|by|after|before|till|until)\s+\w+\s+\d{4}',
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+(?:to|and|till|-)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        r'\d{4}\s*(?:to|-)\s*\d{4}',
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
            f"\n\nRELEVANT PRODUCTS (weave ONE naturally into your answer as a remedy suggestion):\n"
            f"{product_prompt_text}\n"
            f"Example: 'To strengthen [planet], I'd recommend our [Product Name].'"
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
