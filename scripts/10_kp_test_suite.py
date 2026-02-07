"""
Comprehensive KP Astrology Model Test Suite

Runs 22 structured test questions against the vLLM server using real chart data
from chart_schema.json (TestUser, DOB: 01.01.1990, 10:00, Aquarius Lagna).

Each question includes ONLY the relevant chart facts (not the full chart) to stay
within the vLLM max_model_len=2048 token limit.

Outputs a JSON report + markdown summary for scoring against KP books.

Usage:
  python scripts/10_kp_test_suite.py
  python scripts/10_kp_test_suite.py --vllm-url http://localhost:8000/v1
"""

import json
import os
import re
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="KP Model Test Suite")
parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
parser.add_argument("--output", default="./results/kp_test_results.json")
parser.add_argument("--md-output", default="./results/kp_test_results.md")
parser.add_argument("--temperature", type=float, default=0.4)
parser.add_argument("--max-model-len", type=int, default=2048)
parser.add_argument("--no-rag", action="store_true", help="Disable RAG retrieval")
parser.add_argument("--top-k", type=int, default=5, help="RAG chunks to retrieve")
args = parser.parse_args()

client = OpenAI(base_url=args.vllm_url, api_key="not-needed")

# ── RAG: Pinecone + OpenAI embeddings (mirrors 09_chat_ui.py) ────────────────
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

# ── Context budget (calibrated from vLLM errors: 0.78 chars/token) ───────────
MAX_MODEL_LEN = args.max_model_len
OUTPUT_TOKENS = min(400, MAX_MODEL_LEN // 4)  # scale with context window
INPUT_TOKEN_BUDGET = MAX_MODEL_LEN - OUTPUT_TOKENS - 100
MAX_INPUT_CHARS = int(INPUT_TOKEN_BUDGET * 0.78)

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


def _retrieve_rag_chunks(question, top_k=5):
    """Retrieve relevant KP book chunks from Pinecone."""
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
        print(f"  RAG error: {e}")
        return []


def _postprocess(text):
    """Strip duplicate confidence/metadata blocks that the model sometimes repeats."""
    seen_conf = False
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip().lower()
        is_conf = stripped.startswith("confidence:") or stripped.startswith("**confidence")
        is_rules = stripped.startswith("rules_used:") or stripped.startswith("rules used:")
        if is_conf or is_rules:
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

# ── Test questions (compact — only relevant chart data per question) ──────────
TESTS = [
    # ── Marriage / 7th house (M1-M6) ─────────────────────────────────────────
    {
        "id": "M1", "category": "Marriage", "weight": 1.0,
        "expected_format": '{"result":"Yes/No","explanation":"...","rule":"KP_MAR_xxxx","confidence":"high/medium/low"}',
        "question": """KP chart: 7th cusp sub-lord is VEN (282-37-46, rashi SAT, nak MON, sub MAR).
Venus signifies houses: 1,4,6,9,12. House 7 significators: MER,SUN. House 2 sig: JUP,SUN. House 11 sig: JUP,MAR,MER,SAT,SUN.

According to KP: Will marriage occur? VEN is 7th sub-lord but signifies 6th(disputes) and 12th(losses), not 2/7/11.
Give verdict (Yes/No), explain sub-lord's role, cite KP rule, and confidence level.""",
    },
    {
        "id": "M2", "category": "Marriage", "weight": 1.0,
        "expected_format": '{"timing":"...","quote":"..."}',
        "question": """KP chart: 7th cusp sub-lord is VEN, signifying houses 1,4,6,9,12.
Hypothetical: VEN is retrograde. Dasa balance at birth: MAR 0Y 7M 23D.

According to KP: If the 7th sub-lord is retrograde, does it indicate delay, denial, or eventual marriage?
What timing conditions (dasha/transit) apply? Quote the specific KP rule for retrograde sub-lord effects on marriage.""",
    },
    {
        "id": "M3", "category": "Marriage", "weight": 0.9,
        "expected_format": "explanation quoting book rule",
        "question": """KP chart: 7th cusp sub-lord is VEN, signifying houses 1,4,6,9,12.
Hypothetical: Suppose VEN signified houses 1,7,9 simultaneously.

According to KP: What effect does a sub-lord being significator for 1st,7th,9th have on marriage?
Does 9th(dharma) help or complicate? Does 1st(self) strengthen or weaken? Cite the KP rule step by step.""",
    },
    {
        "id": "M4", "category": "Marriage", "weight": 1.0,
        "expected_format": "single planet chosen + step-by-step elimination",
        "question": """KP chart: House 7 significators: MER,SUN. Additional candidates: VEN(7th sub-lord), JUP(2nd sig), MAR(11th sig).
Planet significators — Sun:4,7,9,11,12. Mercury:5,7,8,9,11,12. Venus:1,4,6,9,12. Jupiter:2,5,11,12. Mars:1,3,10,11,12.

Multiple planets qualify for marriage timing. According to KP:
Which is the PRIMARY significator? What is the selection algorithm? Rank: sub-lord vs occupant vs lord vs star-lord. Cite rule.""",
    },
    {
        "id": "M5", "category": "Marriage", "weight": 0.8,
        "expected_format": "delay/deny/complication + rule quote",
        "question": """KP chart: 7th cusp nak-lord is Mars. Mars signifies houses 1,3,10,11,12.
Hypothetical: Mars is retrograde AND debilitated (in Cancer).

According to KP: What is the effect of a retrograde+debilitated planet as 7th cusp nak-lord on marriage?
Does KP treat debilitation differently from Vedic? Is outcome delay, denial, or complication? Quote the KP rule.""",
    },
    {
        "id": "M6", "category": "Marriage", "weight": 0.8,
        "expected_format": "list of stated reasons from KP",
        "question": """KP chart: 7th cusp sub-lord is VEN, signifying houses 1,4,6,9,12.
VEN signifies 6th(disputes/separation) and 12th(losses). Native had 5 broken engagements in 8 years.

According to KP: What explains repeated failed engagements? Which house connections cause this?
Role of 6th house? Role of 12th house? Any remedy or favorable timing? Cite KP rules.""",
    },

    # ── Venus & relationships (V1-V2) ────────────────────────────────────────
    {
        "id": "V1", "category": "Venus", "weight": 0.8,
        "expected_format": "verdict + book quote",
        "question": """KP chart: Venus at 282-37-46 in Aquarius(SAT rashi), Dhanishta nak(MON star), sub MAR.
Venus signifies: 1,4,6,9,12. Moon signifies: 1,3,6,10. Mars signifies: 1,3,10,11,12.

As natural karaka for marriage: How does Venus's nak-lord Moon affect marriage prospects?
Moon signifies 6th — does this hinder Venus? Venus sub-lord Mars signifies 11th — what does this indicate?
Will Venus predict harmonious marriage? Cite KP karaka analysis rule.""",
    },
    {
        "id": "V2", "category": "Venus", "weight": 0.7,
        "expected_format": "difference in interpretation + exact lines",
        "question": """According to KP astrology, what is the difference when Venus is:
(a) In the 5th bhava (love affairs, romance)
(b) In the 7th bhava (marriage, partnership)

What does Venus signify differently in each? How does Venus's sub-lord modify the result?
When does 5th-house Venus lead to marriage vs just love affairs? Cite KP rules for Venus in 5th vs 7th.""",
    },

    # ── Horary / Ruling planets (H1-H2) ──────────────────────────────────────
    {
        "id": "H1", "category": "Horary", "weight": 0.9,
        "expected_format": "list of ruling planets + method trace",
        "question": """A querent selects horary number 147 for a marriage question.

According to KP horary method: How to determine lagna degree from number 147?
What are the ruling planets? Show step by step: ascendant sign lord, star lord, sub-lord, Moon sign lord, Moon star lord, day lord.
How do ruling planets confirm or deny marriage? Cite KP horary rules.""",
    },
    {
        "id": "H2", "category": "Horary", "weight": 0.8,
        "expected_format": "tie-breaker rules + which planets to drop",
        "question": """KP horary: 7th house significators found — Sun(occupant), Mercury(sub-lord connection), Venus(karaka+7th sub-lord), Saturn(depositor chain), Rahu(star of 7th occupant).
Some are benefic, some malefic for marriage.

According to KP: What are the tie-breaker rules? Which planets to drop and why?
Priority order: sub-lord > star-lord > sign-lord? Cite KP significator selection rules.""",
    },

    # ── Financial / 11th house (F1-F2) ───────────────────────────────────────
    {
        "id": "F1", "category": "Financial", "weight": 0.9,
        "expected_format": '{"gains_likely":"Yes/No","timing":"...","rule":"..."}',
        "question": """KP chart: 11th cusp sub-lord is MAR. Mars signifies houses: 1,3,10,11,12.
House 2 sig: JUP,SUN. House 11 sig: JUP,MAR,MER,SAT,SUN.

According to KP: Will native experience financial gains? Mars signifies 11th(gains) but also 12th(losses) — how to interpret?
Timing conditions (dasha)? Role of 2nd house connection? Cite KP rule for 11th sub-lord analysis.""",
    },
    {
        "id": "F2", "category": "Financial", "weight": 0.8,
        "expected_format": "explanation for blocked gains + exact quote",
        "question": """KP chart: 11th sub-lord MAR signifies 1,3,10,11,12 (includes 12th=losses).
8th sub-lord RAH signifies 3,10,12.

According to KP: When 11th sub-lord connects to 12th house, what happens to gains?
When 8th sub-lord connects to 12th, what does it indicate? Gains blocked, delayed, or redirected?
Cite KP rules for 11th-12th and 8th-12th connections.""",
    },

    # ── Dasha / Timing (T1-T2) ───────────────────────────────────────────────
    {
        "id": "T1", "category": "Timing", "weight": 1.0,
        "expected_format": "quote + Yes/No",
        "question": """KP chart: Marriage houses are 2,7,11. 7th cusp sub-lord is VEN(signifies 1,4,6,9,12).
House 7 sig: MER,SUN. House 2 sig: JUP,SUN. House 11 sig: JUP,MAR,MER,SAT,SUN.

According to KP: Must BOTH Mahadasha lord AND Antardasha lord connect to houses 2,7,11 for marriage?
Or is one sufficient? What is the exact KP rule text for dasha-bhukti requirements?
Which Maha-Antar combination would trigger marriage in this chart? Cite rule.""",
    },
    {
        "id": "T2", "category": "Timing", "weight": 0.9,
        "expected_format": "expected outcome/timing",
        "question": """KP chart: Native runs Jupiter Mahadasha. Jupiter signifies: 2,5,11,12 (connected to 2,11 — favorable).
Current Antardasha: Ketu. Ketu signifies: 5,6,8,12 (NOT connected to 2,7,11).

According to KP: When Maha lord connects to marriage houses but Antar lord does NOT, what happens?
Will marriage occur in this sub-period or be delayed? Which Antardasha in Jupiter Maha is most favorable?
Cite KP timing rule.""",
    },

    # ── Transit interactions (TR1-TR2) ────────────────────────────────────────
    {
        "id": "TR1", "category": "Transit", "weight": 0.7,
        "expected_format": "stated effect on timing/outcome",
        "question": """KP chart: 7th sub-lord Venus is at 282-37-46 (Aquarius).
Saturn (malefic) transits over Venus's natal position during a favorable dasha for marriage.

According to KP: Does malefic transit over sub-lord affect event timing?
Does KP prioritize dasha over transits? Will Saturn's transit delay marriage during favorable dasha?
Cite KP rule for transit effects.""",
    },
    {
        "id": "TR2", "category": "Transit", "weight": 0.7,
        "expected_format": "mitigation rules",
        "question": """KP chart: 7th cusp at 122-12-49 (Leo). 7th sub-lord VEN is weak (signifies 6th,12th).
Jupiter (benefic) transits over 7th cusp degree. Dasha is marginally favorable.

According to KP: Can benefic transit rescue outcome when sub-lord is weak?
Does KP give transit power to override sub-lord indications? Transit vs sub-lord primacy?
Cite KP rule.""",
    },

    # ── Edge / complex cases (E1-E3) ─────────────────────────────────────────
    {
        "id": "E1", "category": "Edge", "weight": 0.8,
        "expected_format": "step-by-step evaluation + final verdict",
        "question": """KP chart marriage analysis (houses 2,7,11):
Sun: sig 4,7,9,11,12 (7+11 favorable, 12 unfavorable)
Mercury: sig 5,7,8,9,11,12 (7+11 favorable, 8+12 unfavorable)
Venus(7th sub-lord): sig 1,4,6,9,12 (NOT connected to 7 or 11)
Jupiter: sig 2,5,11,12 (2+11 favorable, 12 unfavorable)

4+ conflicting significators. According to KP: Step-by-step evaluation?
Which significator has priority? VEN doesn't signify 7/11 — what does this mean?
Final verdict: marriage will/won't happen? Cite KP algorithm.""",
    },
    {
        "id": "E2", "category": "Edge", "weight": 0.6,
        "expected_format": "special rules for water signs",
        "question": """According to KP: Are there special rules when:
1. 7th cusp sub-lord is in a watery sign (Cancer/Scorpio/Pisces)?
2. 7th cusp itself falls in a watery sign?
3. Does sign element (fire/earth/air/water) affect marriage predictions in KP?
4. Chart's 7th cusp is at 122-12-49 (Leo, fire sign) — any special significance?
Cite KP rules for sign classification effects on marriage.""",
    },
    {
        "id": "E3", "category": "Edge", "weight": 0.7,
        "expected_format": "mitigation or denial + quote",
        "question": """According to KP: A planet is debilitated (e.g. Venus in Virgo) BUT in nakshatra of a benefic (Jupiter's star).
1. Does KP treat debilitation same as Vedic astrology?
2. Does nakshatra lord override debilitation in KP?
3. If debilitated planet is 7th sub-lord, does benefic nak-lord mitigate?
4. KP position: mitigation or denial?
Cite KP rule for debilitated planets in benefic nakshatras.""",
    },

    # ── Quality / format checks (Q1-Q2) ──────────────────────────────────────
    {
        "id": "Q1", "category": "Quality", "weight": 0.5,
        "expected_format": "verbatim quote with page/section",
        "question": """For rule citation fidelity: What is the exact content of KP rules KP_MAR_0673 and KP_MAR_0971?
What chapter/section of KP Reader do they come from? Are they from KP Reader I,II,III,IV,V, or VI?
Provide verbatim text of each rule.""",
    },
    {
        "id": "Q2", "category": "Quality", "weight": 0.5,
        "expected_format": "Yes/No + evidence",
        "question": """When you cite rules like "KP_MAR_0673", is this an actual rule ID from KP Readers by Prof. K.S. Krishnamurti?
Or internally generated from training data? How confident are you in citation accuracy?
Can you distinguish KP Reader I (general) vs IV (marriage) vs VI (horary)?
What is your confidence the cited rule content matches original KP text?""",
    },
]


# ── Run tests ─────────────────────────────────────────────────────────────────
def run_test(test):
    """Send a test question to vLLM with RAG context (mirrors chat UI)."""
    question = test["question"]

    # Retrieve RAG chunks
    rag_chunks = _retrieve_rag_chunks(question, top_k=args.top_k)

    # Build system prompt with adaptive RAG trimming
    fixed_chars = len(SYSTEM_BASE) + len(question) + 30
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

    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": question},
    ]

    # Calibrated token budget
    total_chars = sum(len(m["content"]) for m in messages)
    est_input_tokens = int(total_chars / 0.78) + 100
    available = MAX_MODEL_LEN - est_input_tokens
    max_output = max(64, min(OUTPUT_TOKENS, available))

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model="kp-astrology-llama",
            messages=messages,
            max_tokens=max_output,
            temperature=args.temperature,
            top_p=0.9,
            extra_body={"repetition_penalty": 1.15},
        )
        elapsed = time.time() - t0
        answer = _postprocess(resp.choices[0].message.content.strip())
        tokens = resp.usage.total_tokens if resp.usage else 0
        return {
            "answer": answer,
            "latency_s": round(elapsed, 2),
            "tokens": tokens,
            "rag_chunks": len(selected_chunks),
            "error": None,
        }
    except Exception as e:
        return {"answer": "", "latency_s": 0, "tokens": 0, "rag_chunks": len(selected_chunks), "error": str(e)}


def main():
    print("=" * 80)
    print("KP ASTROLOGY MODEL TEST SUITE")
    print("=" * 80)
    print(f"Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"vLLM:        {args.vllm_url}")
    print(f"Tests:       {len(TESTS)}")
    print(f"Temperature: {args.temperature}")
    print(f"RAG:         {'Enabled (top-k=' + str(args.top_k) + ')' if rag_index else 'Disabled'}")
    print(f"Output:      {args.output}")
    print("=" * 80)

    results = []
    for i, test in enumerate(TESTS, 1):
        print(f"\n[{i}/{len(TESTS)}] {test['id']} — {test['category']}...")
        res = run_test(test)
        result = {
            "q_id": test["id"],
            "category": test["category"],
            "weight": test["weight"],
            "question": test["question"][:200] + "...",
            "expected_format": test["expected_format"],
            "model_answer": res["answer"],
            "latency_s": res["latency_s"],
            "tokens": res["tokens"],
            "rag_chunks": res.get("rag_chunks", 0),
            "error": res["error"],
            # Placeholders for manual scoring
            "book_answer": "",
            "rule_ref": "",
            "score": None,
            "score_notes": "",
        }
        results.append(result)

        # Quick quality checks
        answer_lower = res["answer"].lower()
        has_rule = "rule" in answer_lower or "kp_" in answer_lower
        has_confidence = "confidence" in answer_lower
        has_house = any(f"{h}" in answer_lower for h in ["sub-lord", "sub lord", "cusp", "significator"])

        status = "OK" if not res["error"] else f"ERROR: {res['error']}"
        print(f"  Status: {status} | {res['latency_s']}s | {len(res['answer'])} chars | RAG:{res.get('rag_chunks',0)} chunks")
        print(f"  Rule citation: {'Yes' if has_rule else 'NO'} | "
              f"Confidence: {'Yes' if has_confidence else 'NO'} | "
              f"KP terms: {'Yes' if has_house else 'NO'}")

    # ── Save JSON results ─────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "vllm_url": args.vllm_url,
                "temperature": args.temperature,
                "rag_enabled": rag_index is not None,
                "top_k": args.top_k,
                "num_tests": len(TESTS),
                "chart": "TestUser, 01.01.1990, 10:00, Aquarius Lagna",
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved: {out_path}")

    # ── Save Markdown report ──────────────────────────────────────────────────
    md_path = Path(args.md_output)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# KP Astrology Model Test Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Model:** kp-astrology-llama (Llama 3.1 8B fine-tuned)  \n")
        f.write(f"**Chart:** TestUser, 01.01.1990, 10:00, Aquarius Lagna  \n")
        f.write(f"**Temperature:** {args.temperature}  \n")
        f.write(f"**RAG:** {'Enabled (top-k=' + str(args.top_k) + ')' if rag_index else 'Disabled'}  \n\n")

        f.write("## Summary\n\n")
        f.write("| # | ID | Category | Weight | Latency | RAG | Rule Cited | Confidence | Score |\n")
        f.write("|---|-----|----------|--------|---------|-----|------------|------------|-------|\n")
        for i, r in enumerate(results, 1):
            al = r["model_answer"].lower()
            rule = "Yes" if ("rule" in al or "kp_" in al) else "No"
            conf = "Yes" if "confidence" in al else "No"
            score = r["score"] if r["score"] is not None else "—"
            f.write(f"| {i} | {r['q_id']} | {r['category']} | {r['weight']} | "
                    f"{r['latency_s']}s | {r.get('rag_chunks',0)} | {rule} | {conf} | {score} |\n")

        f.write("\n## Detailed Results\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"### {r['q_id']} — {r['category']} (weight: {r['weight']})\n\n")
            f.write(f"**Question (truncated):** {r['question']}\n\n")
            f.write(f"**Expected format:** {r['expected_format']}\n\n")
            f.write(f"**Model answer:**\n\n")
            f.write(f"```\n{r['model_answer']}\n```\n\n")
            f.write(f"**Book answer:** _{r['book_answer'] or 'TO BE FILLED'}_\n\n")
            f.write(f"**Rule ref:** _{r['rule_ref'] or 'TO BE FILLED'}_\n\n")
            f.write(f"**Score:** {r['score'] if r['score'] is not None else 'TO BE SCORED'}\n\n")
            f.write(f"**Notes:** {r['score_notes'] or ''}\n\n")
            f.write("---\n\n")

        f.write("## Scoring Rubric\n\n")
        f.write("- **1.0 Exact-match:** model answer matches book passage word-for-word\n")
        f.write("- **0.75 Strong-match:** model paraphrases but keeps all logical conditions, correct rule ID\n")
        f.write("- **0.5 Partial:** partially correct but misses important clause/caveat\n")
        f.write("- **0.0 Mismatch:** model contradicts book or fabricates rule\n\n")
        f.write("### Penalties\n")
        f.write("- Missing rule citation: -0.1\n")
        f.write("- Incorrect application: -0.15\n")
        f.write("- Hallucinated rule: -0.25\n")

    print(f"Markdown saved: {md_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    total = len(results)
    errors = sum(1 for r in results if r["error"])
    avg_latency = sum(r["latency_s"] for r in results) / max(total, 1)
    rules_cited = sum(1 for r in results
                      if "rule" in r["model_answer"].lower() or "kp_" in r["model_answer"].lower())
    confidence_cited = sum(1 for r in results if "confidence" in r["model_answer"].lower())

    print(f"\n{'='*80}")
    print(f"TEST SUITE COMPLETE")
    print(f"{'='*80}")
    print(f"  Total tests:      {total}")
    print(f"  Errors:           {errors}")
    print(f"  Avg latency:      {avg_latency:.2f}s")
    print(f"  Rule citations:   {rules_cited}/{total} ({rules_cited/total*100:.0f}%)")
    print(f"  Confidence cited: {confidence_cited}/{total} ({confidence_cited/total*100:.0f}%)")
    print(f"\nTo score: fill in 'book_answer', 'rule_ref', and 'score' in {args.output}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
