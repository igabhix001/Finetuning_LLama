"""
Comprehensive KP Astrology Model Test Suite

Runs 22 structured test questions against the vLLM server using real chart data
from chart_schema.json (TestUser, DOB: 01.01.1990, 10:00, Aquarius Lagna).

Outputs a JSON report + markdown summary for scoring against KP books.

Usage:
  python scripts/10_kp_test_suite.py
  python scripts/10_kp_test_suite.py --vllm-url http://localhost:8000/v1
  python scripts/10_kp_test_suite.py --output results/kp_test_results.json
"""

import json
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="KP Model Test Suite")
parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
parser.add_argument("--output", default="./results/kp_test_results.json")
parser.add_argument("--md-output", default="./results/kp_test_results.md")
parser.add_argument("--temperature", type=float, default=0.3,
                    help="Lower temp for deterministic evaluation")
args = parser.parse_args()

client = OpenAI(base_url=args.vllm_url, api_key="not-needed")

# ── Chart data (from chart_schema.json — TestUser) ───────────────────────────
CHART = {
    "birth": "01.01.1990, 10:00:00, Lat 28.61N / Lon 77.21E",
    "lagna": "Aquarius", "lagna_lord": "SAT",
    "nakshatra": "Dhanishta", "nakshatra_lord": "MAR",
    "dasa_balance": "MAR 0Y 7M 23D",
    "cusps": {
        1: {"deg": "302-12-49", "rashi": "SAT", "nak": "MAR", "sub": "KET", "ss": "MER"},
        2: {"deg": "343-11-40", "rashi": "JUP", "nak": "SAT", "sub": "MAR", "ss": "KET"},
        3: {"deg": "016-51-49", "rashi": "MAR", "nak": "VEN", "sub": "MON", "ss": "JUP"},
        4: {"deg": "043-28-17", "rashi": "VEN", "nak": "MON", "sub": "RAH", "ss": "MER"},
        5: {"deg": "067-01-45", "rashi": "MER", "nak": "RAH", "sub": "RAH", "ss": "SAT"},
        6: {"deg": "091-35-47", "rashi": "MON", "nak": "JUP", "sub": "MAR", "ss": "SUN"},
        7: {"deg": "122-12-49", "rashi": "SUN", "nak": "KET", "sub": "VEN", "ss": "RAH"},
        8: {"deg": "163-11-40", "rashi": "MER", "nak": "MON", "sub": "RAH", "ss": "JUP"},
        9: {"deg": "196-51-49", "rashi": "VEN", "nak": "RAH", "sub": "SUN", "ss": "KET"},
        10: {"deg": "223-28-17", "rashi": "MAR", "nak": "SAT", "sub": "MAR", "ss": "SUN"},
        11: {"deg": "247-01-45", "rashi": "JUP", "nak": "KET", "sub": "MAR", "ss": "VEN"},
        12: {"deg": "271-35-47", "rashi": "SAT", "nak": "SUN", "sub": "RAH", "ss": "SAT"},
    },
    "planets": {
        "Sun":     {"deg": "256-51-52", "rashi": "JUP", "nak": "VEN", "sub": "MON", "ss": "JUP"},
        "Moon":    {"deg": "305-26-12", "rashi": "SAT", "nak": "MAR", "sub": "MON", "ss": "MAR"},
        "Mars":    {"deg": "226-08-56", "rashi": "MAR", "nak": "SAT", "sub": "JUP", "ss": "SUN"},
        "Mercury": {"deg": "272-07-26", "rashi": "SAT", "nak": "SUN", "sub": "RAH", "ss": "SUN"},
        "Jupiter": {"deg": "071-33-35", "rashi": "MER", "nak": "RAH", "sub": "MER", "ss": "VEN"},
        "Venus":   {"deg": "282-37-46", "rashi": "SAT", "nak": "MON", "sub": "MAR", "ss": "VEN"},
        "Saturn":  {"deg": "261-59-22", "rashi": "JUP", "nak": "VEN", "sub": "JUP", "ss": "MAR"},
        "Rahu":    {"deg": "294-49-13", "rashi": "SAT", "nak": "MAR", "sub": "RAH", "ss": "RAH"},
        "Ketu":    {"deg": "114-49-13", "rashi": "MON", "nak": "MER", "sub": "MAR", "ss": "MER"},
    },
    "house_sig": {
        1: ["MAR","MON","SAT","VEN"], 2: ["JUP","SUN"], 3: ["MAR","MON","RAH"],
        4: ["SAT","SUN","VEN"], 5: ["JUP","KET","MER","SUN"], 6: ["KET","MON","SUN","VEN"],
        7: ["MER","SUN"], 8: ["KET","MER"], 9: ["MER","SAT","SUN","VEN"],
        10: ["MAR","MON","RAH"], 11: ["JUP","MAR","MER","SAT","SUN"],
        12: ["JUP","KET","MAR","MER","RAH","SAT","SUN","VEN"],
    },
    "planet_sig": {
        "Sun": [4,7,9,11,12], "Moon": [1,3,6,10], "Mercury": [5,7,8,9,11,12],
        "Venus": [1,4,6,9,12], "Mars": [1,3,10,11,12], "Jupiter": [2,5,11,12],
        "Saturn": [1,4,9,11,12], "Rahu": [3,10,12], "Ketu": [5,6,8,12],
    },
}

CHART_CONTEXT = f"""Chart Data (KP System):
Birth: {CHART['birth']}
Lagna: {CHART['lagna']} (Lord: {CHART['lagna_lord']}), Nakshatra: {CHART['nakshatra']} (Lord: {CHART['nakshatra_lord']})
Dasa Balance at birth: {CHART['dasa_balance']}

Cusps (Cusp | Degree | Rashi Lord | Nak Lord | Sub Lord | Sub-Sub Lord):
""" + "\n".join(
    f"  {c}: {d['deg']} | {d['rashi']} | {d['nak']} | {d['sub']} | {d['ss']}"
    for c, d in CHART["cusps"].items()
) + """

Planets (Planet | Degree | Rashi Lord | Nak Lord | Sub Lord | Sub-Sub Lord):
""" + "\n".join(
    f"  {p}: {d['deg']} | {d['rashi']} | {d['nak']} | {d['sub']} | {d['ss']}"
    for p, d in CHART["planets"].items()
) + """

House Significators:
""" + "\n".join(
    f"  House {h}: {', '.join(sigs)}" for h, sigs in CHART["house_sig"].items()
) + """

Planet Significators:
""" + "\n".join(
    f"  {p}: houses {sigs}" for p, sigs in CHART["planet_sig"].items()
)

# ── Test questions ────────────────────────────────────────────────────────────
TESTS = [
    # ── Marriage / 7th house (M1-M6) ─────────────────────────────────────────
    {
        "id": "M1", "category": "Marriage", "weight": 1.0,
        "expected_format": '{"result":"Yes/No","explanation":"...","rule":"KP_MAR_xxxx","confidence":"high/medium/low"}',
        "question": f"""Given the following chart, analyze the 7th cusp sub-lord for marriage prospects.

{CHART_CONTEXT}

The 7th cusp sub-lord is VEN. Venus is placed at 282-37-46, in SAT's rashi, MON's nakshatra, with sub-lord MAR.
Venus signifies houses: 1, 4, 6, 9, 12.

According to KP principles:
1. Will marriage occur for this native? (Yes/No)
2. What is the role of the 7th cusp sub-lord (VEN) in determining marriage?
3. Is VEN connected to houses 2, 7, 11 (favorable for marriage)?
4. Is VEN connected to houses 1, 6, 10, 12 (unfavorable)?
5. Cite the specific KP rule and give your verdict with confidence level.

Return your answer in this JSON format:
{{"result":"Yes/No","explanation":"...","rule":"KP_MAR_xxxx","confidence":"high/medium/low"}}""",
    },
    {
        "id": "M2", "category": "Marriage", "weight": 1.0,
        "expected_format": '{"timing":"...","quote":"..."}',
        "question": f"""Using this chart data:

{CHART_CONTEXT}

If the 7th cusp sub-lord (VEN) were retrograde, what would KP astrology predict?
1. Would it indicate delay, denial, or eventual marriage?
2. What are the specific timing conditions (dasha/transit) for marriage when the sub-lord is retrograde?
3. In which Mahadasha-Antardasha period would marriage be most likely?
4. Quote the specific KP rule for retrograde sub-lord effects on marriage.

Format: {{"timing":"delay until X dasha","conditions":"...","rule":"..."}}""",
    },
    {
        "id": "M3", "category": "Marriage", "weight": 0.9,
        "expected_format": "explanation quoting book rule",
        "question": f"""Chart data:

{CHART_CONTEXT}

The 7th cusp sub-lord is VEN. Venus signifies houses 1, 4, 6, 9, 12.
Suppose VEN also signified houses 1, 7, and 9 simultaneously.

According to KP astrology:
1. What effect does a sub-lord being significator for 1st, 7th, and 9th houses have on marriage timing/outcome?
2. Does the 9th house connection (dharma/fortune) help or complicate marriage?
3. Does the 1st house connection (self) strengthen or weaken the marriage indication?
4. Cite the specific KP rule and explain step by step.""",
    },
    {
        "id": "M4", "category": "Marriage", "weight": 1.0,
        "expected_format": "single planet chosen + step-by-step elimination",
        "question": f"""Chart data:

{CHART_CONTEXT}

House 7 significators are: MER, SUN.
Additional candidates through connections: VEN (7th sub-lord), JUP (2nd house sig), MAR (11th house sig).

Multiple planets qualify as 7th house significators for marriage timing.
According to KP astrology:
1. Which planet should be selected as the PRIMARY significator for marriage timing?
2. What is the step-by-step selection/elimination algorithm per KP rules?
3. How do you rank: cusp sub-lord vs house occupant vs house lord vs star-lord?
4. Give the final choice with reasoning and rule citation.""",
    },
    {
        "id": "M5", "category": "Marriage", "weight": 0.8,
        "expected_format": "delay/deny/complication + rule quote",
        "question": f"""Chart data:

{CHART_CONTEXT}

Hypothetical scenario: Mars is retrograde AND debilitated (in Cancer).
Mars is the nakshatra lord of the 7th cusp. Mars signifies houses 1, 3, 10, 11, 12.

According to KP astrology:
1. What is the effect of a retrograde + debilitated planet as nakshatra lord of the 7th cusp on marriage?
2. Does KP consider debilitation differently from Vedic astrology?
3. Is the outcome delay, denial, or complication?
4. Quote the specific KP rule for retrograde debilitated planets.""",
    },
    {
        "id": "M6", "category": "Marriage", "weight": 0.8,
        "expected_format": "list of stated reasons from KP",
        "question": f"""Chart data:

{CHART_CONTEXT}

Scenario: The native has had 5 broken engagements over the past 8 years.
The 7th cusp sub-lord is VEN, signifying houses 1, 4, 6, 9, 12.
Note: VEN signifies the 6th house (disputes/separation) and 12th house (losses).

According to KP astrology:
1. What is the likely explanation for repeated failed engagements?
2. Which house connections of the 7th sub-lord cause broken engagements?
3. What role does the 6th house signification play?
4. What role does the 12th house signification play?
5. Is there any remedy or timing when marriage could succeed?
6. Cite specific KP rules.""",
    },

    # ── Venus & relationships (V1-V2) ────────────────────────────────────────
    {
        "id": "V1", "category": "Venus", "weight": 0.8,
        "expected_format": "verdict + book quote",
        "question": f"""Chart data:

{CHART_CONTEXT}

Venus is at 282-37-46 in Aquarius (SAT's rashi), in Dhanishta nakshatra (MON's star), sub-lord MAR.
Venus signifies houses: 1, 4, 6, 9, 12.

According to KP astrology:
1. As the natural karaka for marriage, how does Venus's nakshatra-lord (Moon) affect marriage prospects?
2. Moon signifies houses 1, 3, 6, 10 — does this help or hinder Venus's role as marriage karaka?
3. Venus's sub-lord is Mars (houses 1, 3, 10, 11, 12) — what does this indicate?
4. Will Venus's placement predict a harmonious marriage?
5. Cite the KP rule for karaka analysis.""",
    },
    {
        "id": "V2", "category": "Venus", "weight": 0.7,
        "expected_format": "difference in interpretation + exact lines",
        "question": f"""According to KP astrology, what is the difference in interpretation when Venus is:
(a) Placed in the 5th bhava (love affairs, romance, courtship)
(b) Placed in the 7th bhava (marriage, partnership, spouse)

For each case:
1. What does Venus signify differently?
2. How does the sub-lord of Venus modify the result?
3. When does 5th house Venus lead to marriage vs just love affairs?
4. Cite the specific KP rules for Venus in 5th vs 7th.""",
    },

    # ── Horary / Ruling planets (H1-H2) ──────────────────────────────────────
    {
        "id": "H1", "category": "Horary", "weight": 0.9,
        "expected_format": "list of ruling planets + method trace",
        "question": f"""A querent selects horary number 147 for a marriage question.

According to KP horary method:
1. How do you determine the lagna degree from number 147?
2. What are the ruling planets for this horary number?
3. Step by step, show: ascendant sign lord, ascendant star lord, ascendant sub-lord, Moon sign lord, Moon star lord, day lord.
4. How do these ruling planets confirm or deny the marriage event?
5. Cite the KP horary method rules.""",
    },
    {
        "id": "H2", "category": "Horary", "weight": 0.8,
        "expected_format": "tie-breaker rules + which planets to drop",
        "question": f"""In a KP horary analysis, the following significators are found for the 7th house (marriage):
- Sun: occupant of 7th, star-lord of 7th cusp
- Mercury: significator of 7th through sub-lord connection
- Venus: natural karaka, 7th cusp sub-lord
- Saturn: connected to 7th through depositor chain
- Rahu: in the star of 7th occupant

These are conflicting — some are benefic for marriage, some are malefic.
According to KP:
1. What are the tie-breaker rules to select the most relevant significators?
2. Which planets should be dropped and why?
3. What is the order of priority: sub-lord > star-lord > sign-lord?
4. Cite the specific KP rules for significator selection.""",
    },

    # ── Financial / 11th house (F1-F2) ───────────────────────────────────────
    {
        "id": "F1", "category": "Financial", "weight": 0.9,
        "expected_format": '{"gains_likely":"Yes/No","timing":"...","rule":"..."}',
        "question": f"""Chart data:

{CHART_CONTEXT}

The 11th cusp sub-lord is MAR. Mars signifies houses: 1, 3, 10, 11, 12.
Mars is connected to the 11th house (gains) and also to the 10th (profession).

According to KP astrology:
1. Will the native experience financial gains? (Yes/No)
2. Mars signifies 11th (gains) but also 12th (losses) — how to interpret this conflict?
3. What are the timing conditions (dasha period) for gains?
4. What is the role of the 2nd house (accumulated wealth) connection?
5. Cite the specific KP rule for 11th cusp sub-lord analysis.""",
    },
    {
        "id": "F2", "category": "Financial", "weight": 0.8,
        "expected_format": "explanation for blocked gains + exact quote",
        "question": f"""Chart data:

{CHART_CONTEXT}

The 11th cusp sub-lord is MAR, signifying houses 1, 3, 10, 11, 12.
Note: MAR also signifies the 12th house (expenses, losses, foreign settlement).

The 8th cusp sub-lord is RAH, signifying houses 3, 10, 12.

According to KP astrology:
1. When the 11th sub-lord is connected to the 12th house, what happens to financial gains?
2. When the 8th sub-lord is connected to the 12th house, what does this indicate?
3. Are gains blocked, delayed, or redirected (e.g., gains through foreign sources)?
4. Cite the specific KP rules for 11th-12th and 8th-12th connections.""",
    },

    # ── Dasha / Timing (T1-T2) ───────────────────────────────────────────────
    {
        "id": "T1", "category": "Timing", "weight": 1.0,
        "expected_format": "quote + Yes/No",
        "question": f"""Chart data:

{CHART_CONTEXT}

For marriage prediction, the relevant houses are 2, 7, 11.
The 7th cusp sub-lord is VEN.

According to KP astrology:
1. Does the KP system require BOTH the Mahadasha lord AND Antardasha lord to be connected to houses 2, 7, 11 for marriage to occur?
2. Or is it sufficient if only one of them is connected?
3. What is the exact rule text from KP regarding dasha-bhukti requirements for event manifestation?
4. In this chart, which Mahadasha-Antardasha combination would trigger marriage?
5. Cite the exact KP rule.""",
    },
    {
        "id": "T2", "category": "Timing", "weight": 0.9,
        "expected_format": "expected outcome/timing",
        "question": f"""Chart data:

{CHART_CONTEXT}

Scenario: The native is running Jupiter Mahadasha.
Jupiter signifies houses: 2, 5, 11, 12.
Jupiter IS connected to houses 2 and 11 (favorable for marriage).

But the current Antardasha lord is Ketu.
Ketu signifies houses: 5, 6, 8, 12.
Ketu is NOT connected to houses 2, 7, or 11.

According to KP astrology:
1. When the Mahadasha lord is connected to marriage houses but the Antardasha lord is NOT, what happens?
2. Will marriage occur in this sub-period or be delayed to a more favorable Antardasha?
3. Which Antardasha within Jupiter Mahadasha would be most favorable for marriage?
4. Cite the specific KP timing rule.""",
    },

    # ── Transit interactions (TR1-TR2) ────────────────────────────────────────
    {
        "id": "TR1", "category": "Transit", "weight": 0.7,
        "expected_format": "stated effect on timing/outcome",
        "question": f"""Chart data:

{CHART_CONTEXT}

Scenario: Saturn (a natural malefic) is transiting over the 7th cusp sub-lord Venus's natal position (282-37-46, Aquarius).
This transit occurs during a favorable dasha period for marriage.

According to KP astrology:
1. Does the transit of a malefic over the sub-lord affect the timing of the event?
2. Does KP give importance to transits, or is the dasha system primary?
3. Will Saturn's transit over Venus delay or obstruct marriage even during a favorable dasha?
4. Cite the KP rule for transit effects.""",
    },
    {
        "id": "TR2", "category": "Transit", "weight": 0.7,
        "expected_format": "mitigation rules",
        "question": f"""Chart data:

{CHART_CONTEXT}

Scenario: Jupiter (natural benefic) is transiting over the 7th cusp degree (122-12-49, Leo).
The 7th cusp sub-lord VEN is weak (signifying 6th and 12th houses).
The dasha period is marginally favorable.

According to KP astrology:
1. Can a benefic transit (Jupiter over 7th cusp) rescue or improve the outcome when the sub-lord is weak?
2. Does KP give transit the power to override sub-lord indications?
3. What is the KP position on transit vs sub-lord primacy?
4. Cite the specific KP rule.""",
    },

    # ── Edge / complex cases (E1-E3) ─────────────────────────────────────────
    {
        "id": "E1", "category": "Edge", "weight": 0.8,
        "expected_format": "step-by-step evaluation + final verdict",
        "question": f"""Chart data:

{CHART_CONTEXT}

Complex case: For marriage (houses 2, 7, 11), the following significators are found:
- Sun: signifies 4, 7, 9, 11, 12 — connected to 7 and 11 (favorable) but also 12 (unfavorable)
- Mercury: signifies 5, 7, 8, 9, 11, 12 — connected to 7 and 11 but also 8 and 12
- Venus (7th sub-lord): signifies 1, 4, 6, 9, 12 — NOT directly connected to 7 or 11
- Jupiter: signifies 2, 5, 11, 12 — connected to 2 and 11 but also 12

There are 4+ conflicting significators, each with both favorable and unfavorable connections.

According to KP:
1. Step by step, how do you evaluate this chart for marriage?
2. Which significator takes priority?
3. The 7th sub-lord VEN does NOT signify 7 or 11 — what does this mean?
4. Final verdict: will marriage happen? When?
5. Cite the KP algorithm for conflicting significators.""",
    },
    {
        "id": "E2", "category": "Edge", "weight": 0.6,
        "expected_format": "special rules for water signs",
        "question": f"""According to KP astrology, are there special rules when:
1. The 7th cusp sub-lord is placed in a watery sign (Cancer, Scorpio, Pisces)?
2. The 7th cusp itself falls in a watery sign?
3. Does the element (fire/earth/air/water) of the sign affect marriage predictions in KP?
4. In this chart, the 7th cusp is at 122-12-49 (Leo, a fire sign) — does this have any special significance?
5. Cite any KP rules regarding sign classification effects on marriage.""",
    },
    {
        "id": "E3", "category": "Edge", "weight": 0.7,
        "expected_format": "mitigation or denial + quote",
        "question": f"""According to KP astrology, consider this scenario:
A planet is debilitated (e.g., Venus in Virgo) BUT placed in the nakshatra of a benefic planet (e.g., Jupiter's star).

1. Does KP consider debilitation the same way as Vedic astrology?
2. In KP, does the nakshatra lord override the debilitation status?
3. If the debilitated planet is the 7th cusp sub-lord, does the benefic nakshatra lord mitigate the negative effect?
4. What is KP's position: mitigation or denial?
5. Cite the specific KP rule for debilitated planets in benefic nakshatras.""",
    },

    # ── Quality / format checks (Q1-Q2) ──────────────────────────────────────
    {
        "id": "Q1", "category": "Quality", "weight": 0.5,
        "expected_format": "verbatim quote with page/section",
        "question": f"""In your previous answers about this chart, you cited rules like KP_MAR_0673, KP_MAR_0971, etc.

For rule citation fidelity:
1. What is the exact content of KP rule KP_MAR_0673?
2. What is the exact content of KP rule KP_MAR_0971?
3. What chapter/section of the KP Reader do these rules come from?
4. Are these rule IDs from KP Reader I, II, III, IV, V, or VI?
5. Provide the verbatim text of each rule.""",
    },
    {
        "id": "Q2", "category": "Quality", "weight": 0.5,
        "expected_format": "Yes/No + evidence",
        "question": f"""Evaluate the quality and provenance of KP rule citations:

1. When you cite a rule like "KP_MAR_0673", is this an actual rule ID from the KP Readers by Prof. K.S. Krishnamurti?
2. Or is this an internally generated identifier from your training data?
3. How confident are you in the accuracy of your rule citations?
4. Can you distinguish between rules from KP Reader I (general principles) vs KP Reader IV (marriage) vs KP Reader VI (horary)?
5. What is your confidence that the rule content you cite matches the original KP text?""",
    },
]


# ── Run tests ─────────────────────────────────────────────────────────────────
def run_test(test):
    """Send a test question to vLLM and collect the response."""
    messages = [
        {"role": "system", "content": (
            "You are an expert KP (Krishnamurti Paddhati) Astrology assistant. "
            "Always cite specific KP rules. Include confidence level. "
            "Use proper KP terminology. Explain reasoning step by step."
        )},
        {"role": "user", "content": test["question"]},
    ]
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model="kp-astrology-llama",
            messages=messages,
            max_tokens=1024,
            temperature=args.temperature,
            top_p=0.9,
        )
        elapsed = time.time() - t0
        answer = resp.choices[0].message.content.strip()
        tokens = resp.usage.total_tokens if resp.usage else 0
        return {
            "answer": answer,
            "latency_s": round(elapsed, 2),
            "tokens": tokens,
            "error": None,
        }
    except Exception as e:
        return {"answer": "", "latency_s": 0, "tokens": 0, "error": str(e)}


def main():
    print("=" * 80)
    print("KP ASTROLOGY MODEL TEST SUITE")
    print("=" * 80)
    print(f"Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"vLLM:        {args.vllm_url}")
    print(f"Tests:       {len(TESTS)}")
    print(f"Temperature: {args.temperature}")
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
        print(f"  Status: {status} | {res['latency_s']}s | {len(res['answer'])} chars")
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
        f.write(f"**Temperature:** {args.temperature}  \n\n")

        f.write("## Summary\n\n")
        f.write("| # | ID | Category | Weight | Latency | Rule Cited | Confidence | Score |\n")
        f.write("|---|-----|----------|--------|---------|------------|------------|-------|\n")
        for i, r in enumerate(results, 1):
            al = r["model_answer"].lower()
            rule = "Yes" if ("rule" in al or "kp_" in al) else "No"
            conf = "Yes" if "confidence" in al else "No"
            score = r["score"] if r["score"] is not None else "—"
            f.write(f"| {i} | {r['q_id']} | {r['category']} | {r['weight']} | "
                    f"{r['latency_s']}s | {rule} | {conf} | {score} |\n")

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
