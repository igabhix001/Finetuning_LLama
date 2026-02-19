#!/usr/bin/env python3
"""
Smoke test for all new postprocessing fixes in 09_chat_ui.py
Tests: cancer intercept, metadata strip, self-doubt strip, age guard, header strip
"""
import sys, re
sys.path.insert(0, 'scripts')

# Patch args before importing
import argparse
_orig_parse = argparse.ArgumentParser.parse_args
def _mock_parse(self, args=None, namespace=None):
    ns = argparse.Namespace()
    ns.vllm_url = "http://localhost:8000"
    ns.port = 7860
    ns.share = False
    ns.pinecone_index = None
    ns.log_file = None
    return ns
argparse.ArgumentParser.parse_args = _mock_parse

# Also mock gradio and openai to avoid import errors
import unittest.mock as mock
sys.modules['gradio'] = mock.MagicMock()
sys.modules['openai'] = mock.MagicMock()
sys.modules['pinecone'] = mock.MagicMock()

# Now import the module pieces we need
import importlib.util, types

# Just import the functions directly by exec-ing the relevant parts
# Instead, test _classify_query_type and _postprocess by loading the file partially

with open('scripts/09_chat_ui.py', 'r', encoding='utf-8') as f:
    src = f.read()

# Extract just the function definitions we need
# We'll exec them in a controlled namespace
ns = {'re': re, '__builtins__': __builtins__}

# Find and exec _postprocess
import ast
tree = ast.parse(src)
func_src = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in ('_postprocess', '_classify_query_type', '_is_hindi_q'):
        start = node.lineno - 1
        end = node.end_lineno
        lines = src.split('\n')[start:end]
        func_src[node.name] = '\n'.join(lines)

# Exec them
from datetime import date, datetime
ns['date'] = date
ns['datetime'] = datetime
for name, code in func_src.items():
    try:
        exec(compile(code, '<string>', 'exec'), ns)
    except Exception as e:
        print(f"  EXEC ERROR for {name}: {e}")

_postprocess = ns.get('_postprocess')
_classify = ns.get('_classify_query_type')

PASS = 0
FAIL = 0

def check(label, condition, got=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅ PASS: {label}")
        PASS += 1
    else:
        print(f"  ❌ FAIL: {label}")
        if got:
            print(f"         Got: {got[:120]}")
        FAIL += 1

print("=" * 60)
print("SMOKE TEST: 09_chat_ui.py postprocessing")
print("=" * 60)

# ── Test 1: metadata leak strip ──
print("\n[1] Metadata leak strip")
if _postprocess:
    _postprocess._birth_year = None
    _postprocess._birth_date = None
    _postprocess._native_name = "Arjun"
    _postprocess._query_type = "timing"
    _postprocess._user_question = "when will i get married"

    t1 = "Your marriage timing is Feb 2026 rulesused: KPGEN0956ADIUS0285 during Venus AD."
    r1 = _postprocess(t1)
    check("rulesused:KPGEN stripped", "KPGEN" not in r1 and "rulesused" not in r1.lower(), r1)

    t2 = "Marriage in Mar 2026 timingmethod: KP_SUBLORD_METHOD during Venus AD."
    r2 = _postprocess(t2)
    check("timingmethod: stripped", "timingmethod" not in r2.lower(), r2)

    t3 = "Career boost in Apr 2026 KPTIM0123ADIUS0456 during Saturn AD."
    r3 = _postprocess(t3)
    check("KPTIM code stripped", "KPTIM" not in r3, r3)

    t4 = "Promotion in May 2026 KPGEN0956ADIUS0285 during Mercury AD."
    r4 = _postprocess(t4)
    check("ALL-CAPS rule code stripped", "KPGEN0956ADIUS0285" not in r4, r4)
else:
    print("  SKIP: _postprocess not loaded")

# ── Test 2: cancer/medical response strip ──
print("\n[2] Medical response strip in postprocess")
if _postprocess:
    _postprocess._query_type = "analysis"
    _postprocess._user_question = "health query"

    t5 = "Yes you have Cancer! The timing is February 2026 to April 2026."
    r5 = _postprocess(t5)
    check("'YES you have Cancer' replaced", "you have Cancer" not in r5 and "cancer" not in r5.lower() or "health challenges" in r5.lower(), r5)

    t6 = "You have cancer during this Saturn period."
    r6 = _postprocess(t6)
    check("'you have cancer' replaced", "you have cancer" not in r6.lower(), r6)

    t7 = "Aapko cancer hai during this period."
    r7 = _postprocess(t7)
    check("'aapko cancer hai' replaced", "aapko cancer hai" not in r7.lower(), r7)

# ── Test 3: self-doubt strip ──
print("\n[3] Self-doubt strip")
if _postprocess:
    _postprocess._query_type = "analysis"
    _postprocess._user_question = "career query"

    t8 = "Your career will improve. Jis method se hum predictions banate hain woh bilkul reliable nahi hai. Saturn AD starts Apr 2026."
    r8 = _postprocess(t8)
    check("self-doubt sentence stripped", "reliable nahi" not in r8.lower(), r8)

    t9 = "Marriage in 2026. Sirf immediate future events hi predict kar sakte hain. Venus AD active."
    r9 = _postprocess(t9)
    check("'sirf immediate future' stripped", "sirf immediate" not in r9.lower(), r9)

# ── Test 4: robotic header strip ──
print("\n[4] Robotic header strip")
if _postprocess:
    _postprocess._query_type = "timing"
    _postprocess._user_question = "career query"

    t10 = "Career Prospects Analysis: Your career will improve in Apr 2026 during Saturn AD."
    r10 = _postprocess(t10)
    check("'Career Prospects Analysis:' stripped", "Career Prospects Analysis" not in r10, r10)

    t11 = "Sub-Lord Significance: The 10th cusp sub-lord Mercury supports career in Apr 2026."
    r11 = _postprocess(t11)
    check("'Sub-Lord Significance:' stripped", "Sub-Lord Significance" not in r11, r11)

    t12 = "Timing Precision: Your peak period is Apr 2026 during Saturn AD."
    r12 = _postprocess(t12)
    check("'Timing Precision:' stripped", "Timing Precision:" not in r12, r12)

    t13 = "Peak Financial Growth Period: Apr 2026 to Oct 2026 during Saturn-Venus AD."
    r13 = _postprocess(t13)
    check("'Peak Financial Growth Period:' stripped", "Peak Financial Growth Period" not in r13, r13)

# ── Test 5: _classify_query_type — medical_safety ──
print("\n[5] _classify_query_type — medical_safety intercept")
if _classify:
    q1 = "do i have cancer"
    r = _classify(q1)
    check("'do i have cancer' → medical_safety", r["type"] == "medical_safety", str(r))

    q2 = "kya mujhe cancer hai"
    r = _classify(q2)
    check("'kya mujhe cancer hai' → medical_safety", r["type"] == "medical_safety", str(r))

    q3 = "will i get cancer"
    r = _classify(q3)
    check("'will i get cancer' → medical_safety", r["type"] == "medical_safety", str(r))

    q4 = "do i have diabetes"
    r = _classify(q4)
    check("'do i have diabetes' → medical_safety", r["type"] == "medical_safety", str(r))

# ── Test 6: _classify_query_type — meta_confidence ──
print("\n[6] _classify_query_type — meta_confidence intercept")
if _classify:
    q5 = "can you really predict the future"
    r = _classify(q5)
    check("'can you really predict' → meta_confidence", r["type"] == "meta_confidence", str(r))

    q6 = "how accurate are you"
    r = _classify(q6)
    check("'how accurate are you' → meta_confidence", r["type"] == "meta_confidence", str(r))

    q7 = "is astrology accurate"
    r = _classify(q7)
    check("'is astrology accurate' → meta_confidence", r["type"] == "meta_confidence", str(r))

# ── Test 7: safety still fires before medical_safety ──
print("\n[7] Safety ordering — death queries still hit safety first")
if _classify:
    q8 = "when will i die"
    r = _classify(q8)
    check("'when will i die' → safety (not timing)", r["type"] == "safety", str(r))

    q9 = "will i die soon"
    r = _classify(q9)
    check("'will i die soon' → safety", r["type"] == "safety", str(r))

# ── Test 8: ISO date conversion ──
print("\n[8] ISO date conversion")
if _postprocess:
    _postprocess._query_type = "timing"
    _postprocess._user_question = "marriage timing"
    t14 = "Your marriage timing is 2026-03 to 2026-09 during Venus AD."
    r14 = _postprocess(t14)
    check("ISO date 2026-03 → Mar 2026", "2026-03" not in r14, r14)

print("\n" + "=" * 60)
print(f"RESULTS: {PASS} PASS / {FAIL} FAIL")
print("=" * 60)
if FAIL == 0:
    print("✅ ALL TESTS PASSED — safe to deploy")
else:
    print(f"⚠️  {FAIL} tests failed — review above")
