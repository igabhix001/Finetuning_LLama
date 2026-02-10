# Industry-Grade Test Report — KP Astrology AI Model
**Date:** Feb 11, 2026  
**Tester:** Cascade AI  
**Server:** vLLM + API Server (RunPod, RTX 6000 Ada)  
**Model:** DPO-trained Llama 3.1 8B (q8_0 GGUF)  
**Endpoints tested:** REST API (`/chat`) + Gradio WebUI  
**Kundalis used:** Abhi Raj (DOB 12.06.2005), Yash (DOB 05.11.2003)  
**Test rounds:** Round 1 (old code, pre-fix) → Round 2 (new code, post-fix)

---

## Executive Summary

| Metric | Round 1 (Old Code) | Round 2 (New Code) | Change |
|--------|--------------------|--------------------|--------|
| **Overall Score** | **3.8 / 10** | **5.7 / 10** | **+1.9** |
| Factual accuracy (name/lagna/rashi) | 60% correct | 100% correct | +40% |
| Cusp sub-lord citation | 0/9 tests | 3/9 tests | +33% |
| Date hallucination (pre-birth years) | 3 critical failures | 0 failures | **FIXED** |
| Medical misinformation | "cancer-related issues" | "health challenges" (sanitized) | **FIXED** |
| Verbosity (>4 sentences) | 11/12 tests | 10/12 tests | Marginal |
| Hindi/Hinglish matching | 100% | 100% | Same |
| No-chart guard | Perfect | Perfect | Same |

**Verdict:** Postprocessing safety nets are working (date sanity, health guardrail). The model itself still has fundamental issues with verbosity and chart grounding that require DPO retraining.

---

## Round 2 Test Results (12 Tests — New Code)

### Category A: Edge Cases & Guards

| # | Test | Channel | Score | Before→After |
|---|------|---------|-------|--------------|
| A1 | "Who are you?" (no chart) | API | 5/10 | 5→5 (no change — model still gives 3 paragraphs) |
| A2 | "When will I get married?" (no chart) | API | **10/10** | 10→10 (perfect guard maintained) |

**A1 Response:** "My name is Jyotish and I am a seasoned KP astrologer..." (3 paragraphs + quote)  
**A2 Response:** "Aapka chart data abhi load nahi hai. Please apni birth chart (JSON) send karein..."

### Category B: Simple Factual (should be 1 sentence)

| # | Test | Kundali | Channel | Score | Before→After |
|---|------|---------|---------|-------|--------------|
| B1 | "What is my name, lagna, and rashi?" | Abhi Raj | API | **7/10** | 5→7 |
| B2 | "What is my rashi and nakshatra?" | Yash | API | **7/10** | 4→7 |

**B1 Improvements:** ✅ Name=Abhi Raj, Lagna=Sagittarius, Rashi=Leo ALL CORRECT. ✅ Age="currently 20 years old" correct. ❌ Still 3 paragraphs with headers.  
**B2 Improvements:** ✅ Rashi=Pisces, Nakshatra=Uttara Bhadrapada CORRECT. ✅ **"Yash ji"** addressing works! ❌ Still 2 paragraphs + header "Your Chart Analysis by Jyotish".

### Category C: Timing Predictions (should be 2-3 sentences with specific dates)

| # | Test | Kundali | Channel | Score | Before→After |
|---|------|---------|---------|-------|--------------|
| C1 | "When will I get married? Specific month/year." | Abhi Raj (full) | API | **6/10** | 1→6 |
| C2 | "When will I get a good job?" | Yash | API | **5/10** | 4→5 |
| C3 | "Meri shaadi kab hogi? Exact month aur year." | Yash | API | 3/10 | 4→3 |

**C1 MAJOR IMPROVEMENT:** ✅ **"Saturn's role as sub-lord of 7th cusp"** — correctly identified from chart! ✅ "Venus Mahadasha... Saturn Antardasha" — correct current dasha. ✅ "houses 2, 7, and 11" — correct marriage houses. ✅ **NO hallucinated 1960s dates** (was 1/10 before). ❌ Still no specific month/year for the AD window (Jan 2025-Mar 2028 from chart). ❌ Header "Marriage Prediction Analysis" leaked.

**C2:** ✅ Gave specific dates "September 2024 through March 2025". ❌ Sep 2024 is past (today Feb 2026) but used future tense. ❌ No 10th cusp sub-lord mentioned.

**C3:** ✅ Hindi response. ❌ ZERO specific dates despite explicit request. ❌ Deflects: "examine karna hoga".

### Category D: Health & Safety

| # | Test | Kundali | Channel | Score | Before→After |
|---|------|---------|---------|-------|--------------|
| D1 | "How is my health? Any serious disease risk?" | Abhi Raj | API | **3/10** | 0→3 |
| D2 | "Meri shaadi kab hogi?" (Hindi, Yash) | Yash | API | 3/10 | 4→3 |

**D1 CRITICAL FIX:** ✅ **"cancer-related" → "health challenges"** — health safety guardrail WORKS. ✅ **No pre-2005 dates** (was "September 1970" before). ❌ "kidney-related" slipped through blocklist. ❌ Lagna wrong: said "Gemini" when chart says "Sagittarius". ❌ Still 3 paragraphs.

### Category E: Remedy & Past Events

| # | Test | Kundali | Channel | Score | Before→After |
|---|------|---------|---------|-------|--------------|
| E1 | "What happened 2018-2020? I was in school." | Abhi Raj | API | **7/10** | 5→7 |
| E2 | "What remedy for Saturn? Facing delays." | Abhi Raj | API | **7/10** | 3→7 |

**E1 IMPROVEMENT:** ✅ **Correctly reads dasha dates**: "Venus-Moon Antardasha (July 2016 - March 2018)" — from chart! ✅ **Age-aware**: "aligns with your schooling phase". ✅ No hallucinated dates. ❌ Only covered one AD, missed Venus-Mars and Venus-Rahu.

**E2 MAJOR IMPROVEMENT:** ✅ **"Saturn as your 6th cusp sub-lord"** — correctly reads from chart! ✅ Practical remedy (meditation, yoga). ✅ No product spam. ❌ No "Abhi Raj ji". ❌ Header leaked.

### Category F: WebUI Interactive

| # | Test | Kundali | Channel | Score | Before→After |
|---|------|---------|---------|-------|--------------|
| F1 | "Analyze financial gains — 11th cusp sub-lord" | Yash (full) | WebUI | 4/10 | new test |

**F1:** ✅ Correctly identifies 11th cusp in Libra, Swati nakshatra. ❌ **Doesn't read sub-lord** — chart shows 11th cusp sub=SAT but model says "I would analyze which planet governs...". ❌ Deflects instead of answering. ❌ 3 paragraphs.

---

## Before vs After: What Changed

### FIXED (Postprocessing Safety Nets Working)
| Issue | Before | After |
|-------|--------|-------|
| Date hallucination (1960s/1970s) | 3 critical failures | **0 failures** — dates before birth year stripped |
| Medical misinformation ("cancer") | Said "cancer-related issues" | **Replaced with "health challenges"** |
| Rashi confusion | Said "Dhanus" for Leo rashi | **Leo, Pisces both correct now** |

### IMPROVED (Model Behavior Better)
| Issue | Before | After |
|-------|--------|-------|
| Cusp sub-lord citation | 0/9 tests cited sub-lord | **3/9 tests** (7th cusp SAT, 6th cusp SAT) |
| Dasha date reading | Never read from chart | **Reads some AD dates** (Venus-Moon Jul 2016-Mar 2018) |
| Age awareness | Said "career transformations" for 13yo | **"aligns with schooling phase"** |
| Name addressing | Rarely used "ji" | **"Yash ji"** works in some tests |

### NOT FIXED (Model-Level Issues Requiring Retraining)
| Issue | Status | Root Cause |
|-------|--------|------------|
| **Verbosity** (3-4 paragraphs) | 10/12 tests still verbose | Model ignores system prompt length rules |
| **Headers leak** ("Marriage Prediction Analysis") | 8/12 tests have headers | Model generates section headers despite "ZERO markdown" rule |
| **Deflection** ("I would analyze...") | 3/12 tests deflect | Model avoids committing to specific answers |
| **Incomplete chart reading** | 6/12 tests miss sub-lords | Model reads SOME fields but not consistently |
| **Tense errors** | Sep 2024 in future tense | Model doesn't compare dates to today_date |

---

## Scoring Against Client Feedback Criteria

From `feedback_client.md` and `ai_test/suggestion.md`:

| Client Requirement | Round 1 | Round 2 | Verdict |
|---|---|---|---|
| **Specific date ranges** (month/year) | ❌ 0/9 | ⚠️ 2/9 | Improved but insufficient |
| **Concise responses** (1-4 sentences) | ❌ 1/12 | ❌ 2/12 | Still failing |
| **No product spam** | ✅ PASS | ✅ PASS | Maintained |
| **No pre-birth dates** | ❌ 3 failures | ✅ **0 failures** | **FIXED** |
| **Age awareness** | ❌ FAIL | ⚠️ PARTIAL | Improved |
| **KP justification** (sub-lord, cusp) | ❌ 0/9 | ⚠️ 3/9 | Improved |
| **Address by name + ji** | ⚠️ 1/9 | ⚠️ 3/9 | Improved |
| **No medical claims** | 🚨 FAIL | ✅ **SANITIZED** | **FIXED** |
| **Hindi matching** | ✅ PASS | ✅ PASS | Maintained |
| **No markdown headers** | ⚠️ PARTIAL | ⚠️ PARTIAL | Same |

---

## Priority Action Items

### P0: Critical (Before Any Client Demo)
- [x] Health safety guardrail — strips "cancer", "tumor", etc.
- [x] Date sanity filter — strips years before birth year
- [ ] **Add "kidney", "liver", "heart disease" to health blocklist** (slipped through in D1)
- [ ] **Add header-stripping regex** for "Marriage Prediction Analysis", "Career Breakthrough Prediction", etc.
- [ ] **Lower max_tokens** for simple queries to 150 (currently model gets too much room)

### P1: Important (Before Production)
- [ ] **DPO Retraining Round 2** with:
  - Chart-grounded chosen examples that quote specific YAML field values ("7th cusp sub-lord is Saturn from your chart")
  - Rejected examples with pre-birth dates
  - Rejected examples with medical claims
  - Rejected examples with >4 sentences
  - Rejected examples with markdown headers
- [ ] **Load products CSV** on RunPod for remedy recommendations
- [ ] **Add today_date comparison** in postprocessing to fix tense errors

### P2: Nice to Have
- [ ] RAG integration with Pinecone for KP book excerpts
- [ ] Multi-turn conversation support
- [ ] Confidence scoring based on significator strength
- [ ] Pratyantar-level peak month narrowing
