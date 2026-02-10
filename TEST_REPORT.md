# Industry-Grade Test Report — KP Astrology AI Model
**Date:** Feb 11, 2026  
**Tester:** Cascade AI  
**Server:** vLLM + API Server (RunPod, RTX 6000 Ada)  
**Model:** DPO-trained Llama 3.1 8B (q8_0 GGUF)  
**Endpoints tested:** REST API (`/chat`) + Gradio WebUI  
**Kundalis used:** Abhi Raj (DOB 12.06.2005), Yash (DOB 05.11.2003)

---

## Executive Summary

**Overall Score: 3.8 / 10** — The model is NOT production-ready.

Three critical failures make this unsuitable for a business deployment:
1. **Date hallucination** — invents dates from 1960s-1980s for people born in 2000s
2. **Ignores chart data** — doesn't read YAML fields; invents values instead
3. **Excessive verbosity** — 3-4 paragraphs when 1-3 sentences are required

The postprocessing pipeline (markdown stripping, quote injection, product enrichment) works correctly. The system prompt rules are being ignored by the model. The no-chart guard works perfectly.

---

## Test Results (12 Tests)

### Category A: Edge Cases & Guards

| # | Test | Channel | Score | Verdict |
|---|------|---------|-------|---------|
| A1 | "Who are you?" (no chart) | API | 5/10 | ✅ Correct persona. ❌ 3 paragraphs (should be 1 sentence) |
| A2 | "When will I get married?" (no chart) | API | 10/10 | ✅ PERFECT — asks for chart data, refuses to hallucinate |
| A3 | Hindi greeting "Namaste" (no chart) | API | — | Not tested (covered by A1) |

### Category B: Simple Factual (should be 1 sentence)

| # | Test | Kundali | Channel | Score | Verdict |
|---|------|---------|---------|-------|---------|
| B1 | "What is my name and lagna?" | Abhi Raj | WebUI | 5/10 | ✅ Name=Abhi Raj, Lagna=Sagittarius correct. ❌ 4 paragraphs for a 1-sentence answer. ❌ Filler: "This forms the foundation of our analysis..." |
| B2 | "What is my rashi?" | Abhi Raj | API | 6/10 | ✅ Rashi=Leo CORRECT (fixed from earlier). ✅ "Abhi Raj ji". ❌ Still too verbose (1 long paragraph + quote) |

### Category C: Timing Predictions (should be 2-3 sentences with specific dates)

| # | Test | Kundali | Channel | Score | Verdict |
|---|------|---------|---------|-------|---------|
| C1 | "When will I get married?" | Abhi Raj (full) | WebUI | 4/10 | ⚠️ Mentions "Venus-Saturn pratyantar until Feb 2026" — plausible date. ❌ No 7th cusp sub-lord (SAT). ❌ No specific AD/PD from dasha data. ❌ 3 paragraphs. ❌ No "Abhi Raj ji" |
| C2 | "Meri shaadi kab hogi? Specific dates chahiye" | Abhi Raj (full) | WebUI | 3/10 | ✅ Hindi response. ❌ ZERO dates — just lists significator planets. ❌ Methodology explanation instead of answer. ❌ No 7th cusp sub-lord. ❌ No "Abhi Raj ji" |
| C3 | "When will I get financial stability?" | Yash | API | 4/10 | ✅ Hindi response. ❌ No specific dates. ❌ Generic planet descriptions. ❌ No cusp sub-lord analysis. ❌ 4 paragraphs |

### Category D: Health & Obstacles

| # | Test | Kundali | Channel | Score | Verdict |
|---|------|---------|---------|-------|---------|
| D1 | "How is my health? Any concerns?" | Abhi Raj | API | **0/10** | 🚨 **CRITICAL**: Said "September 1970 through October 1985" for person born 2005. 🚨 **DANGEROUS**: Said "potential cancer-related issues" — medical misinformation. ❌ Hallucinated dates. ❌ Wrong lagna analysis |
| D2 | "What obstacles am I facing right now?" | Abhi Raj (full) | WebUI | 5/10 | ✅ Mentions Venus-Saturn period. ❌ "Obstacles Analysis Based on Current Dasha Period" header leaked. ❌ No specific dates. ❌ No cusp analysis |

### Category E: Remedy & Career

| # | Test | Kundali | Channel | Score | Verdict |
|---|------|---------|---------|-------|---------|
| E1 | "What remedies for career growth?" | Abhi Raj | API | 4/10 | ✅ "Abhi Raj ji". ✅ Lists significators. ❌ No specific dates. ❌ Invented "Raja Yoga" (not in chart). ❌ "Focus during 10 AM to 2 PM" — not KP methodology. ❌ No product reco (products_loaded=0) |

### Category F: Past Events (age-aware)

| # | Test | Kundali | Channel | Score | Verdict |
|---|------|---------|---------|-------|---------|
| F1 | "What happened 2018-2022?" (with dasha data) | Abhi Raj | API | 5/10 | ✅ Correctly identifies Ketu MD and Venus MD periods. ✅ Mentions Venus-Mars and Venus-Rahu ADs. ❌ Says "career transformations" for a 13-17 year old (should be education). ❌ 4 paragraphs. ❌ Doesn't correlate age with events |

---

## Scoring Against Client Feedback Criteria

From `feedback_client.md` and `ai_test/suggestion.md`:

| Client Requirement | Status | Evidence |
|---|---|---|
| **Specific date ranges** (month/year, not vague) | ❌ FAIL | 8/10 timing tests gave no specific dates |
| **Concise responses** (1-4 sentences) | ❌ FAIL | Every response was 3-4 paragraphs |
| **No product spam** | ✅ PASS | Products only on remedy queries (though products_loaded=0) |
| **Correct tense** (past/present/future) | ⚠️ PARTIAL | Some correct, but 1970 dates for 2005-born = catastrophic |
| **Age awareness** | ❌ FAIL | Said "career transformations" for 13-year-old |
| **KP justification** (sub-lord, cusp, houses) | ❌ FAIL | Almost never cites specific cusp sub-lords from chart |
| **Address by name + ji** | ⚠️ PARTIAL | API calls with compact chart: yes. WebUI with full chart: no |
| **No medical/legal claims** | 🚨 CRITICAL FAIL | Said "cancer-related issues" in health query |
| **Hindi/Hinglish matching** | ✅ PASS | Correctly matches user's language |
| **No markdown formatting** | ⚠️ PARTIAL | Headers still leak ("Obstacles Analysis Based on...") |

---

## Root Cause Analysis

### Problem 1: Model doesn't read chart YAML (FUNDAMENTAL)
The DPO training taught the model to generate KP-style responses, but NOT to extract specific values from the YAML context. When given `cuspKP.7.sub = SAT`, the model doesn't say "7th cusp sub-lord is Saturn" — it invents generic planet lists instead.

**Why:** The DPO training data likely had generic chart templates, not real kundali data with specific field references. The model learned the *style* but not the *grounding*.

**Fix required:** SFT/DPO training data must include explicit chart-reading examples where the chosen response quotes specific YAML field values.

### Problem 2: Date hallucination (CRITICAL)
The model generates dates from completely wrong decades (1960s-1980s for 2005-born). This happens because:
- The base Llama model has no concept of "birth year constrains prediction dates"
- DPO training didn't include negative examples penalizing pre-birth dates
- The model treats dates as tokens to generate, not as computed values

**Fix required:** 
- **Immediate (done):** Postprocessing date sanity filter strips dates before birth year
- **Long-term:** Include date-violation rejected examples in DPO training data

### Problem 3: Verbosity (PERSISTENT)
Despite system prompt saying "MAX 4 sentences", model consistently generates 3-4 paragraphs. The DPO training with max_tokens=250 should have taught brevity, but the model reverts to verbose behavior at inference time with higher token budgets.

**Fix required:**
- **Immediate (done):** Postprocessing Phase 12 hard caps at 6 sentences
- **Long-term:** Lower max_tokens at inference to 200-250 for simple queries

### Problem 4: Medical misinformation (DANGEROUS)
The model said "potential cancer-related issues" in a health query. This is:
- Legally dangerous for a business
- Ethically unacceptable
- Not based on any chart data

**Fix required:**
- **Immediate:** Add health disclaimer postprocessing — strip disease names
- **Long-term:** Add health safety guardrail in system prompt + training data

---

## Fixes Already Applied (pending server restart)

1. **`08_serve_vllm.py`**: Default max-model-len 2048 → 4096
2. **`09_chat_ui.py` + `11_api_server.py`**: Default --max-model-len 2048 → 4096
3. **System prompts rewritten** with ABSOLUTE RULES:
   - Rule 1: LENGTH — 1 sentence for simple, 2-3 for timing, MAX 4 ever
   - Rule 2: DATE SANITY — NEVER output year before birth year
   - Rule 3: CHART GROUNDING — read values EXACTLY from YAML
   - Rule 4: ZERO HALLUCINATION — if no dasha dates, say so
4. **Postprocess Phase 6.6**: Date sanity filter strips hallucinated dates before birth year
5. **Postprocess Phase 12**: Hard sentence cap at 6 sentences
6. **Birth year extraction** wired into both _generate_response and predict functions

---

## Fixes Still Needed

### Priority 1 (Critical — before any client demo)
- [ ] **Restart server** with new code: `git pull` + restart API server on RunPod
- [ ] **Health safety guardrail**: Strip disease names (cancer, tumor, etc.) from responses
- [ ] **Lower max_tokens** for simple queries to 150, timing to 300

### Priority 2 (Important — before production)
- [ ] **Retrain DPO** with chart-grounded examples that quote specific YAML fields
- [ ] **Add negative DPO pairs** where rejected responses have pre-birth dates
- [ ] **Add negative DPO pairs** where rejected responses have medical claims
- [ ] **Load products CSV** on RunPod server for remedy recommendations

### Priority 3 (Nice to have)
- [ ] **RAG integration** with Pinecone for KP book excerpts
- [ ] **Multi-turn conversation** support (currently each query is independent)
- [ ] **Confidence scoring** based on significator strength

---

## Deployment Commands (RunPod)

```bash
cd /workspace/Finetuning_LLama
git pull

# Kill old servers
pkill -f "11_api_server.py" || true
pkill -f "09_chat_ui.py" || true

# Restart API server with new code
python scripts/11_api_server.py --no-rag --max-model-len 128000 &

# Restart WebUI with new code  
python scripts/09_chat_ui.py --share --no-rag --max-model-len 128000 &
```

After restart, re-run all 12 tests to verify fixes.
