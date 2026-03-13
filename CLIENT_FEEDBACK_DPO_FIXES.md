# Client Feedback Analysis & DPO Fixes
## Addressing All 6 Critical Gaps via DPO Training

**Date**: 13 March 2026  
**Status**: ✅ DPO Prompts Enhanced — Ready for Dataset Generation  
**SFT Model Score**: 93%  
**Target After DPO**: 97-98%

---

## Client Feedback Summary (6 Critical Gaps)

### Gap #1: Hallucination Without Grounding
> "It hallucinates confidently without grounding. When you said 'child was born after 2020,' it kept insisting you weren't a parent yet — then flipped to confidently saying 'Jun 2023' with no basis."

**Root Cause**: Model pattern-matches to sound plausible instead of reading actual chart data.

**DPO Fix**: 
- CHOSEN: Only use dasha dates that ACTUALLY exist in chart YAML
- REJECTED: Confidently give dates that don't exist in chart

---

### Gap #2: Self-Contradiction
> "Marriage was first placed in Rahu-Jupiter (2013–2015), then later in Rahu-Venus (2022–2025). Same question, same chart, two completely different answers."

**Root Cause**: No consistency mechanism, no acknowledgment of revision.

**DPO Fix**:
- CHOSEN: If revising, explicitly acknowledge: "Pehle maine galat bataya tha. Chart dobara dekha..."
- REJECTED: Give different answers without acknowledging contradiction

---

### Gap #3: Ignores User Corrections (CRITICAL)
> "You corrected it three times about the child. A real KP astrologer would say: 'Theek hai, aapne confirm kiya hai ki child 2020 ke baad born hua. Let me look at which dasha period falls post-2020.' Instead it kept gaslighting you."

**Root Cause**: Model doesn't respect conversation context, keeps asserting old answer.

**DPO Fix**:
- CHOSEN: "Got it — child born after 2020. Checking 5th CSL activation post-2020: Saturn antardasha (2021-2024) shows 5th house strongly activated — was it around 2022-2023? Confirm karo."
- REJECTED: "Based on your chart, there hasn't been a significant childbirth event since 2020..."

**New Question Types Added** (weight 2.0-2.5):
```python
("user_correction", "No, my child was born after 2020. When?", 2.5)
("user_correction", "I'm already a parent. My child was born after 2020. Tell me when.", 2.5)
("user_correction", "You're wrong. Child was born in 2022, not 2014.", 2.0)
("user_correction", "No, I got married in 2018, not 2014.", 2.0)
```

---

### Gap #4: Significations as Filler (Not Reasoning)
> "'Venus signifies 3,5,10,12' is repeated like a mantra but never actually used to derive anything. The logic chain is missing."

**Root Cause**: Model states houses but doesn't explain what they indicate.

**DPO Fix**:
- CHOSEN: "Venus signifies 3,5,10,12 — houses 3+12 point to communication/media, house 5 to creativity, house 10 to profession. This combination indicates creative media or spiritual business."
- REJECTED: "Venus signifies houses 3,5,10,12 while being connected to leadership positions through its natural signification." (filler)

---

### Gap #5: Inconsistent Event Dating
> "Marriage and first child placed in overlapping periods (both in Rahu-Jupiter, 2013–2015, months apart) with no acknowledgment that this is unusual."

**Root Cause**: No plausibility check for unusual overlaps.

**DPO Fix**:
- CHOSEN: "Marriage and first child both showing in Rahu-Jupiter (2013-2015) — this is unusual timing. Either chart data needs verification or there's a special circumstance. Can you confirm?"
- REJECTED: "Marriage Jun 2014, first child Aug 2014" (no acknowledgment)

---

### Gap #6: Generic Career Answers
> "'Government service or authority-based roles' from a 10th cusp sub-lord Venus signifying 3,5,10,12 — that's not a logical derivation."

**Root Cause**: Generic answers not derived from specific house meanings.

**DPO Fix**:
- CHOSEN: "10th CSL Venus signifies houses 3,5,10,12. Houses 3+12 indicate media, communication, creative/spiritual business — more likely than government. Sahi hai?"
- REJECTED: "Your career field strongly indicates government service or authority-based roles since your 10th cusp sub-lord Venus signifies houses 3,5,10,12."

---

## The 9/10 KP Astrologer Pattern (Client's Requirement)

Every prediction must follow this EXACT reasoning chain:

1. **IDENTIFY** the relevant Cusp Sub-Lord (CSL) for the house in question
2. **STATE** which houses that CSL signifies (from chart data)
3. **MATCH** to the dasha/antardasha period that activates those houses
4. **GIVE** a specific month-year prediction
5. **END** with a confirmation ask: "Sahi hai?" or "Confirm karo"

### Example Transformations

**Before (Bad - Current SFT)**:
> "Your marriage likely occurred during Rahu-Jupiter period from May 2013 to Oct 2015, specifically around Jun 2014 when you were 26 years old — Jupiter as both 7th cusp sub-lord and natural significator of marriage created the perfect timing for matrimony."

**After (Good - DPO Target)**:
> "7th CSL Jupiter signifies houses 3,7,10. Rahu-Jupiter period (May 2013 to Oct 2015) activated these — most likely Jun 2014 when you were 26. Sahi hai?"

---

## DPO Prompt Enhancements Made

### CHOSEN_SYSTEM_PROMPT Additions:

1. **9/10 KP Astrologer Pattern** — 5-step reasoning chain with examples
2. **User Correction Handling** — Explicit rules to acknowledge and re-analyze
3. **No Hallucination Rule** — Only use dates from actual chart YAML
4. **Logical Reasoning Chain** — Houses must be USED for derivation, not filler
5. **Flag Unusual Overlaps** — Acknowledge when events are unusually close
6. **Confirmation Ask** — End with "Sahi hai?" or "Confirm karo"

### REJECTED_SYSTEM_PROMPT Additions:

1. **Client Complaint #1** — Hallucination without grounding
2. **Client Complaint #2** — Self-contradiction without acknowledgment
3. **Client Complaint #3** — Ignoring user corrections (gaslighting)
4. **Client Complaint #4** — Significations as filler
5. **Client Complaint #5** — Inconsistent event dating
6. **Client Complaint #6** — Generic career answers
7. **No Confirmation Ask** — Assert confidently without verification

### QUESTION_POOL Additions:

```python
# USER CORRECTION SCENARIOS — CRITICAL FOR CLIENT COMPLAINT #3
("user_correction", "No, my child was born after 2020. When?", 2.5)
("user_correction", "I'm already a parent. My child was born after 2020. Tell me when.", 2.5)
("user_correction", "You're wrong. Child was born in 2022, not 2014.", 2.0)
("user_correction", "No, I got married in 2018, not 2014.", 2.0)
("user_correction", "You said Rahu-Jupiter but I got married in Rahu-Venus period.", 1.5)
("user_correction", "You gave me a different answer earlier. Which one is correct?", 1.5)

# CONFIRMATION-SEEKING QUERIES
("confirmation_needed", "When did I get married? Be specific.", 1.5)
("confirmation_needed", "When was my first child born? Give me the exact period.", 1.5)
```

---

## Can DPO Fix These Issues?

| Gap | DPO Fixable? | Mechanism |
|-----|--------------|-----------|
| #1 Hallucination | ✅ YES | Train preference for chart-grounded responses |
| #2 Self-contradiction | ✅ YES | Train to acknowledge/revise explicitly |
| #3 Ignore corrections | ✅ YES | Train to accept corrections with acknowledgment |
| #4 Filler significations | ✅ YES | Train complete CSL→houses→meaning chain |
| #5 Inconsistent dating | ✅ YES | Train to flag unusual overlaps |
| #6 Generic careers | ✅ YES | Train specific derivation from houses |

**Answer: YES — All 6 gaps can be addressed via DPO training.**

DPO works by teaching the model to prefer "chosen" responses over "rejected" responses. By including:
- Chosen: Chart-grounded, acknowledges corrections, logical reasoning, confirmation asks
- Rejected: Hallucinated, ignores corrections, filler significations, generic answers

The model will learn to shift its behavior toward the 9/10 astrologer pattern.

---

## Expected Impact

### Before DPO (Current SFT - 93%):
- ❌ Hallucination: "Jun 2023" with no chart basis
- ❌ Contradiction: Different answers for same question
- ❌ Gaslighting: "there hasn't been a childbirth event since 2020"
- ❌ Filler: "Venus signifies 3,5,10,12" (not used for reasoning)
- ❌ No overlap flag: "Marriage Jun 2014, child Aug 2014"
- ❌ Generic: "government service or authority-based roles"

### After DPO (Target - 97-98%):
- ✅ Grounded: Only dates from actual chart YAML
- ✅ Consistent: "Pehle maine galat bataya tha. Chart dobara dekha..."
- ✅ Respectful: "Got it — child born after 2020. Checking 5th CSL..."
- ✅ Reasoning: "Houses 3+12 indicate communication/media..."
- ✅ Flagging: "Marriage and child in same period — unusual. Can you confirm?"
- ✅ Specific: "creative media or spiritual business rather than government"

---

## Next Steps

### 1. Generate DPO Dataset (3000+ pairs)
```bash
cd /workspace/Finetuning_LLama
python scripts/20_generate_dpo_consultation.py --count 3000 --batch
```

### 2. Monitor Batch Jobs
```bash
python scripts/20_generate_dpo_consultation.py --batch-check <batch_id>
```

### 3. Download & Audit
```bash
python scripts/20_generate_dpo_consultation.py --batch-download <batch_id>
python scripts/dpo_quality_audit.py --filter
```

### 4. Train DPO
```bash
python scripts/05b_merge_sft_lora.py  # Merge SFT LoRA first
python scripts/15_train_dpo.py        # Train DPO
python scripts/16_merge_dpo_lora.py   # Merge DPO LoRA
```

### 5. Re-test with Client
Deploy updated model and have client re-test the same scenarios.

---

## Files Modified

1. ✅ `scripts/20_generate_dpo_consultation.py`
   - Enhanced CHOSEN_SYSTEM with 9/10 astrologer pattern
   - Enhanced REJECTED_SYSTEM with all 6 client complaints
   - Added 20+ user correction question types
   - Added confirmation-seeking question types

2. ✅ `CLIENT_FEEDBACK_DPO_FIXES.md` (this file)
   - Complete documentation of all fixes

---

## Research Basis

- **NeurIPS 2025 BeeS**: Margin-based filtering, quality > quantity
- **NeurIPS 2024 Ivison**: Data quality > algorithm choice
- **arXiv 2508.18312**: Chosen quality dominates DPO performance
- **MixDPO**: Length bias is #1 silent killer — we keep chosen/rejected same length

---

**Status**: Ready for DPO dataset generation. All 6 client complaints addressed in prompts.
