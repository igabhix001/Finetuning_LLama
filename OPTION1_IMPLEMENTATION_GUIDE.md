# Option 1 Implementation - DPO Dataset Fix & Retrain
**Date**: Feb 16, 2026  
**Status**: IN PROGRESS  
**Timeline**: 3-4 days  
**Cost**: ~$100 (OpenAI API)

---

## 📊 Audit Results Summary

### Existing DPO Dataset Quality: **GOOD (3.5% issue rate)**

**Total pairs**: 2,557  
**Issues found**:
- ✅ Name mismatches: 0 (excellent!)
- ✅ Metadata leakage: 0 (excellent!)
- ⚠️ Safety issues: 21 (FALSE POSITIVES - "Cancer" zodiac sign, not disease)
- ⚠️ Past/future confusion: 68 pairs (2.7% - needs fixing)

**Verdict**: Dataset is actually quite clean. The model's inference problems are NOT primarily due to bad training data.

---

## 🔍 Root Cause Analysis

### Why is the deployed model failing despite good training data?

1. **Name Hallucination (40% failure rate in testing)**
   - **NOT in training data** (0 name mismatches found)
   - **Cause**: Model not learning to extract name from input JSON
   - **Fix**: Add explicit name extraction examples with stricter prompts

2. **Scripted Responses ("Feb 2026 - Jul 2026" repeated)**
   - **Cause**: Model memorized common patterns instead of reading dasha dates
   - **Fix**: Add diverse examples with different date ranges, enforce date reading

3. **Past/Future Confusion**
   - **68 pairs (2.7%)** in training data have this issue
   - **Fix**: Remove these 68 pairs, add 100+ correct past event examples

4. **Medical Malpractice**
   - **NOT in training data** (safety examples were correct)
   - **Cause**: Insufficient safety examples
   - **Fix**: Add 30+ safety intercept examples

---

## 🎯 Solution: Generate 500 Targeted High-Quality Pairs

### Focus Areas (Question Distribution):

1. **Name Extraction** (50 pairs)
   - "What is my name?" with various charts
   - Must extract exact name from YAML

2. **Past Events** (100 pairs)
   - "What happened in 2020/2021/2022/2023/2024?"
   - "When did I complete my education?"
   - "When did I get my first job?"
   - Must use PAST tense and correct years

3. **Simple Factual** (50 pairs)
   - "What is my lagna/rashi/nakshatra?"
   - Must be EXACTLY 1 sentence

4. **Safety Intercepts** (30 pairs)
   - "When will I die?" / "Do I have cancer?"
   - Must redirect to medical professional, NOT predict

5. **Marriage/Career/Financial Timing** (150 pairs)
   - Must read ACTUAL dasha dates from YAML
   - Must show diverse date ranges (not scripted)

6. **Emotional Queries** (30 pairs)
   - Must show empathy + use correct name

7. **General** (90 pairs)
   - Mix of all categories

---

## 🔧 Enhanced System Prompts

### Key Improvements in V2 Prompts:

#### **CHOSEN Prompt Enhancements:**

```
*** CRITICAL: NAME EXTRACTION (ZERO TOLERANCE) ***
STEP 1: Extract the person's name from chart YAML ("name:" field)
STEP 2: Use ONLY that exact name: "[Name] ji"
STEP 3: NEVER use any other name
VIOLATION = INVALID RESPONSE

*** CRITICAL: PAST/FUTURE TENSE (ZERO TOLERANCE) ***
STEP 1: Read "today_date:" from YAML
STEP 2: Compare every date to today_date
  - BEFORE → PAST tense: "that period has passed (yeh period beet chuka hai)"
  - AFTER → FUTURE tense: "starting from [month year]"
VIOLATION = INVALID RESPONSE

*** CRITICAL: DATE READING (ZERO TOLERANCE) ***
STEP 1: Read dasha dates from YAML
STEP 2: Use ACTUAL dates from YAML, NOT made-up dates
VIOLATION = INVALID RESPONSE
```

#### **REJECTED Prompt Enhancements:**

```
Wrong patterns to include:
- Use DIFFERENT name than chart
- For past questions → give future dates
- Use ISO format: "2025-10"
- No justification
- For death queries → predict death dates
```

---

## 📋 Implementation Steps

### ✅ **Step 1: Audit Existing Dataset** (COMPLETED)
- Created `audit_dpo_dataset.py`
- Ran audit on 2,557 pairs
- Found 3.5% issue rate (GOOD quality)
- Identified 68 past/future confused pairs

### ✅ **Step 2: Create V2 Generation Script** (COMPLETED)
- Created `generate_dpo_v2_sync.py`
- Implements stricter prompts
- Uses OpenAI synchronous API for immediate results
- Parallel generation with 10 workers

### ✅ **Step 3: Create Merge Script** (COMPLETED)
- Created `merge_dpo_datasets.py`
- Filters out 68 confused pairs from old dataset
- Merges with new 500 pairs
- Final dataset: ~3,000 pairs (2,489 old + 500 new)

### 🔄 **Step 4: Generate 500 New Pairs** (IN PROGRESS)
```bash
python generate_dpo_v2_sync.py --count 500 --workers 10
```
- Status: Running (10% complete)
- ETA: 5-10 minutes
- Output: `data/dpo/dpo_pairs_v2_fixes.jsonl`

### ⏳ **Step 5: Merge Datasets** (PENDING)
```bash
python merge_dpo_datasets.py
```
- Removes 68 confused pairs
- Adds 500 new pairs
- Updates `data/dpo/dpo_pairs.jsonl`

### ⏳ **Step 6: Prepare for Training** (PENDING)
```bash
python scripts/14_prepare_dpo_dataset.py
```
- Splits into train/test
- Validates format
- Creates prepared dataset

### ⏳ **Step 7: Upload to RunPod** (PENDING)
```bash
# On local machine
git add .
git commit -m "DPO V2 dataset with targeted fixes"
git push

# On RunPod
cd /workspace/Finetuning_LLama
git pull
```

### ⏳ **Step 8: Train DPO** (PENDING)
```bash
# On RunPod
python scripts/15_train_dpo.py
```
- Uses same hyperparameters as before
- Monitors for reward hacking
- Early stopping at best checkpoint
- ETA: 4-6 hours

### ⏳ **Step 9: Merge DPO LoRA** (PENDING)
```bash
# On RunPod
python scripts/16_merge_dpo_lora.py \
  --base-model ./models/merged_sft \
  --dpo-lora ./checkpoints/dpo_lora/final \
  --output ./models/final_dpo_v2
```

### ⏳ **Step 10: Restart vLLM Server** (PENDING)
```bash
# On RunPod - Stop current server
pkill -f vllm

# Start with new model
python scripts/08_serve_vllm.py --model ./models/final_dpo_v2
```

### ⏳ **Step 11: Retest Model** (PENDING)
- Use `DPO_MODEL_TEST_SCRIPT.md`
- Run all 30 questions
- Target: **24+/30 points (80%+)**
- Focus on:
  - ✅ Name consistency (must be 100%)
  - ✅ Past/future correctness (must be 100%)
  - ✅ Medical safety (must be 100%)
  - ✅ Date diversity (no scripted responses)

---

## 📊 Expected Improvements

### Current Model (V1) Issues:
| Issue | Current | Target | Fix |
|-------|---------|--------|-----|
| Name hallucination | 40% | 0% | Explicit extraction examples |
| Past/future confusion | 17% | 0% | Remove 68 pairs, add 100+ correct |
| Scripted responses | High | 0% | Diverse date examples |
| Medical malpractice | Yes | No | 30+ safety examples |
| Metadata leakage | Yes | No | Already fixed in existing data |

### Expected Test Score:
- **Current**: 12/30 (40%)
- **Target**: 24+/30 (80%+)
- **Improvement**: +12 points (+40%)

---

## 🎯 Success Criteria

### Before Deployment, Model MUST Achieve:

1. **✅ Name Consistency**: 0% hallucination (30/30 correct names)
2. **✅ Past/Future Correctness**: 100% (no future dates for past questions)
3. **✅ Medical Safety**: 100% (redirect, never diagnose)
4. **✅ Date Diversity**: <10% repetition (dates must vary by chart)
5. **✅ Overall Score**: 24+/30 points (80%+)

### Critical Failures (Any ONE blocks deployment):
- ❌ Name hallucination >5%
- ❌ Medical diagnosis (any instance)
- ❌ Past/future confusion >5%
- ❌ Metadata leakage (any instance)

---

## 💰 Cost Breakdown

### OpenAI API Costs:
- **500 pairs** × 2 requests (chosen + rejected) = 1,000 requests
- **Avg tokens per request**: ~1,500 (prompt) + 250 (completion) = 1,750 tokens
- **Total tokens**: 1,000 × 1,750 = 1,750,000 tokens
- **Cost**: ~$8.75 (gpt-4o: $5/1M input, $15/1M output)

### RunPod Costs:
- **Training time**: 4-6 hours
- **GPU**: A100 80GB (~$2/hour)
- **Cost**: ~$12

### **Total**: ~$21 (much less than estimated $100)

---

## 📝 Files Created

1. **`audit_dpo_dataset.py`** - Audit existing dataset quality
2. **`generate_dpo_v2_sync.py`** - Generate 500 new targeted pairs
3. **`merge_dpo_datasets.py`** - Merge old + new datasets
4. **`OPTION1_IMPLEMENTATION_GUIDE.md`** - This document
5. **`DPO_FAILURE_ANALYSIS.md`** - Detailed failure analysis

---

## 🚀 Quick Start Commands

### On Local Machine (Windows):

```powershell
# Step 1: Generate new pairs (IN PROGRESS)
python generate_dpo_v2_sync.py --count 500 --workers 10

# Step 2: Merge datasets
python merge_dpo_datasets.py

# Step 3: Prepare for training
python scripts/14_prepare_dpo_dataset.py

# Step 4: Push to Git
git add .
git commit -m "DPO V2 dataset with targeted fixes"
git push
```

### On RunPod (Linux):

```bash
# Step 1: Pull latest code
cd /workspace/Finetuning_LLama
git pull

# Step 2: Train DPO
python scripts/15_train_dpo.py

# Step 3: Merge LoRA
python scripts/16_merge_dpo_lora.py \
  --base-model ./models/merged_sft \
  --dpo-lora ./checkpoints/dpo_lora/final \
  --output ./models/final_dpo_v2

# Step 4: Restart vLLM
pkill -f vllm
python scripts/08_serve_vllm.py --model ./models/final_dpo_v2

# Step 5: Test on Gradio
# Open: https://[runpod-id].gradio.live/
```

---

## 📞 Support & Troubleshooting

### If Generation Fails:
- Check OpenAI API key: `echo $OPENAI_API_KEY`
- Check API quota: https://platform.openai.com/usage
- Reduce workers: `--workers 5`
- Reduce count: `--count 250`

### If Training Fails:
- Check GPU memory: `nvidia-smi`
- Check logs: `tail -f logs/dpo_training.log`
- Reduce batch size in `scripts/15_train_dpo.py`

### If Model Still Fails After Retraining:
- Run audit again: `python audit_dpo_dataset.py`
- Check test results: Compare to `DPO_MODEL_TEST_SCRIPT.md`
- Consider Option 2 (aggressive postprocessing) as backup

---

## 📈 Progress Tracking

- [x] Audit existing dataset
- [x] Create V2 generation script
- [x] Create merge script
- [ ] Generate 500 new pairs (IN PROGRESS - 10%)
- [ ] Merge datasets
- [ ] Prepare for training
- [ ] Upload to RunPod
- [ ] Train DPO
- [ ] Merge LoRA
- [ ] Restart vLLM
- [ ] Retest model
- [ ] Deploy if score ≥24/30

---

**Last Updated**: Feb 16, 2026 9:20 PM  
**Status**: Generation in progress (ETA: 5-10 minutes)  
**Next Action**: Wait for generation to complete, then merge datasets
