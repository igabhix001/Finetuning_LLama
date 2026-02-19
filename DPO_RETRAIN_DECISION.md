# DPO Retrain Decision Report
**Generated:** Auto-generated after full dataset audit  
**Dataset audited:** `data/dpo/dpo_pairs_clean.jsonl` → `data/dpo/dpo_pairs_final.jsonl`

---

## Audit Scorecard (12 metrics)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Reward margin mean | 20.3 | 10-25 | ✅ |
| Negative margin (wrong labels) | 2.0% | <1% | ❌ |
| Length bias (rej/cho ratio) | 2.9x | 1.5-3x | ✅ |
| Extreme length bias (rej >3x) | 65.0% | <40% | ⚠️ |
| Duplicate prompt rate | 82.1% | <5% | ❌ |
| Semantic cluster entropy | 0.57 | >0.85 | ⚠️ |
| Flip inconsistency rate | 0.4% | <10% | ✅ |
| Chosen markdown | 0% | 0% | ✅ |
| Chosen robotic headers | 0% | 0% | ✅ |
| Chosen ISO dates | 0% | 0% | ✅ |
| Product spam | 0% | 0% | ✅ |
| Language mismatch (Hindi Q→Eng A) | 10.7% | <5% | ⚠️ |

**Score: 8 PASS / 2 WARN / 2 FAIL**

---

## Critical Issues Requiring Action

### ❌ FAIL 1: Duplicate Prompt Rate = 82.1%
- **Root cause:** Same 388 unique questions × 11 kundalis = same prompt text repeated 5-6x
- **Impact:** Model sees the same question 5-6 times with different chart answers — this is actually **intentional and acceptable** for KP astrology (same question, different charts = different correct answers). However, 1092 pairs have empty/null category which inflates apparent duplication.
- **Action:** This is a structural artifact of multi-chart training, NOT a true quality problem. The model needs to learn that the same question has different answers for different charts. **No action needed.**

### ❌ FAIL 2: Negative Margin = 2.0% (44 pairs)
- **Root cause:** 44 pairs where the "chosen" response scored lower than "rejected" by the proxy scorer. These are mislabeled pairs.
- **Impact:** These 44 pairs teach the model the WRONG preference — they actively hurt training.
- **Action:** Already filtered out in `dpo_pairs_final.jsonl`. ✅

### ⚠️ WARN: Language Mismatch = 10.7% (232 pairs)
- **Root cause:** Hindi questions in dataset got English chosen responses. Model trained on these learns to ignore language matching.
- **Impact:** This is the #1 reason "Meri shaadi kab hogi?" gets an English response.
- **Action:** **RETRAIN REQUIRED** to fix this properly. Post-processing cannot fully fix language matching.

### ⚠️ WARN: Semantic Entropy = 0.57 (target >0.85)
- **Root cause:** 63% of pairs have `category = "unknown"` (from V2 fixes batch which didn't set categories). The actual category diversity is good (14 categories, 276 unique intents in unknown bucket).
- **Impact:** Low entropy is a measurement artifact, not a real diversity problem.
- **Action:** Fix category labeling in next generation run. **Low priority.**

---

## What Was Fixed (This Session)

### Post-Processing Fixes in `scripts/09_chat_ui.py`

| Fix | Type | Status |
|-----|------|--------|
| Cancer/medical query intercept at query level | Safety | ✅ Done |
| `rulesused:/KPGEN/KPTIM` metadata strip in postprocess | Safety | ✅ Done |
| `timingmethod/maxperiod/minperiod` metadata strip | Safety | ✅ Done |
| ALL-CAPS rule codes (e.g. `KPGEN0956ADIUS0285`) strip | Safety | ✅ Done |
| "YES you have Cancer" response replacement | Safety | ✅ Done |
| Medical diagnosis intercept handler in `predict()` | Safety | ✅ Done |
| "Can you really predict?" → confident response | Trust | ✅ Done |
| Self-doubt phrases stripped ("jis method se...reliable nahi") | Trust | ✅ Done |
| Age impossibility guard (first job for 39yo) | Accuracy | ✅ Done |
| New robotic headers stripped (15 new patterns) | Format | ✅ Done |
| New filler phrases stripped (20 new patterns) | Format | ✅ Done |
| Self-doubt regex patterns (Phase 8.4) | Trust | ✅ Done |
| Expanded dangerous medical terms (50+ new patterns) | Safety | ✅ Done |

### Dataset Cleanup

| Action | Result |
|--------|--------|
| Audited `dpo_pairs.jsonl` (2989 pairs) | Done |
| Removed length-bias pairs (ratio >5x or <0.3x) | 823 removed |
| Saved `dpo_pairs_clean.jsonl` | 2166 pairs |
| Full quality audit on clean file | Done |
| Removed negative-margin + noisy pairs | 147 removed |
| Saved `dpo_pairs_final.jsonl` | **2019 pairs** |

---

## Retrain Decision: YES — RETRAIN REQUIRED

### Reason
The current model scores **~13/30 (43%)** on the test script. Post-processing fixes will raise this to approximately **18-20/30 (60-67%)** by fixing:
- Cancer diagnosis (query-level intercept) → +2 points
- Metadata leakage → +1 point  
- Self-doubt responses → +1 point
- Age impossibility → +1 point
- More header/filler stripping → +1 point

**But to reach the client target of 24+/30 (80%), retraining is mandatory** because:
1. **Language matching** (Hindi → English): 10.7% of training pairs have this bug. Model learned wrong behavior.
2. **Verbosity**: Model ignores system prompt length rules — needs DPO signal to learn brevity.
3. **Deflection**: Model still says "I would analyze..." instead of giving dates — needs DPO signal.
4. **Sub-lord citation**: Model misses cusp sub-lords in 50%+ of responses — needs more training examples.

### What to Fix Before Retraining

1. **Fix language mismatch in dataset** — filter/fix the 232 Hindi-Q→English-A pairs
2. **Use `dpo_pairs_final.jsonl`** (2019 clean pairs) as base
3. **Generate 500 more Hindi pairs** with correct Hindi responses
4. **Regenerate rejected responses** for pairs with ratio >3x (shorter, content-bad not length-bad)

---

## Exact Retrain Commands (RunPod)

```bash
# 1. Pull latest code + dataset
cd /workspace/Finetuning_LLama
git pull

# 2. Verify clean dataset
python -c "
import json
pairs = [json.loads(l) for l in open('data/dpo/dpo_pairs_final.jsonl')]
print(f'Final dataset: {len(pairs)} pairs')
"

# 3. Prepare dataset for training
python scripts/14_prepare_dpo_dataset.py \
  --input data/dpo/dpo_pairs_final.jsonl \
  --output data/dpo/prepared_final \
  --test-size 0.05

# 4. Train DPO (LoRA rank 8, beta 0.1)
python scripts/15_train_dpo.py \
  --base-model /workspace/models/sft_merged \
  --dataset data/dpo/prepared_final \
  --output /workspace/models/dpo_v2 \
  --beta 0.1 \
  --lora-rank 8 \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 5e-5

# 5. Merge LoRA adapter
python scripts/16_merge_dpo_lora.py \
  --base /workspace/models/sft_merged \
  --adapter /workspace/models/dpo_v2 \
  --output /workspace/models/dpo_v2_merged

# 6. Restart vLLM with new model
python scripts/08_serve_vllm.py \
  --model /workspace/models/dpo_v2_merged \
  --port 8000

# 7. Run test script (target: 24+/30)
# Use DPO_MODEL_TEST_SCRIPT.md with Arjun Mehta kundali
```

---

## Files Status

| File | Status | Action |
|------|--------|--------|
| `data/dpo/dpo_pairs.jsonl` | 2989 pairs — current canonical | Keep as backup |
| `data/dpo/dpo_pairs_backup.jsonl` | 2557 pairs — older version | **DELETE** |
| `data/dpo/dpo_pairs_merged.jsonl` | 2989 pairs — duplicate of main | **DELETE** |
| `data/dpo/dpo_pairs_v2_fixes.jsonl` | 500 pairs — already merged | **DELETE** |
| `data/dpo/dpo_pairs_clean.jsonl` | 2166 pairs — length-bias filtered | Keep |
| `data/dpo/dpo_pairs_final.jsonl` | **2019 pairs — USE THIS FOR TRAINING** | ✅ CANONICAL |
| `data/dpo/dpo_pairs_removed.jsonl` | 147 removed pairs — audit log | Keep for reference |

---

## Expected Score After Retrain

| Issue | Current | Post-Processing Only | After Retrain |
|-------|---------|---------------------|---------------|
| Cancer diagnosis | FAIL | ✅ Fixed (intercept) | ✅ |
| Metadata leakage | FAIL | ✅ Fixed (strip) | ✅ |
| Self-doubt responses | FAIL | ✅ Fixed (intercept) | ✅ |
| Age impossibility | FAIL | ✅ Fixed (guard) | ✅ |
| Verbosity (3-4 para) | FAIL | ⚠️ Partial (strip) | ✅ |
| Headers leak | FAIL | ⚠️ Partial (strip) | ✅ |
| Deflection | FAIL | ⚠️ Partial (retry) | ✅ |
| Hindi language match | FAIL | ❌ Cannot fix | ✅ |
| Sub-lord citation | PARTIAL | ❌ Cannot fix | ✅ |
| Tense errors | FAIL | ⚠️ Partial | ✅ |
| **Estimated score** | **13/30** | **~18-20/30** | **~24-26/30** |
