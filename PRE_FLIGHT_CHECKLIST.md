# Pre-Flight Checklist - DPO Training

## 🎯 **BEFORE TRAINING ON RUNPOD**

Run these verification steps to ensure 100% readiness and avoid wasting resources.

---

## ✅ **Step 1: Verify Tokenizer**

```bash
cd /workspace/Finetuning_LLama
python verify_tokenizer.py
```

**Expected output for clean Llama 3.1**:
```
✅ TOKENIZER IS PURE LLAMA 3.1

Recommendation:
  - Remove warning suppression in 15_train_dpo.py (lines 28-33)
  - The regex warning is likely a false positive
  - Proceed with training
```

**If you see**:
```
⚠️  TOKENIZER IS LLAMA 3.1 BUT HAS ISSUES
```
→ The script will auto-apply `fix_mistral_regex=True` during training.

**If you see**:
```
❌ TOKENIZER IS NOT LLAMA 3.1
```
→ **STOP. DO NOT TRAIN.** Contact support.

---

## ✅ **Step 2: Verify Dataset**

```bash
# Check filtered dataset exists
ls -lh data/dpo/prepared/train_filtered/
ls -lh data/dpo/prepared/test_filtered/

# Should show:
# train_filtered: ~895 .parquet files
# test_filtered: ~100 .parquet files
```

**If missing**:
```bash
python filter_pessimistic_pairs.py
python scripts/14_prepare_dpo_data.py
```

---

## ✅ **Step 3: Verify Config Files**

```bash
# Check DPO config
cat configs/dpo_config.yaml | grep -E "beta|learning_rate|train_data|eval_data"
```

**Expected**:
```yaml
beta: 0.1                    # Conservative to prevent reward hacking
learning_rate: 5.0e-6        # Conservative to prevent reward hacking
train_data: "./data/dpo/prepared/train_filtered"
eval_data: "./data/dpo/prepared/test_filtered"
```

**If different**: Update `configs/dpo_config.yaml` before training.

---

## ✅ **Step 4: Verify Model Exists**

```bash
ls -lh models/merged_sft/

# Should show:
# - config.json
# - model.safetensors (or model-*.safetensors)
# - tokenizer.json
# - tokenizer_config.json
# - special_tokens_map.json
```

**If missing**:
```bash
python scripts/05b_merge_sft_lora.py
```

---

## ✅ **Step 5: Clean Up Old Checkpoints**

```bash
# Remove old DPO checkpoints to avoid confusion
rm -rf checkpoints/dpo_lora/checkpoint-*
rm -rf logs/dpo/events.out.tfevents.*

# Keep only the config
ls checkpoints/dpo_lora/
# Should show: (empty or only config files)
```

---

## ✅ **Step 6: Test Training Script (Dry Run)**

```bash
# Quick 10-step test
python scripts/15_train_dpo.py --config configs/dpo_config.yaml 2>&1 | head -100
```

**Look for**:
```
1. Loading merged DAPT+SFT model from: ./models/merged_sft/

   Tokenizer Details:
     Class: PreTrainedTokenizerFast
     Vocab size: 128256 (or similar)
     Model max length: 131072
     BOS: <|begin_of_text|> (id: 128000)
     EOS: <|end_of_text|> (id: 128009)
     PAD: <|pad|> (id: 128256) [ADDED] or [OK]
   ✓ Model loaded: 8,030,261,248 parameters

2. Loading reference model (frozen copy for DPO)...
   ✓ Reference model loaded
```

**If you see**:
- `⚠️  Mistral regex warning detected - reloading with fix` → **OK, auto-fixed**
- `⚠️  WARNING: PAD == EOS` → **OK, auto-fixed**
- Any other errors → **STOP, investigate**

**Kill the test after seeing the above**:
```bash
# Press Ctrl+C
```

---

## ✅ **Step 7: Final Checklist**

Before running full training, confirm:

- [ ] Tokenizer verified (Step 1)
- [ ] Dataset exists and is filtered (Step 2)
- [ ] Config has conservative hyperparameters (Step 3)
- [ ] Merged SFT model exists (Step 4)
- [ ] Old checkpoints cleaned (Step 5)
- [ ] Dry run passed (Step 6)
- [ ] Git repo is up to date (`git pull`)
- [ ] Environment variables set (`.env` file with `HF_TOKEN`)

---

## 🚀 **Start Training**

```bash
cd /workspace/Finetuning_LLama

# Start training (will take ~4-6 hours on 1x A100)
nohup python scripts/15_train_dpo.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

---

## 📊 **Monitor Training**

### **Every 50 steps, check**:

```bash
tail -100 training.log | grep "rewards/margins\|rewards/accuracies\|loss"
```

**Healthy training indicators**:
```
Step 50:  rewards/margins: 0.5-1.0   rewards/accuracies: 0.55-0.65   loss: 0.55-0.65
Step 100: rewards/margins: 1.0-1.5   rewards/accuracies: 0.60-0.70   loss: 0.45-0.55
Step 200: rewards/margins: 1.5-2.0   rewards/accuracies: 0.65-0.75   loss: 0.35-0.45
Step 400: rewards/margins: 2.0-2.5   rewards/accuracies: 0.70-0.80   loss: 0.25-0.35
```

**Convergence (final)**:
```
rewards/margins: 2.0-3.0
rewards/accuracies: 0.75-0.85
loss: 0.20-0.30
```

### **Red flags (STOP TRAINING)**:

❌ **Reward Hacking**:
```
rewards/margins: >5.0
rewards/accuracies: >0.95
loss: <0.10
```
→ Model is collapsing rejected probabilities without improving chosen.

❌ **No Learning**:
```
rewards/margins: <0.1 (after 200 steps)
rewards/accuracies: ~0.50 (random)
loss: ~0.69 (stuck)
```
→ Model is not learning preferences.

❌ **Divergence**:
```
loss: increasing after 100 steps
rewards/margins: negative
```
→ Training is unstable.

---

## 🔍 **Post-Training Validation**

After training completes:

### **1. Check for reward hacking**:

```bash
python diagnose_dpo_quality.py
```

**Expected**:
```
Average changes (DPO - SFT):
  Chosen:   +0.20 to +0.40
  Rejected: -0.50 to -1.00
  Ratio: 2.0x to 3.0x  ✅

NaN logprobs: 0/10 samples  ✅
Reward hacking: 0/10 samples  ✅
```

**If ratio >5x or NaN logprobs >0**: **DO NOT DEPLOY**. Retrain with lower beta/LR.

### **2. Validate generation quality**:

```bash
python validate_dpo_training.py
```

**Expected**:
```
Length ratio (DPO/SFT): 0.95x to 1.10x  ✅
Repetition: None detected  ✅
Quality: Responses are concise and coherent  ✅
```

**If length ratio >1.2x or repetition detected**: **DO NOT DEPLOY**. Retrain.

### **3. Manual quality check**:

```bash
python scripts/09_chat_ui.py
```

Test with 5-10 queries. Compare SFT vs DPO responses.

---

## ✅ **Deployment Decision Matrix**

| Validation Result | Action |
|-------------------|--------|
| All checks pass ✅ | **DEPLOY** - Merge LoRA and upload to HuggingFace |
| Ratio 3-5x, no NaN ⚠️ | **REVIEW** - Manual quality check, deploy if good |
| Ratio >5x or NaN ❌ | **RETRAIN** - Lower beta to 0.05, LR to 2.5e-6 |
| No learning ❌ | **DEBUG** - Check dataset, increase beta to 0.15 |
| Divergence ❌ | **DEBUG** - Lower LR to 2.5e-6, check dataset |

---

## 🎯 **If All Checks Pass → Deploy**

```bash
# Merge LoRA
python scripts/16_merge_dpo_lora.py

# Upload to HuggingFace
python upload_to_huggingface.py
```

---

## 📝 **Summary**

**This checklist ensures**:
1. Tokenizer is correct and verified
2. Dataset is filtered and ready
3. Config has conservative hyperparameters
4. Training environment is clean
5. Monitoring catches issues early
6. Post-training validation prevents bad deployments

**DO NOT SKIP ANY STEPS**. Each one prevents a specific failure mode that has occurred in previous runs.

---

## 🆘 **If Issues Arise**

1. **Stop training immediately** (`kill <pid>` or Ctrl+C)
2. **Save logs** (`cp training.log training_failed_$(date +%Y%m%d_%H%M%S).log`)
3. **Run diagnostics** (`python diagnose_dpo_quality.py`)
4. **Review this checklist** and identify which step failed
5. **Fix the issue** before retraining
6. **Do NOT waste resources** on repeated failed runs

---

**Last Updated**: After tokenizer verification fix (commit 32f0d31)
