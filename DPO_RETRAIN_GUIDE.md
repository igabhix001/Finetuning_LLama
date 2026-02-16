# DPO Retrain Guide - Fixing Reward Hacking

## 🔴 **What Went Wrong (Evidence-Based)**

Your diagnostics revealed **confirmed reward hacking**:

### **Critical Evidence from Your Own Run**

1. **32x Imbalanced Ratio**
   ```
   Average changes (DPO - SFT):
     Chosen:   +0.10
     Rejected: -3.20
     Ratio: 32.00x
   ```
   - **Healthy DPO**: Ratio <3x
   - **Your training**: 32x = annihilating rejected, not improving chosen

2. **80% NaN Logprobs**
   ```
   8/10 samples: logp = 0.00 (avg: nan)
   ```
   - Logprob computation broke
   - Entire training accuracy computed in wrong token space

3. **Tokenizer Still Broken**
   ```
   This will lead to incorrect tokenization.
   Use fix_mistral_regex=True
   ```
   - Warning appeared in diagnostics
   - Fix was never applied

4. **PAD == EOS Token**
   ```
   pad_token_id: 128009 == eos_token_id: 128009
   ```
   - Known Mistral failure mode
   - Causes rejected sequences to look like they end immediately
   - Artificial margin inflation

### **What Your Training Actually Did**

**Should have optimized**:
```
↑ π(chosen)   (improve chosen responses)
↓ π(rejected) (reduce rejected responses)
Both change moderately
```

**Actually optimized**:
```
→ π(chosen)   (barely moved: +0.1)
↓↓↓ π(rejected) (annihilated: -3.2)
```

**Result**: Perfect training curves, broken model behavior.

---

## ✅ **Fixes Applied**

### **1. Fixed Tokenizer Regex**
```python
# scripts/15_train_dpo.py
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    fix_mistral_regex=True  # CRITICAL FIX
)
```

### **2. Separated PAD from EOS**
```python
# scripts/15_train_dpo.py
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
```

**Why this matters**:
- PAD == EOS makes padded rejected tails look like valid EOS-terminated completions
- DPO learns "rejected sequences should end immediately everywhere"
- Margins shoot up artificially without semantic learning

### **3. Reduced Beta (0.2 → 0.1)**
```yaml
# configs/dpo_config.yaml
beta: 0.1  # Conservative to prevent reward hacking
```

### **4. Reduced Learning Rate (1e-5 → 5e-6)**
```yaml
# configs/dpo_config.yaml
learning_rate: 5.0e-6  # Conservative to prevent reward hacking
```

### **5. Adjusted Health Guard (6.0 → 3.0)**
```python
# scripts/15_train_dpo.py
DPOHealthCallback(max_margin=3.0, min_loss=0.05)
```

---

## 🚀 **Retrain Procedure (RunPod)**

### **Step 1: Pull Fixes**
```bash
cd /workspace/Finetuning_LLama
git pull
```

### **Step 2: Clean Old Checkpoints**
```bash
# Remove broken DPO checkpoint
rm -rf checkpoints/dpo_lora/*

# Verify clean state
ls -la checkpoints/dpo_lora/
```

### **Step 3: Restart Training**
```bash
python scripts/15_train_dpo.py
```

**Expected output**:
```
1. Loading merged DAPT+SFT model from: ./models/merged_sft/
   ✓ Added separate PAD token: <|pad|> (id: 128010)
   ✓ Model loaded: 8,030,261,248 parameters

2. Loading reference model (frozen copy for DPO)...
   ✓ Reference model loaded and frozen

5. Setting up DPO training configuration...
   ✓ DPO Config:
     Beta: 0.1
     Learning rate: 5e-06
     Health guard: stop if margins > 3.0 or loss < 0.05
```

**Key differences from broken run**:
- ✅ No tokenizer regex warning
- ✅ Separate PAD token (not EOS)
- ✅ Conservative beta (0.1 vs 0.2)
- ✅ Conservative LR (5e-6 vs 1e-5)

---

## 📊 **What to Monitor**

### **Healthy Training Indicators**

**Step 50-100**:
```
rewards/chosen:   -0.05 to +0.05  (small improvement)
rewards/rejected: -0.5 to -1.0    (moderate degradation)
rewards/margins:  0.5 to 1.0      (growing steadily)
rewards/accuracies: 0.6 to 0.75   (NOT 1.0 immediately!)
loss: 0.5 to 0.4                  (gradual decrease)
```

**Step 200-300**:
```
rewards/chosen:   +0.1 to +0.2
rewards/rejected: -1.5 to -2.0
rewards/margins:  1.5 to 2.0
rewards/accuracies: 0.75 to 0.85
loss: 0.3 to 0.25
```

**Final (Step 378)**:
```
rewards/chosen:   +0.2 to +0.3
rewards/rejected: -2.0 to -2.5
rewards/margins:  2.0 to 2.5      (NOT 3.0+)
rewards/accuracies: 0.80 to 0.90  (NOT 1.0)
loss: 0.20 to 0.25
```

### **Red Flags (Reward Hacking)**

❌ Accuracy → 1.0 in <100 steps  
❌ Margins > 3.0 before epoch 2  
❌ Rejected change > 3x chosen change  
❌ Loss < 0.15  

If you see these, **STOP TRAINING** and report back.

---

## 🔬 **Post-Training Validation**

### **Step 1: Run Diagnostics**
```bash
python diagnose_dpo_quality.py
```

**Expected healthy output**:
```
Average changes (DPO - SFT):
  Chosen:   +0.20 to +0.40
  Rejected: -0.50 to -1.00
  Ratio: 2.0x to 3.0x  ✅ (NOT 32x!)

Reward hacking detected: 0/10 samples  ✅
```

### **Step 2: Validate Generation**
```bash
python validate_dpo_training.py
```

**Expected healthy output**:
```
Average length ratio (DPO/SFT): 0.95x to 1.10x  ✅

✅ GOOD: Response lengths are reasonable.
DPO training appears successful.
```

### **Step 3: Manual Quality Check**

Test on real queries:
```bash
python scripts/test_model.py \
  --model ./checkpoints/dpo_lora/final \
  --kundali sample_kundali/kundali_Amit_Kumar.json \
  --question "When will I get married?"
```

Compare with SFT baseline:
```bash
python scripts/test_model.py \
  --model ./models/merged_sft \
  --kundali sample_kundali/kundali_Amit_Kumar.json \
  --question "When will I get married?"
```

**Check for**:
- ✅ DPO response is more concise
- ✅ DPO response is more direct
- ✅ No hallucinations
- ✅ No verbosity increase
- ✅ Maintains factuality

---

## ✅ **Decision Matrix**

### **Scenario A: Diagnostics Pass**

**Indicators**:
- Ratio <3x
- 0 reward hacking samples
- Length ratio 0.9-1.1x
- Manual quality good

**Action**:
```bash
# Merge DPO LoRA
python scripts/16_merge_dpo_lora.py

# Upload to HuggingFace
python upload_to_huggingface.py \
  --model ./models/llama-3.1-8b-dpo-final \
  --repo YOUR_USERNAME/kp-astrology-dpo-v4
```

### **Scenario B: Diagnostics Fail**

**Indicators**:
- Ratio >5x
- Reward hacking detected
- Length ratio >1.3x or <0.7x

**Action**:
- Report diagnostic results
- Further reduce beta (0.1 → 0.05)
- Further reduce LR (5e-6 → 2e-6)
- Retrain again

---

## 📋 **Summary**

**Previous run**: Reward hacking confirmed (32x ratio, 80% NaN logprobs)

**Root causes**:
1. Tokenizer regex bug
2. PAD == EOS token
3. Too aggressive beta (0.2)
4. Too aggressive LR (1e-5)

**Fixes applied**:
1. ✅ `fix_mistral_regex=True`
2. ✅ Separate PAD token
3. ✅ Beta: 0.2 → 0.1
4. ✅ LR: 1e-5 → 5e-6
5. ✅ Health guard: 6.0 → 3.0

**Expected training time**: ~2.5 hours

**Next steps**:
1. Pull fixes on RunPod
2. Clean checkpoints
3. Retrain with conservative config
4. Run diagnostics
5. Deploy if validation passes

---

## 🎯 **Key Takeaway**

**Perfect training curves ≠ successful alignment**

Your previous run had:
- ✅ Loss → 0.20
- ✅ Accuracy → 1.0
- ✅ Margins → 3.7

But diagnostics showed:
- ❌ 32x imbalanced ratio
- ❌ 80% NaN logprobs
- ❌ Reward hacking detected

**This time**: Trust diagnostics, not curves.
