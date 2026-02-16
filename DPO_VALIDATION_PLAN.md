# DPO Training Validation Plan

## 🚨 Critical Concerns Raised by GPT

GPT identified potential **reward hacking** in your DPO training. While metrics look good, this could be degenerate policy formation.

### **Red Flags**

1. **Tokenizer Regex Mismatch**
   - Warning: "incorrect regex pattern... will lead to incorrect tokenization"
   - Impact: DPO optimizing wrong token sequences
   - **This is fundamental - if tokenization is wrong, everything is wrong**

2. **PAD == EOS Token**
   - `pad_token_id: 128009 == eos_token_id: 128009`
   - Known TRL bug: padded sequences terminate early
   - Artificially inflates margins without semantic learning

3. **Suspicious Logprob Gap**
   - Your training: `Δlogp ≈ 840` (chosen: -290, rejected: -1130)
   - Normal DPO: `Δlogp ≈ 1-10`
   - **100x larger than expected!**
   - Suggests model is annihilating rejected likelihood, not learning preferences

4. **Too-Rapid Convergence**
   - Accuracy: 0.45 → 1.00 in 40 steps
   - Loss: 0.693 → 0.231 in 50 steps
   - Classic symptom of reward hacking per alignment literature

---

## ⚠️ What Reward Hacking Looks Like

**Healthy DPO**:
```
π(chosen) ↑ (increase chosen probability)
π(rejected) ↓ (decrease rejected probability)
Both change moderately
```

**Reward Hacking**:
```
π(chosen) → (barely changes)
π(rejected) → 0 (collapses to zero)
Margins inflate artificially
```

**Result**: Model becomes verbose, loses factuality, hallucinates more, while training curves look "perfect".

---

## ✅ Validation Steps (After Training Completes)

### **Step 1: Let Training Finish**
- Don't interrupt current training
- Wait for completion (~378 steps)
- Save checkpoint

### **Step 2: Diagnose Reward Hacking**
```bash
cd /workspace/Finetuning_LLama

# Run diagnostic script
python diagnose_dpo_quality.py
```

**What it checks**:
- Compares SFT vs DPO logprobs on test set
- Detects if rejected collapsed more than chosen improved
- Flags reward hacking if `|Δrejected| > 3 × |Δchosen|`

**Expected output**:
```
Sample 1:
  SFT:  chosen: -280, rejected: -1150, margin: 870
  DPO:  chosen: -290, rejected: -1130, margin: 840
  Changes: chosen: -10, rejected: +20
  ⚠️  WARNING: Rejected collapsed more than chosen improved!
```

### **Step 3: Validate Generation Quality**
```bash
python validate_dpo_training.py
```

**What it checks**:
- Generates responses from SFT vs DPO models
- Measures verbosity increase (common failure mode)
- Checks for repetition and coherence

**Red flags**:
- DPO responses >1.5x longer than SFT
- High repetition
- Loss of factuality

### **Step 4: Fix Tokenizer (If Needed)**

If diagnostics show issues, the tokenizer regex must be fixed:

```python
# In scripts/15_train_dpo.py, line ~105
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    fix_mistral_regex=True  # ADD THIS
)
```

Then retrain from scratch.

---

## 🔬 Decision Matrix

### **Scenario A: Diagnostics Show Reward Hacking**

**Symptoms**:
- `|Δrejected| > 3 × |Δchosen|` on most samples
- DPO responses >1.5x longer than SFT
- Loss of coherence

**Action**:
```bash
# 1. Fix tokenizer
# Edit scripts/15_train_dpo.py to add fix_mistral_regex=True

# 2. Reduce beta (less aggressive)
# Edit configs/dpo_config.yaml:
beta: 0.1  # Was 0.2

# 3. Add length penalty
# Edit configs/dpo_config.yaml:
length_penalty: 1.2

# 4. Retrain from scratch
rm -rf checkpoints/dpo_lora/*
python scripts/15_train_dpo.py
```

### **Scenario B: Diagnostics Show Healthy Learning**

**Symptoms**:
- Chosen improved more than rejected degraded
- DPO responses similar length to SFT
- Coherence maintained

**Action**:
```bash
# Proceed with deployment
python scripts/16_merge_dpo_lora.py
python upload_to_huggingface.py \
  --model ./models/llama-3.1-8b-dpo-final \
  --repo YOUR_USERNAME/kp-astrology-dpo-v4
```

### **Scenario C: Mixed Results**

**Symptoms**:
- Some samples show reward hacking, some don't
- Moderate verbosity increase (1.2-1.4x)

**Action**:
- Test model extensively on real queries
- Compare with SFT model side-by-side
- Decide based on production quality needs

---

## 📊 Key Metrics to Watch

### **Healthy DPO Training**

| Metric | Healthy Range | Your Training | Status |
|--------|---------------|---------------|--------|
| Δlogp | 1-10 | ~840 | ❌ Too high |
| Accuracy | 0.6-0.85 | 1.00 | ⚠️ Too perfect |
| Length ratio | 0.8-1.2x | TBD | Need to check |
| Chosen change | Positive | TBD | Need to check |
| Rejected change | Negative | TBD | Need to check |
| Change ratio | <3x | TBD | Need to check |

### **Research References**

From alignment literature:
- "Typical preference datasets cause DPO to overconfidently assign rewards"
- "Reward hacking remains pivotal... models excessively reduce probability of rejected completions"
- "After a few steps, loss got to 0 and rewards got to 1.0" (TRL issue report)

---

## 🎯 Immediate Actions

1. **Let training complete** (don't interrupt)
2. **Run diagnostics immediately after**:
   ```bash
   python diagnose_dpo_quality.py
   python validate_dpo_training.py
   ```
3. **Review outputs carefully**
4. **Decide**: Deploy, retrain, or test more

---

## 🔧 If Retrain is Needed

### **Conservative Config** (safer, slower learning)
```yaml
# configs/dpo_config.yaml
beta: 0.1  # Reduced from 0.2
learning_rate: 5.0e-6  # Reduced from 1e-5
label_smoothing: 0.1  # Increased from 0.05
max_length: 1024  # Keep
```

### **Fix Tokenizer**
```python
# scripts/15_train_dpo.py
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    fix_mistral_regex=True  # CRITICAL FIX
)
```

### **Retrain Command**
```bash
rm -rf checkpoints/dpo_lora/*
python scripts/15_train_dpo.py
```

---

## 📋 Summary

**Current Status**: Training proceeding, but GPT raised valid concerns about potential reward hacking.

**Critical Issue**: Tokenizer regex mismatch + PAD==EOS + suspicious Δlogp gap.

**Next Steps**:
1. ✅ Let training complete
2. ⏳ Run diagnostics
3. ⏳ Validate generation quality
4. ⏳ Decide: deploy vs retrain

**DO NOT deploy without validation!**
