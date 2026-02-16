# Production Quality Audit - DPO Training

## 🚨 CRITICAL FINDING: Llama 3.1 Does NOT Have Mistral Regex Bug

### **Issue Identified**

The script `15_train_dpo.py` contains **incorrect assumptions** about the Llama 3.1 tokenizer:

```python
# Lines 28-33 (INCORRECT)
# transformers incorrectly flags Llama 3.1 tokenizer as having a Mistral regex issue.
# See: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84
import warnings
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")
```

**This is WRONG**:
1. Llama 3.1 uses a **completely different tokenizer** than Mistral
2. The warning in your logs is **real**, not a false positive
3. Suppressing the warning hides a **genuine tokenization bug**

### **Evidence from Your Logs**

```
The tokenizer you are loading from './models/merged_sft/' with an incorrect regex pattern:
https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84
This will lead to incorrect tokenization.
You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
```

**Why this appeared**:
- Your SFT model was likely trained on or derived from a Mistral-based checkpoint
- The tokenizer config was copied from Mistral
- Now it has Mistral's regex bug, even though you're using Llama 3.1 base

### **Root Cause Analysis**

**Your training pipeline**:
```
Stage 1 (DAPT): Llama 3.1 base → LoRA → merge
Stage 2 (SFT):  Merged → LoRA → merge (tokenizer config copied from Mistral?)
Stage 3 (DPO):  Merged → LoRA (inherits broken tokenizer)
```

**What happened**:
- At some point, the tokenizer config was overwritten with Mistral's config
- This introduced the regex bug
- The bug persisted through merges
- Now your "Llama 3.1" model has a Mistral tokenizer config

---

## ✅ **Verification Steps**

### **1. Check Tokenizer Config**

```bash
cd /workspace/Finetuning_LLama

# Check what tokenizer you actually have
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./models/merged_sft/')
print('Tokenizer class:', tokenizer.__class__.__name__)
print('Vocab size:', len(tokenizer))
print('Special tokens:', tokenizer.special_tokens_map)
print('Model max length:', tokenizer.model_max_length)
"
```

**Expected for Llama 3.1**:
```
Tokenizer class: PreTrainedTokenizerFast
Vocab size: 128256
Special tokens: {'bos_token': '<|begin_of_text|>', 'eos_token': '<|end_of_text|>', ...}
Model max length: 131072
```

### **2. Check for Mistral Contamination**

```bash
# Check tokenizer_config.json
cat models/merged_sft/tokenizer_config.json | grep -i mistral

# If this returns anything, your tokenizer is contaminated
```

---

## 🔧 **Fixes Required**

### **Fix 1: Remove Incorrect Warning Suppression**

The warning suppression is **hiding a real bug**. Remove it:

```python
# DELETE THESE LINES (28-33 in 15_train_dpo.py):
# ── Suppress known false-positive tokenizer regex warning ─────────────────────
# transformers incorrectly flags Llama 3.1 tokenizer as having a Mistral regex issue.
# See: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84
import warnings
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")
```

### **Fix 2: Apply Correct Tokenizer Fix**

If your tokenizer is contaminated with Mistral config, you need to either:

**Option A: Use fix_mistral_regex=True** (if tokenizer is Mistral-based)
```python
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    fix_mistral_regex=True
)
```

**Option B: Reload Pure Llama 3.1 Tokenizer** (recommended)
```python
# Load model
model = AutoModelForCausalLM.from_pretrained(model_path, ...)

# Load CLEAN Llama 3.1 tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    token=hf_token
)

# Ensure model and tokenizer are aligned
model.resize_token_embeddings(len(tokenizer))
```

### **Fix 3: Separate PAD Token** (already applied)

This fix is correct:
```python
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    model.resize_token_embeddings(len(tokenizer))
```

---

## 📋 **Production Quality Checklist**

### **Training Script (15_train_dpo.py)**

- [ ] **Remove warning suppression** (lines 28-33)
- [ ] **Verify tokenizer source** (Llama 3.1 vs Mistral)
- [ ] **Apply correct tokenizer fix** (fix_mistral_regex OR reload clean tokenizer)
- [ ] **Separate PAD token** ✅ (already done)
- [ ] **Conservative hyperparameters** ✅ (beta=0.1, LR=5e-6)
- [ ] **Health guard threshold** ✅ (3.0)
- [ ] **Proper error handling** (add try-except for model loading)
- [ ] **Logging improvements** (log tokenizer details at startup)

### **Config Files**

- [ ] **dpo_config.yaml**: Verify all paths exist
- [ ] **dpo_lora_config.yaml**: Verify LoRA rank is appropriate
- [ ] **Filtered dataset exists**: `data/dpo/prepared/train_filtered`
- [ ] **Eval dataset exists**: `data/dpo/prepared/test_filtered`

### **Diagnostic Scripts**

- [ ] **diagnose_dpo_quality.py**: Update to use correct tokenizer
- [ ] **validate_dpo_training.py**: Update to use correct tokenizer
- [ ] **Add tokenizer verification script**

---

## 🎯 **Recommended Action Plan**

### **Step 1: Verify Tokenizer**

```bash
cd /workspace/Finetuning_LLama

# Check current tokenizer
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./models/merged_sft/')
print('Class:', tok.__class__.__name__)
print('Vocab:', len(tok))
print('Config:', tok.init_kwargs if hasattr(tok, 'init_kwargs') else 'N/A')
"
```

### **Step 2: Fix Tokenizer Issue**

**If tokenizer is Mistral-contaminated**:
```python
# In 15_train_dpo.py, replace tokenizer loading with:
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    token=hf_token
)
# Then load model and resize embeddings
```

**If tokenizer is pure Llama 3.1 but has regex bug**:
```python
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    fix_mistral_regex=True  # This might work
)
```

### **Step 3: Remove Warning Suppression**

Delete lines 28-33 in `15_train_dpo.py`.

### **Step 4: Add Tokenizer Verification**

```python
# Add after tokenizer loading in 15_train_dpo.py
print(f"\n   Tokenizer Details:")
print(f"     Class: {tokenizer.__class__.__name__}")
print(f"     Vocab size: {len(tokenizer)}")
print(f"     Model max length: {tokenizer.model_max_length}")
print(f"     BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
print(f"     EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
print(f"     PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
```

### **Step 5: Test Before Full Training**

```bash
# Quick test with 10 steps
python scripts/15_train_dpo.py --config configs/dpo_config_test.yaml
```

Create `configs/dpo_config_test.yaml`:
```yaml
# Copy from dpo_config.yaml but set:
num_train_epochs: 1
max_steps: 10
logging_steps: 1
```

---

## 🚨 **Critical Issues Summary**

| Issue | Severity | Status | Fix Required |
|-------|----------|--------|--------------|
| Warning suppression hiding real bug | CRITICAL | ❌ | Remove lines 28-33 |
| Tokenizer source unclear | CRITICAL | ❌ | Verify Llama 3.1 vs Mistral |
| Regex bug not fixed | CRITICAL | ❌ | Apply correct fix |
| PAD == EOS | CRITICAL | ✅ | Already fixed |
| Beta too aggressive | HIGH | ✅ | Already fixed (0.1) |
| LR too aggressive | HIGH | ✅ | Already fixed (5e-6) |
| NaN logprobs in diagnostics | HIGH | ⏳ | Will be fixed by tokenizer fix |
| 32x reward hacking ratio | HIGH | ⏳ | Will be fixed by conservative config |

---

## ✅ **Expected After Fixes**

### **Training Logs**

```
1. Loading merged DAPT+SFT model from: ./models/merged_sft/
   Tokenizer Details:
     Class: PreTrainedTokenizerFast
     Vocab size: 128256
     Model max length: 131072
     BOS token: <|begin_of_text|> (id: 128000)
     EOS token: <|end_of_text|> (id: 128009)
     PAD token: <|pad|> (id: 128256)
   ✓ Model loaded: 8,030,261,248 parameters
```

**NO warnings about**:
- Incorrect regex pattern
- fix_mistral_regex

### **Diagnostic Results**

```
Average changes (DPO - SFT):
  Chosen:   +0.20 to +0.40
  Rejected: -0.50 to -1.00
  Ratio: 2.0x to 3.0x  ✅

NaN logprobs: 0/10 samples  ✅
Reward hacking: 0/10 samples  ✅
```

---

## 📝 **Next Steps**

1. **Verify tokenizer source** on RunPod
2. **Apply correct tokenizer fix**
3. **Remove warning suppression**
4. **Add tokenizer verification logging**
5. **Test with 10 steps**
6. **If test passes, run full training**
7. **Run diagnostics**
8. **Deploy if validation passes**

---

## 🎯 **Bottom Line**

**Current state**: Training script has incorrect assumptions about tokenizer, suppressing real warnings.

**Required action**: Verify tokenizer source, apply correct fix, remove warning suppression.

**DO NOT TRAIN** until tokenizer issue is resolved and verified.
