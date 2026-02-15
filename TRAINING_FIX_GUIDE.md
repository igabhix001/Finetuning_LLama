# DPO Training Fix Guide

## 🔴 Problems Identified in Your Training

Your training showed these critical issues:

```
rewards/margins: -0.078  ❌ (should be positive ~0.5-2.0)
rewards/accuracies: 0.40625  ❌ (should be >0.6)
learning_rate: 2.7e-07  ❌ (too low, barely learning)
```

**Root Causes**:
1. **Beta too high** (0.5) → Over-constraining the model
2. **Max length too short** (1024) → Truncating chart YAML
3. **Learning rate too low** (2e-6) → Not learning effectively
4. **Max prompt length too short** (512) → Chart YAML is ~1500 chars

---

## ✅ Fixed Configuration

I've created `configs/dpo_config_FIXED.yaml` with corrected hyperparameters:

| Parameter | Old (Broken) | New (Fixed) | Reason |
|-----------|--------------|-------------|--------|
| `beta` | 0.5 | **0.1** | Standard DPO beta, less constraint |
| `learning_rate` | 2e-6 | **5e-6** | 2.5x higher for better learning |
| `max_length` | 1024 | **2048** | Chart YAML needs space |
| `max_prompt_length` | 512 | **1536** | Chart YAML is ~1500 chars |
| `label_smoothing` | 0.1 | **0.0** | Remove noise for clearer signal |
| `weight_decay` | 0.05 | **0.01** | Lighter regularization |
| `max_grad_norm` | 0.3 | **1.0** | Allow larger gradients |
| `eval_steps` | 10 | **50** | Less frequent eval |
| `warmup_ratio` | 0.15 | **0.1** | Shorter warmup |

---

## 🚀 Corrected Training Commands

### Step 1: Stop Current Training (Already Done)
```bash
# You already interrupted it with Ctrl+C ✓
```

### Step 2: Clean Up Failed Checkpoint
```bash
cd /workspace/Finetuning_LLama
rm -rf checkpoints/dpo_lora/*
```

### Step 3: Start Training with Fixed Config
```bash
python scripts/15_train_dpo.py --config configs/dpo_config_FIXED.yaml
```

### Expected Training Metrics (Fixed)
```
Step 10:
  rewards/margins: 0.5-1.0  ✅ (positive, model learning)
  rewards/accuracies: 0.6-0.7  ✅ (better than random)
  learning_rate: 5e-6  ✅ (effective learning)
  loss: 0.65-0.70  ✅ (decreasing)

Step 50:
  rewards/margins: 1.0-1.5  ✅ (improving)
  rewards/accuracies: 0.7-0.8  ✅ (good preference)
  loss: 0.55-0.60  ✅ (converging)

Step 100:
  rewards/margins: 1.5-2.0  ✅ (strong preference)
  rewards/accuracies: 0.75-0.85  ✅ (excellent)
  loss: 0.45-0.50  ✅ (well-trained)
```

---

## 📊 Monitoring Training

### Watch Training Progress
```bash
# In another terminal
tail -f logs/dpo/training.log
```

### TensorBoard (Optional)
```bash
tensorboard --logdir=logs/dpo --port 6006
# Then open: http://localhost:6006
```

### GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

---

## ⏱️ Expected Training Time

- **Total steps**: ~429 (2,294 samples / 16 batch size × 3 epochs)
- **Time per step**: ~25-30 seconds (RTX 6000 Ada)
- **Total time**: ~3-4 hours

**Checkpoints saved every 50 steps** in `checkpoints/dpo_lora/`

---

## 🎯 When Training Completes

### Step 1: Verify Best Checkpoint
```bash
ls -lh checkpoints/dpo_lora/
# Look for checkpoint with lowest eval_loss
```

### Step 2: Merge DPO LoRA into SFT Model
```bash
python scripts/16_merge_dpo_lora.py \
  --base_model ./models/merged_sft \
  --lora_model ./checkpoints/dpo_lora/checkpoint-XXX \
  --output_dir ./models/llama-3.1-8b-dpo-final
```

### Step 3: Test Final Model
```bash
python scripts/test_model.py \
  --model ./models/llama-3.1-8b-dpo-final \
  --kundali sample_kundali/kundali_Amit_Kumar.json \
  --question "When will I get married?"
```

---

## 📤 Upload to HuggingFace

I've created `upload_to_huggingface.py` for easy uploading.

### Step 1: Login to HuggingFace
```bash
huggingface-cli login
# Or set HF_TOKEN environment variable
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

### Step 2: Upload Final Model
```bash
python upload_to_huggingface.py \
  --model ./models/llama-3.1-8b-dpo-final \
  --repo YOUR_USERNAME/kp-astrology-dpo-v4
```

### Step 3: Upload DPO LoRA (Optional, for reproducibility)
```bash
python upload_to_huggingface.py \
  --model ./checkpoints/dpo_lora/checkpoint-XXX \
  --repo YOUR_USERNAME/kp-astrology-dpo-lora
```

### Step 4: Make Private (Optional)
```bash
python upload_to_huggingface.py \
  --model ./models/llama-3.1-8b-dpo-final \
  --repo YOUR_USERNAME/kp-astrology-dpo-v4 \
  --private
```

---

## 🔍 Troubleshooting

### If Training Still Shows Negative Margins
```bash
# Stop training (Ctrl+C)
# Increase learning rate further
# Edit configs/dpo_config_FIXED.yaml:
learning_rate: 8.0e-6  # Even higher

# Restart training
python scripts/15_train_dpo.py --config configs/dpo_config_FIXED.yaml
```

### If CUDA Out of Memory
```bash
# Reduce batch size
# Edit configs/dpo_config_FIXED.yaml:
per_device_train_batch_size: 1
gradient_accumulation_steps: 8  # Reduce from 16
max_length: 1536  # Reduce from 2048
```

### If Training is Too Slow
```bash
# Reduce max_length (if YAML fits)
max_length: 1536

# Or use fp16 instead of bf16
fp16: true
bf16: false
```

---

## 📋 Quick Reference

### Files Created
- ✅ `configs/dpo_config_FIXED.yaml` - Corrected training config
- ✅ `upload_to_huggingface.py` - HuggingFace upload script
- ✅ `TRAINING_FIX_GUIDE.md` - This guide

### Key Changes
- ✅ Beta: 0.5 → 0.1 (less constraint)
- ✅ Learning rate: 2e-6 → 5e-6 (better learning)
- ✅ Max length: 1024 → 2048 (no truncation)
- ✅ Max prompt: 512 → 1536 (full chart YAML)

### Training Command
```bash
python scripts/15_train_dpo.py --config configs/dpo_config_FIXED.yaml
```

### Upload Command
```bash
python upload_to_huggingface.py \
  --model ./models/llama-3.1-8b-dpo-final \
  --repo YOUR_USERNAME/kp-astrology-dpo-v4
```

---

**Status**: Ready to restart training with fixed configuration ✅
