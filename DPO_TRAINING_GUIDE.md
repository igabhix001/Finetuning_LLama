# DPO Training on RunPod — Complete Setup Guide

## Overview

Your merged DAPT+SFT model is at: `igabhix001/kp-astrology-llama-8b`
DPO dataset: 999 pairs (895 train / 100 eval)

Pipeline:
1. Install deps + prepare DPO dataset
2. DPO training (LoRA on merged model)
3. Merge DPO LoRA → final model
4. Quantize with Unsloth (GGUF)
5. Serve with vLLM + inference test

---

## Step 1: Install Dependencies

```bash
cd /workspace/Finetuning_LLama

# Core training deps
pip install trl>=0.9.0 peft transformers>=4.41.0 datasets accelerate bitsandbytes
pip install tensorboard

# For quantization (Step 4)
pip install unsloth

# For serving (Step 5)
pip install vllm
```

## Step 2: Prepare DPO Dataset

The DPO dataset (`data/dpo/dpo_pairs.jsonl`) should already be in the repo from git pull.

```bash
python scripts/14_prepare_dpo_dataset.py
```

This creates:
- `data/dpo/prepared/train/` — 895 pairs
- `data/dpo/prepared/test/` — 100 pairs

## Step 3: Download Merged Model (if not already done)

```bash
# Your model is already downloaded at ./models/merged_sft/
# If you need to re-download:
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='igabhix001/kp-astrology-llama-8b',
    local_dir='./models/merged_sft/'
)
print('Done')
"

# Verify:
ls ./models/merged_sft/*.safetensors
```

## Step 4: Run DPO Training

```bash
python scripts/15_train_dpo.py
```

This will:
1. Load `./models/merged_sft/` (your DAPT+SFT merged model)
2. Apply a fresh LoRA (rank 8, alpha 16) for DPO
3. Train for 2 epochs on 895 pairs with eval on 100 pairs
4. Save best checkpoint to `./checkpoints/dpo_lora/`
5. Save final adapter to `./checkpoints/dpo_lora/final/`

### Training Config (`configs/dpo_config.yaml`):
| Parameter | Value |
|-----------|-------|
| Beta | 0.1 |
| Loss type | sigmoid |
| Epochs | 2 |
| Batch size | 1 x 8 grad accum = 8 effective |
| Learning rate | 5e-5 |
| LoRA rank | 8 |
| Max length | 1024 |
| Optimizer | paged_adamw_8bit |
| Precision | bf16 |

Expected time: ~30-60 min on RTX 6000 Ada (48GB).

Monitor: `tensorboard --logdir=./logs/dpo/ --port 6006`

## Step 5: Merge DPO LoRA into Final Model

```bash
python scripts/16_merge_dpo_lora.py
```

Output: `./models/final_dpo/` (~16GB, bf16 safetensors)

## Step 6: Quantize with Unsloth

```bash
# Q4_K_M (recommended — small + fast, ~4.5GB)
python scripts/06_quantize_unsloth.py --model ./models/final_dpo/ --method q4_k_m

# Or Q8_0 (higher quality, ~8.5GB)
python scripts/06_quantize_unsloth.py --model ./models/final_dpo/ --method q8_0

# Or both
python scripts/06_quantize_unsloth.py --model ./models/final_dpo/ --method both
```

Output: `./models/quantized/` (GGUF files)

**Note**: If Unsloth fails, you can skip quantization and serve the bf16 model directly with vLLM.

## Step 7: Serve with vLLM

```bash
# Serve the bf16 final model (recommended for RTX 6000 Ada with 48GB)
python scripts/08_serve_vllm.py --model-path ./models/final_dpo/

# Or serve quantized if available
# python scripts/08_serve_vllm.py --model-path ./models/quantized/
```

Server starts at `http://0.0.0.0:8000/v1` (OpenAI-compatible API).

## Step 8: Inference Test

Once vLLM is running, test in a **separate terminal**:

```bash
# Quick health check
curl http://localhost:8000/health

# Chat completion test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kp-astrology-llama",
    "messages": [
      {"role": "system", "content": "You are Jyotish, a KP astrologer."},
      {"role": "user", "content": "Who are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

Or use the Gradio chat UI (connects to vLLM backend):
```bash
python scripts/09_chat_ui.py --model ./models/final_dpo/ --no-rag --port 7860
```

## Step 9: Push Final Model to HuggingFace (Optional)

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('igabhix001/kp-astrology-llama-8b-dpo', exist_ok=True)
api.upload_folder(
    folder_path='./models/final_dpo/',
    repo_id='igabhix001/kp-astrology-llama-8b-dpo',
    commit_message='Final DPO model (DAPT+SFT+DPO merged)'
)
print('Pushed')
"
```

---

## Quick Reference — Copy-Paste Command Sequence

```bash
cd /workspace/Finetuning_LLama

# 1. Install deps
pip install trl>=0.9.0 peft transformers>=4.41.0 datasets accelerate bitsandbytes unsloth vllm

# 2. Prepare dataset
python scripts/14_prepare_dpo_dataset.py

# 3. DPO training (~30-60 min)
python scripts/15_train_dpo.py

# 4. Merge DPO LoRA
python scripts/16_merge_dpo_lora.py

# 5. Quantize (optional)
python scripts/06_quantize_unsloth.py --model ./models/final_dpo/

# 6. Serve
python scripts/08_serve_vllm.py --model-path ./models/final_dpo/

# 7. Test (separate terminal)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kp-astrology-llama","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":256}'
```

---

## Troubleshooting

**OOM during training**: Reduce `per_device_train_batch_size` to 1 and increase `gradient_accumulation_steps` in `configs/dpo_config.yaml`.

**DPO dataset not found**: Run `python scripts/14_prepare_dpo_dataset.py` first.

**Unsloth import error**: `pip install unsloth`. If still fails, skip quantization — serve bf16 directly with vLLM.

**vLLM OOM**: Add `--gpu-memory-utilization 0.85` or reduce `--max-model-len 1024`.

**Tokenizer warnings**: Safe to ignore. The script sets `pad_token = eos_token` automatically.
