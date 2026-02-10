# DPO Training on RunPod — Complete Setup Guide

## Overview

You have completed DAPT (Stage 1) and SFT (Stage 2) on a previous RunPod session.
Now you need to:
1. Set up a new RunPod pod
2. Pull latest code (with new DPO dataset + updated prompts)
3. Fetch your DAPT+SFT models from HuggingFace
4. Merge them into a single base for DPO
5. Run DPO training (Stage 3)
6. Merge DPO LoRA into final model

---

## Step 1: RunPod Setup

Spin up a pod with:
- **GPU**: RTX 6000 Ada (48GB VRAM) or A100 (80GB)
- **Image**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- **Disk**: 100GB+ (model is ~16GB, training needs headroom)

## Step 2: Clone Repo & Install Dependencies

```bash
cd /workspace
git clone https://github.com/<your-repo>/Dataset_preprossecing_pipeline.git
cd Dataset_preprossecing_pipeline/Finetuning_LLama

# Install dependencies
pip install -r requirements.txt
pip install trl>=0.9.0 peft transformers datasets accelerate bitsandbytes
pip install tensorboard

# Set up environment
cp .env.example .env   # or create manually
# Edit .env and add:
#   HF_TOKEN=hf_xxxxxxxxxxxxx
#   OPENAI_API_KEY=sk-xxxxxxxxxxxxx  (only if generating DPO data on pod)
```

## Step 3: Download DPO Dataset (already generated)

The DPO dataset (`data/dpo/dpo_pairs.jsonl`) is generated locally via OpenAI Batch API.
Push it to git before pulling on RunPod:

```bash
# ON YOUR LOCAL MACHINE (Windows):
cd D:\Dataset_preprossecing_pipeline
git add Finetuning_LLama/data/dpo/dpo_pairs.jsonl
git commit -m "Add DPO v2 dataset (1100 pairs, concise responses)"
git push

# ON RUNPOD:
cd /workspace/Dataset_preprossecing_pipeline
git pull
```

Then prepare the dataset for training:
```bash
cd /workspace/Dataset_preprossecing_pipeline/Finetuning_LLama
python scripts/14_prepare_dpo_dataset.py
```
This creates:
- `data/dpo/prepared/train/` — 990 pairs (HuggingFace Dataset format)
- `data/dpo/prepared/test/` — 110 pairs

## Step 4: Fetch DAPT + SFT Models from HuggingFace

You previously trained and pushed DAPT and SFT LoRA adapters to HuggingFace.
Download them:

```bash
# Option A: If you pushed the MERGED model to HuggingFace
# (check your HF repos for the merged DAPT+SFT model)
python -c "
from huggingface_hub import snapshot_download
import os
token = os.getenv('HF_TOKEN')

# Download merged DAPT+SFT model (if you pushed it)
snapshot_download(
    repo_id='<your-hf-username>/jyotish-llama3.1-8b-sft-merged',
    local_dir='./models/merged_sft/',
    token=token
)
print('✓ Merged SFT model downloaded')
"

# Option B: If you only pushed individual LoRA adapters
# Download base + both LoRAs and merge locally
python -c "
from huggingface_hub import snapshot_download
import os
token = os.getenv('HF_TOKEN')

# Download DAPT LoRA
snapshot_download(
    repo_id='<your-hf-username>/jyotish-dapt-lora',
    local_dir='./checkpoints/dapt_lora/final/',
    token=token
)
print('✓ DAPT LoRA downloaded')

# Download SFT LoRA
snapshot_download(
    repo_id='<your-hf-username>/jyotish-sft-lora',
    local_dir='./checkpoints/sft_lora/final/',
    token=token
)
print('✓ SFT LoRA downloaded')
"

# Then merge them:
python scripts/05b_merge_sft_lora.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --dapt-lora ./checkpoints/dapt_lora/final/ \
    --sft-lora ./checkpoints/sft_lora/final/ \
    --output ./models/merged_sft/
```

**Important**: Replace `<your-hf-username>` and repo names with your actual HuggingFace repo IDs.
Check your HF account at https://huggingface.co/settings/tokens to find your repos.

### How to find your HF model repos:
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv('HF_TOKEN'))
models = api.list_models(author=api.whoami()['name'])
for m in models:
    print(f'  {m.modelId} ({m.lastModified})')
"
```

## Step 5: Verify Merged Model Exists

```bash
ls -la ./models/merged_sft/
# Should contain: config.json, tokenizer files, model-*.safetensors
```

The DPO config (`configs/dpo_config.yaml`) expects the merged model at `./models/merged_sft/`.

## Step 6: Run DPO Training

```bash
python scripts/15_train_dpo.py
```

This will:
1. Load the merged DAPT+SFT model from `./models/merged_sft/`
2. Apply a fresh LoRA (rank 8, alpha 16) for DPO
3. Train for 2 epochs on 990 pairs with eval on 110 pairs
4. Save checkpoints to `./checkpoints/dpo_lora/`
5. Save final adapter to `./checkpoints/dpo_lora/final/`

### Training Config Summary (from `configs/dpo_config.yaml`):
| Parameter | Value |
|-----------|-------|
| Beta | 0.1 |
| Loss type | sigmoid |
| Epochs | 2 |
| Batch size | 1 × 8 grad accum = 8 effective |
| Learning rate | 5e-5 |
| LoRA rank | 8 |
| Max length | 1024 |
| Optimizer | paged_adamw_8bit |
| Precision | bf16 |

### Monitor Training:
```bash
# In a separate terminal:
tensorboard --logdir=./logs/dpo/ --port 6006
```

Expected training time: ~30-60 minutes on RTX 6000 Ada (48GB).

## Step 7: Merge DPO LoRA into Final Model

```bash
python scripts/16_merge_dpo_lora.py
```

This merges the DPO LoRA into the DAPT+SFT base, producing the final model at `./models/final_dpo/`.

## Step 8: Push Final Model to HuggingFace (Optional)

```bash
python -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv('HF_TOKEN'))
api.create_repo('<your-hf-username>/jyotish-llama3.1-8b-dpo-final', exist_ok=True)
api.upload_folder(
    folder_path='./models/final_dpo/',
    repo_id='<your-hf-username>/jyotish-llama3.1-8b-dpo-final',
    commit_message='Final DPO model (DAPT+SFT+DPO merged)'
)
print('✓ Final model pushed to HuggingFace')
"
```

## Step 9: Test Inference

```bash
# Quick test with Gradio UI
python scripts/09_chat_ui.py \
    --model ./models/final_dpo/ \
    --no-rag \
    --port 7860

# Or API server
python scripts/11_api_server.py \
    --model ./models/final_dpo/ \
    --no-rag \
    --port 8000
```

---

## Quick Reference — Full Command Sequence

```bash
# 1. Setup
cd /workspace/Dataset_preprossecing_pipeline/Finetuning_LLama
pip install -r requirements.txt
pip install trl>=0.9.0 peft transformers datasets accelerate bitsandbytes

# 2. Pull latest code + DPO data
git pull

# 3. Prepare DPO dataset
python scripts/14_prepare_dpo_dataset.py

# 4. Fetch models (use Option A or B from Step 4 above)
# ... download merged_sft or merge from LoRAs ...

# 5. Train DPO
python scripts/15_train_dpo.py

# 6. Merge final
python scripts/16_merge_dpo_lora.py

# 7. Test
python scripts/09_chat_ui.py --model ./models/final_dpo/ --no-rag
```

---

## Troubleshooting

**OOM during training**: Reduce `per_device_train_batch_size` to 1 and increase `gradient_accumulation_steps` in `configs/dpo_config.yaml`.

**Model not found on HF**: Run the `list_models` snippet from Step 4 to find your exact repo names.

**DPO dataset not found**: Make sure you ran `python scripts/14_prepare_dpo_dataset.py` and `data/dpo/prepared/train/` exists.

**Tokenizer warnings**: Safe to ignore. The script sets `pad_token = eos_token` automatically.
