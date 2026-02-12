# Training Guide — Authoritative Runbook

> **SOURCE OF TRUTH.** All paths, configs, and scripts referenced here match the
> actual repo. Do NOT follow older docs that reference `/workspace/data/arrow/`
> or inline script snippets — those are outdated.

**Prerequisites:** RunPod environment with RTX 6000 Ada (48 GB VRAM). Complete `RUNPOD_SETUP.md` first.

---

## Pipeline Overview

```
Stage 0: (Optional) Re-normalize SFT dataset
Stage 1: DAPT LoRA  →  checkpoints/dapt_lora/final/
Stage 2: SFT LoRA   →  checkpoints/sft_lora/final/
Stage 3: Merge DAPT+SFT into single base  →  models/merged_sft/
Stage 4: DPO dataset generation + preparation
Stage 5: DPO LoRA   →  checkpoints/dpo_lora/final/
Stage 6: Merge DPO into final model  →  models/final_dpo/
Stage 7: Export for vLLM  →  models/quantized/
Stage 8: Serve + Test
```

**Total time:** ~12-20 hours on RTX 6000 Ada
**LoRA staging rule:** NEVER stack LoRA on LoRA. Always merge before applying the next LoRA.

---

## Stage 0 (Optional): Re-normalize SFT Dataset

If the SFT dataset contains markdown, headers, or "the native" phrasing, clean it first:

```bash
cd /workspace/Finetuning_LLama

# Preview changes (no writes)
python scripts/17_renormalize_sft_dataset.py --dry-run

# Apply normalization (creates backup automatically)
python scripts/17_renormalize_sft_dataset.py
```

- **Input:** `data/sft_train/`, `data/sft_validation/`
- **Output:** same paths (in-place), backups at `data/sft_train_backup/`

---

## Stage 1: DAPT Training

```bash
cd /workspace/Finetuning_LLama
python scripts/03_train_dapt.py
```

- **Config:** `configs/dapt_config.yaml` + `configs/dapt_lora_config.yaml`
- **Dataset:** `data/dapt_corpus/` (654 examples)
- **Base model:** `meta-llama/Llama-3.1-8B-Instruct` (downloaded via HF_TOKEN)
- **Output:** `checkpoints/dapt_lora/final/`
- **LoRA rank:** 16, alpha: 32, targets: q/k/v/o/gate/up/down_proj
- **Time:** ~2-4 hours
- **Monitor:** `tensorboard --logdir=logs/dapt/`

---

## Stage 2: SFT Training

```bash
python scripts/04_train_sft.py
```

- **Config:** `configs/sft_config.yaml` + `configs/lora_config.yaml`
- **Datasets:** `data/sft_train/` (19,303), `data/sft_validation/` (398)
- **Base:** loads base model + merges DAPT LoRA in-memory, then applies SFT LoRA
- **Output:** `checkpoints/sft_lora/final/`
- **LoRA rank:** 16, alpha: 32
- **Time:** ~6-10 hours
- **Monitor:** `tensorboard --logdir=logs/sft/`

---

## Stage 3: Merge DAPT+SFT → Single Base

```bash
python scripts/05b_merge_sft_lora.py
```

- **Input:** base model + `checkpoints/dapt_lora/final/` + `checkpoints/sft_lora/final/`
- **Output:** `models/merged_sft/`
- **Time:** ~30 minutes

---

## Stage 4: DPO Dataset

### 4a: Generate DPO pairs

```bash
python scripts/13_generate_dpo_dataset.py --count 1100 --model gpt-4o-mini
```

- **Output:** `data/dpo/dpo_pairs.jsonl`
- Uses `chart_preprocessor.chart_to_yaml()` for format consistency

### 4b: Prepare for DPOTrainer

```bash
python scripts/14_prepare_dpo_dataset.py
```

- **Output:** `data/dpo/prepared/` (HF Dataset with train/test split)
- Now preserves `category` and `chart_name` metadata columns

---

## Stage 5: DPO Training

```bash
python scripts/15_train_dpo.py
```

- **Config:** `configs/dpo_config.yaml` + `configs/dpo_lora_config.yaml`
- **Dataset:** `data/dpo/prepared/` (895 train / 100 eval)
- **Base model:** `models/merged_sft/` (merged DAPT+SFT)
- **Output:** `checkpoints/dpo_lora/final/`
- **LoRA rank:** 8, alpha: 8 (conservative for refinement)
- **DPO beta:** 0.1, loss: sigmoid
- **Time:** ~2-4 hours
- **Monitor:** `tensorboard --logdir=logs/dpo/`

---

## Stage 6: Merge DPO → Final Model

```bash
python scripts/16_merge_dpo_lora.py
```

- **Input:** `models/merged_sft/` + `checkpoints/dpo_lora/final/`
- **Output:** `models/final_dpo/`

---

## Stage 7: Export for vLLM

```bash
# Default: safetensors copy for vLLM (recommended)
python scripts/06_quantize_unsloth.py --model ./models/final_dpo/

# Or explicit:
python scripts/06_quantize_unsloth.py --method safetensors --model ./models/final_dpo/
```

- **Output:** `models/quantized/` (safetensors, ready for vLLM)
- For llama.cpp only: `--method q4_k_m` or `--method q8_0`

---

## Stage 8: Serve & Test

### Start vLLM

```bash
python scripts/08_serve_vllm.py --model-path ./models/final_dpo/
# Or with quantized:
python scripts/08_serve_vllm.py --model-path ./models/quantized/
# dtype is configurable: --dtype auto (default), bfloat16, float16
```

### Start UI or API

```bash
# Gradio UI (port 7860)
python scripts/09_chat_ui.py

# FastAPI (port 8080)
python scripts/11_api_server.py
```

### Run evaluation

```bash
# Standard test suite (compact chart data)
python scripts/10_kp_test_suite.py

# Production-mirror eval (full kundali JSON → YAML → vLLM → format checks)
python scripts/10_kp_test_suite.py --kundali-json ../sample_kundali/kundali_Abhi_Raj.json
```

---

## Config Files Reference

| File | Purpose |
|------|---------|
| `configs/dapt_config.yaml` | DAPT training hyperparameters |
| `configs/dapt_lora_config.yaml` | DAPT LoRA rank/alpha/targets |
| `configs/sft_config.yaml` | SFT training hyperparameters |
| `configs/lora_config.yaml` | SFT LoRA rank/alpha/targets |
| `configs/dpo_config.yaml` | DPO training hyperparameters |
| `configs/dpo_lora_config.yaml` | DPO LoRA rank/alpha/targets |

---

## Troubleshooting

- **OOM during training:** reduce `per_device_train_batch_size` or enable `gradient_checkpointing: true`
- **vLLM context length error:** reduce `--max-model-len` or trim chart YAML
- **TRL not found:** `pip install -r requirements.txt` (TRL is now pinned)
- **GGUF vs safetensors confusion:** vLLM uses safetensors. GGUF is for llama.cpp only.
