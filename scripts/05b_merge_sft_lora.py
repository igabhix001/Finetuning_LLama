"""
Merge SFT LoRA into Base+DAPT Model (Stage 2 → Stage 3 Bridge)
================================================================
Following the correct LoRA staging approach:
  Stage 1 (DAPT): Base → train LoRA → merge into base
  Stage 2 (SFT):  Merged-DAPT → train LoRA → merge into model  ← THIS SCRIPT
  Stage 3 (DPO):  Merged-SFT → train NEW LoRA (never stack LoRA on LoRA)

This script:
1. Loads base Llama 3.1 8B Instruct
2. Merges DAPT LoRA
3. Merges SFT LoRA
4. Saves the fully merged model ready for DPO LoRA training

Usage:
  python scripts/05b_merge_sft_lora.py
  python scripts/05b_merge_sft_lora.py --output ./models/merged_sft/
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="Merge SFT LoRA into base model for DPO")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--dapt-lora", type=str, default="./checkpoints/dapt_lora/final/")
parser.add_argument("--sft-lora", type=str, default="./checkpoints/sft_lora/final/")
parser.add_argument("--output", type=str, default="./models/merged_sft/")
parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
args = parser.parse_args()

print("=" * 80)
print("MERGE SFT LORA → BASE+DAPT MODEL (Stage 2→3 Bridge)")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Base model: {args.base_model}")
print(f"DAPT LoRA: {args.dapt_lora}")
print(f"SFT LoRA: {args.sft_lora}")
print(f"Output: {args.output}")
print("=" * 80)

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN not found in .env")
    sys.exit(1)

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
model_dtype = dtype_map[args.dtype]

# 1. Load base model
print("\n1. Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    token=hf_token,
    torch_dtype=model_dtype,
    device_map="auto",
    trust_remote_code=True,
)
print(f"   ✓ Base model loaded: {model.num_parameters():,} parameters")

# 2. Merge DAPT LoRA
dapt_path = Path(args.dapt_lora)
if dapt_path.exists():
    print("\n2. Merging DAPT LoRA...")
    model = PeftModel.from_pretrained(model, str(dapt_path))
    model = model.merge_and_unload()
    print("   ✓ DAPT LoRA merged")
else:
    print(f"\n2. DAPT LoRA not found at {dapt_path}, skipping...")

# 3. Merge SFT LoRA
sft_path = Path(args.sft_lora)
if not sft_path.exists():
    print(f"❌ SFT LoRA not found: {sft_path}")
    sys.exit(1)

print("\n3. Merging SFT LoRA...")
model = PeftModel.from_pretrained(model, str(sft_path))
model = model.merge_and_unload()
print("   ✓ SFT LoRA merged")

# 4. Save merged model
print(f"\n4. Saving merged model to {args.output}...")
output_path = Path(args.output)
output_path.mkdir(parents=True, exist_ok=True)

model.save_pretrained(str(output_path), safe_serialization=True, max_shard_size="5GB")
tokenizer.save_pretrained(str(output_path))

model_size = sum(p.stat().st_size for p in output_path.rglob("*.safetensors")) / (1024**3)

print(f"\n{'=' * 80}")
print("MERGE COMPLETE — Model ready for DPO LoRA training")
print(f"{'=' * 80}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Merged model: {output_path}")
print(f"Model size: {model_size:.2f} GB")
print(f"\nThis model includes:")
print(f"  ✓ Base Llama 3.1 8B Instruct")
print(f"  ✓ DAPT LoRA (KP domain adaptation) — merged")
print(f"  ✓ SFT LoRA (instruction tuning) — merged")
print(f"\nNext step: Train DPO LoRA on this merged model")
print(f"  python scripts/15_train_dpo.py")
print(f"{'=' * 80}")
