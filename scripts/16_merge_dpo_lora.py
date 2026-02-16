"""
Merge DPO LoRA into Merged DAPT+SFT Model — Final Production Model
====================================================================
Stage 3 final step: Merge the DPO LoRA adapter into the already-merged
DAPT+SFT base model to produce the final deployment-ready model.

Usage:
  python scripts/16_merge_dpo_lora.py
  python scripts/16_merge_dpo_lora.py --output ./models/final_dpo/
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

# Suppress known false-positive tokenizer regex warning (Llama 3.1 != Mistral)
import warnings
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")

parser = argparse.ArgumentParser(description="Merge DPO LoRA into final model")
parser.add_argument("--base-model", type=str, default="./models/merged_sft/",
                    help="Merged DAPT+SFT model path")
parser.add_argument("--dpo-lora", type=str, default="./checkpoints/dpo_lora/final/",
                    help="DPO LoRA adapter path")
parser.add_argument("--output", type=str, default="./models/final_dpo/",
                    help="Output path for final merged model")
parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
args = parser.parse_args()

print("=" * 80)
print("MERGE DPO LORA → FINAL PRODUCTION MODEL")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Base (merged SFT): {args.base_model}")
print(f"DPO LoRA: {args.dpo_lora}")
print(f"Output: {args.output}")
print("=" * 80)

hf_token = os.getenv("HF_TOKEN")
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
model_dtype = dtype_map[args.dtype]

# 1. Load merged DAPT+SFT model
base_path = Path(args.base_model)
if not base_path.exists():
    print(f"❌ Merged SFT model not found: {base_path}")
    print("Run 05b_merge_sft_lora.py first")
    sys.exit(1)

print("\n1. Loading merged DAPT+SFT model...")
tokenizer = AutoTokenizer.from_pretrained(str(base_path))
model = AutoModelForCausalLM.from_pretrained(
    str(base_path),
    torch_dtype=model_dtype,
    device_map="auto",
    trust_remote_code=True,
)
print(f"   ✓ Model loaded: {model.num_parameters():,} parameters")

# 1b. Load DPO tokenizer to check vocab size
dpo_path = Path(args.dpo_lora)
if not dpo_path.exists():
    print(f"❌ DPO LoRA not found: {dpo_path}")
    print("Run 15_train_dpo.py first")
    sys.exit(1)

print("\n1b. Checking tokenizer compatibility...")
dpo_tokenizer = AutoTokenizer.from_pretrained(str(dpo_path))
if len(dpo_tokenizer) != model.config.vocab_size:
    print(f"   Resizing model embeddings: {model.config.vocab_size} → {len(dpo_tokenizer)}")
    model.resize_token_embeddings(len(dpo_tokenizer))
    print(f"   ✓ Embeddings resized to match DPO checkpoint")
else:
    print(f"   ✓ Vocab sizes match: {len(dpo_tokenizer)}")

# Update tokenizer to DPO version (has PAD token if added)
tokenizer = dpo_tokenizer

# 2. Merge DPO LoRA
print("\n2. Merging DPO LoRA...")
model = PeftModel.from_pretrained(model, str(dpo_path))
model = model.merge_and_unload()
print("   ✓ DPO LoRA merged")

# 3. Save final model
print(f"\n3. Saving final production model to {args.output}...")
output_path = Path(args.output)
output_path.mkdir(parents=True, exist_ok=True)

model.save_pretrained(str(output_path), safe_serialization=True, max_shard_size="5GB")
tokenizer.save_pretrained(str(output_path))

model_size = sum(p.stat().st_size for p in output_path.rglob("*.safetensors")) / (1024**3)

print(f"\n{'=' * 80}")
print("FINAL MODEL MERGE COMPLETE")
print(f"{'=' * 80}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Final model: {output_path}")
print(f"Model size: {model_size:.2f} GB")
print(f"\nThis model includes ALL 3 training stages:")
print(f"  ✓ Stage 1: DAPT LoRA (KP domain adaptation) — merged")
print(f"  ✓ Stage 2: SFT LoRA (instruction tuning) — merged")
print(f"  ✓ Stage 3: DPO LoRA (preference optimization) — merged")
print(f"\nNext step: Quantize for deployment")
print(f"  python scripts/06_quantize_unsloth.py --model {output_path}")
print(f"{'=' * 80}")
