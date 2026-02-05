"""
Merge DAPT and SFT LoRA adapters with base model
Creates a full model ready for quantization
"""

import sys
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
import os

load_dotenv()

print("="*80)
print("MERGING DAPT + SFT LORA ADAPTERS WITH BASE MODEL")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Verify HF token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN not found in .env file")
    sys.exit(1)

# Paths
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
dapt_lora_path = Path("./checkpoints/dapt_lora/final/")
sft_lora_path = Path("./checkpoints/sft_lora/final/")
output_path = Path("./models/merged/")

# Verify paths
if not dapt_lora_path.exists():
    print(f"❌ DAPT LoRA adapters not found: {dapt_lora_path}")
    print("Please run DAPT training first")
    sys.exit(1)

if not sft_lora_path.exists():
    print(f"❌ SFT LoRA adapters not found: {sft_lora_path}")
    print("Please run SFT training first")
    sys.exit(1)

# Load base model
print("\n1. Loading base Llama 3.1 8B Instruct model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"✓ Base model loaded: {model.num_parameters():,} parameters")
except Exception as e:
    print(f"❌ Failed to load base model: {e}")
    sys.exit(1)

# Load and merge DAPT LoRA
print("\n2. Loading DAPT LoRA adapters...")
try:
    model = PeftModel.from_pretrained(model, str(dapt_lora_path))
    print(f"✓ DAPT LoRA adapters loaded")
    
    print("   Merging DAPT LoRA into base model...")
    model = model.merge_and_unload()
    print(f"✓ DAPT LoRA merged")
except Exception as e:
    print(f"❌ Failed to load/merge DAPT LoRA: {e}")
    sys.exit(1)

# Load and merge SFT LoRA
print("\n3. Loading SFT LoRA adapters...")
try:
    model = PeftModel.from_pretrained(model, str(sft_lora_path))
    print(f"✓ SFT LoRA adapters loaded")
    
    print("   Merging SFT LoRA into model...")
    model = model.merge_and_unload()
    print(f"✓ SFT LoRA merged")
except Exception as e:
    print(f"❌ Failed to load/merge SFT LoRA: {e}")
    sys.exit(1)

# Save merged model
print("\n4. Saving fully merged model...")
output_path.mkdir(parents=True, exist_ok=True)

try:
    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="5GB"
    )
    tokenizer.save_pretrained(str(output_path))
    print(f"✓ Merged model saved to: {output_path}")
except Exception as e:
    print(f"❌ Failed to save model: {e}")
    sys.exit(1)

# Get model size
model_size = sum(p.stat().st_size for p in output_path.rglob('*.safetensors')) / (1024**3)

print(f"\n{'='*80}")
print("MERGE COMPLETE")
print(f"{'='*80}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Merged model location: {output_path}")
print(f"Model size: {model_size:.2f} GB")
print(f"\nModel includes:")
print(f"  ✓ Base Llama 3.1 8B Instruct")
print(f"  ✓ DAPT LoRA (KP domain adaptation)")
print(f"  ✓ SFT LoRA (instruction tuning)")
print(f"\nNext step: Quantize with Unsloth (8-bit)")
print(f"  python scripts/06_quantize_unsloth.py")
print(f"{'='*80}\n")
