"""
Quantize merged model to 8-bit using Unsloth for RTX 3090 deployment
"""

import sys
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*80)
print("QUANTIZING MODEL WITH UNSLOTH (8-BIT)")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Target: RTX 3090 (24GB VRAM)")
print("="*80)

# Paths
model_path = Path("./models/merged/")
output_path = Path("./models/quantized_8bit/")

# Verify model exists
if not model_path.exists():
    print(f"❌ Merged model not found: {model_path}")
    print("Please run merge script first: python scripts/05_merge_adapters.py")
    sys.exit(1)

# Load tokenizer
print("\n1. Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    print(f"✓ Tokenizer loaded")
except Exception as e:
    print(f"❌ Failed to load tokenizer: {e}")
    sys.exit(1)

# Load and quantize model with Unsloth
print("\n2. Loading model with Unsloth 8-bit quantization...")
print("   This will take 15-30 minutes...")

try:
    # Try to import unsloth
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
        print("   Using Unsloth for quantization")
    except ImportError:
        print("   ⚠️  Unsloth not found, using bitsandbytes 8-bit quantization")
        use_unsloth = False
    
    if use_unsloth:
        # Unsloth quantization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=2048,
            dtype=None,  # Auto-detect
            load_in_4bit=False,  # Use 8-bit
            load_in_8bit=True,
        )
        print(f"✓ Model quantized with Unsloth (8-bit)")
    else:
        # Fallback to bitsandbytes
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print(f"✓ Model quantized with bitsandbytes (8-bit)")
    
except Exception as e:
    print(f"❌ Quantization failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Install Unsloth: pip install unsloth")
    print("  2. Ensure sufficient GPU memory")
    print("  3. Check CUDA compatibility")
    sys.exit(1)

# Save quantized model
print("\n3. Saving quantized model...")
output_path.mkdir(parents=True, exist_ok=True)

try:
    if use_unsloth:
        # Unsloth save
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
    else:
        # Standard save
        model.save_pretrained(
            str(output_path),
            safe_serialization=True
        )
        tokenizer.save_pretrained(str(output_path))
    
    print(f"✓ Quantized model saved to: {output_path}")
except Exception as e:
    print(f"❌ Failed to save model: {e}")
    sys.exit(1)

# Get model size
try:
    model_size = sum(p.stat().st_size for p in output_path.rglob('*')) / (1024**3)
except:
    model_size = 0

print(f"\n{'='*80}")
print("QUANTIZATION COMPLETE")
print(f"{'='*80}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Quantized model location: {output_path}")
print(f"Model size: {model_size:.2f} GB (should be ~8-10GB for 8-bit)")
print(f"\nModel is ready for RTX 3090 deployment!")
print(f"\nNext step: Setup vLLM serving")
print(f"  python scripts/08_serve_vllm.py")
print(f"{'='*80}\n")
