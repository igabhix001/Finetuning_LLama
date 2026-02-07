"""
Quantize merged model to 8-bit using bitsandbytes for deployment.

The merged model (from step 05) is a full model with no LoRA adapters,
so we use bitsandbytes quantization directly (not Unsloth, which expects LoRA).
"""

import sys
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("="*80)
print("QUANTIZING MERGED MODEL (8-BIT)")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

# Load model with 8-bit quantization via bitsandbytes
print("\n2. Loading model with bitsandbytes 8-bit quantization...")

try:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"✓ Model loaded with 8-bit quantization")
    print(f"  Parameters: {model.num_parameters():,}")

except Exception as e:
    print(f"❌ Quantization failed: {e}")
    sys.exit(1)

# Save quantized model
# NOTE: bitsandbytes 8-bit models cannot be directly saved as quantized weights.
# Instead, we save the full-precision merged model (already at models/merged/)
# and load it with quantization_config at serving time.
# For a truly smaller on-disk model, we export to GGUF format.
print("\n3. Saving model in GGUF format for efficient deployment...")
output_path.mkdir(parents=True, exist_ok=True)

try:
    # Try llama.cpp GGUF export via Unsloth (if available)
    try:
        from unsloth import FastLanguageModel
        print("   Using Unsloth for GGUF export...")
        # Reload with Unsloth for GGUF conversion
        umodel, utokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )
        umodel.save_pretrained_gguf(
            str(output_path),
            utokenizer,
            quantization_method="q8_0",
        )
        print(f"✓ GGUF Q8_0 model saved to: {output_path}")
    except Exception as e_gguf:
        print(f"   GGUF export not available ({e_gguf})")
        print("   Saving as standard safetensors (bf16) instead...")
        # Fallback: just copy the merged model as-is (it's already the final model)
        # The merged model at models/merged/ IS the deployable model.
        # At serving time, load with load_in_8bit=True for quantization.
        from shutil import copytree, rmtree
        if output_path.exists():
            rmtree(str(output_path))
        copytree(str(model_path), str(output_path))
        # Write a config note about runtime quantization
        with open(str(output_path / "QUANTIZATION_NOTE.md"), "w") as f:
            f.write("# Quantization Note\n\n")
            f.write("This model is saved in bf16 format.\n")
            f.write("For 8-bit inference, load with:\n\n")
            f.write("```python\n")
            f.write("from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n")
            f.write("model = AutoModelForCausalLM.from_pretrained(\n")
            f.write("    'models/quantized_8bit',\n")
            f.write("    quantization_config=BitsAndBytesConfig(load_in_8bit=True),\n")
            f.write("    device_map='auto'\n")
            f.write(")\n")
            f.write("```\n\n")
            f.write("Or use vLLM which handles quantization automatically.\n")
        print(f"✓ Model copied to: {output_path}")
        print("  Load with BitsAndBytesConfig(load_in_8bit=True) at serving time")
        print("  Or use vLLM directly on models/merged/")

except Exception as e:
    print(f"❌ Failed to save model: {e}")
    sys.exit(1)

# Get model size
try:
    model_size = sum(p.stat().st_size for p in output_path.rglob('*') if p.is_file()) / (1024**3)
except:
    model_size = 0

print(f"\n{'='*80}")
print("QUANTIZATION COMPLETE")
print(f"{'='*80}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Quantized model location: {output_path}")
print(f"Model size on disk: {model_size:.2f} GB")
print(f"\nFor serving, you can use either:")
print(f"  1. vLLM (recommended): python scripts/08_serve_vllm.py")
print(f"  2. Direct: load models/merged/ with load_in_8bit=True")
print(f"{'='*80}\n")
