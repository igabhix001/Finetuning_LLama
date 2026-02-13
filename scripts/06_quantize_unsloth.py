"""
Quantize/export a merged model for deployment.

Modes:
  safetensors  — copy bf16/fp16 safetensors for vLLM serving (DEFAULT, recommended)
  q4_k_m       — Unsloth GGUF 4-bit (for llama.cpp, NOT vLLM)
  q8_0         — Unsloth GGUF 8-bit (for llama.cpp, NOT vLLM)
  both         — export both GGUF variants

NOTE: vLLM serves safetensors natively. GGUF is for llama.cpp only.
      Use --method safetensors (default) when deploying with vLLM.

Usage:
  python scripts/06_quantize_unsloth.py                          # safetensors for vLLM
  python scripts/06_quantize_unsloth.py --method safetensors     # explicit vLLM mode
  python scripts/06_quantize_unsloth.py --method q4_k_m          # GGUF for llama.cpp
  python scripts/06_quantize_unsloth.py --method q8_0            # GGUF for llama.cpp
  python scripts/06_quantize_unsloth.py --method both            # both GGUF variants
  python scripts/06_quantize_unsloth.py --model ./models/final_dpo/
"""

import sys
import os
import argparse
import torch
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser(description="Quantize model to GGUF via Unsloth")
parser.add_argument("--model", type=str, default="./models/final_dpo/",
                    help="Path to the merged model to quantize")
parser.add_argument("--output", type=str, default="./models/quantized/",
                    help="Output directory for quantized model")
parser.add_argument("--method", type=str, default="safetensors",
                    choices=["safetensors", "q4_k_m", "q8_0", "both"],
                    help="Export method: safetensors (vLLM, default), q4_k_m/q8_0/both (GGUF for llama.cpp)")
parser.add_argument("--max-seq-length", type=int, default=2048,
                    help="Max sequence length for the model")
args = parser.parse_args()

model_path = Path(args.model)
output_path = Path(args.output)

# ── Auto-detect model path ──────────────────────────────────────────────────
if not model_path.exists() or not any(model_path.glob("*.safetensors")):
    for fallback in ["./models/final_dpo/", "./models/merged_sft/", "./models/merged/"]:
        fp = Path(fallback)
        if fp.exists() and any(fp.glob("*.safetensors")):
            model_path = fp
            print(f"ℹ️  Using fallback model path: {model_path}")
            break
    else:
        print(f"❌ No model found at {args.model} or any fallback path.")
        print("Run DPO merge first: python scripts/16_merge_dpo_lora.py")
        sys.exit(1)

print("=" * 80)
print("QUANTIZE MODEL — Unsloth GGUF Export")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model:      {model_path}")
print(f"Output:     {output_path}")
print(f"Method:     {args.method}")
if torch.cuda.is_available():
    print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 80)

output_path.parent.mkdir(parents=True, exist_ok=True)

# ── Safetensors export for vLLM (default, recommended) ──────────────────────
if args.method == "safetensors":
    from shutil import copytree, rmtree
    print("\n1. Exporting safetensors for vLLM serving...")
    if output_path.exists():
        rmtree(str(output_path))
    copytree(str(model_path), str(output_path))
    print(f"   ✓ Model copied to: {output_path}")
    print(f"   Serve with: python scripts/08_serve_vllm.py --model-path {output_path}")

    model_size = sum(p.stat().st_size for p in output_path.rglob("*") if p.is_file()) / (1024**3)
    print(f"\n{'=' * 80}")
    print("EXPORT COMPLETE (safetensors for vLLM)")
    print(f"{'=' * 80}")
    print(f"Output:    {output_path}")
    print(f"Size:      {model_size:.2f} GB")
    print(f"{'=' * 80}")
    sys.exit(0)

# ── GGUF export (for llama.cpp only, NOT vLLM) ─────────────────────────────

# Clean stale llama.cpp cache (causes QWEN35 enum errors)
stale_llama_cpp = Path("llama.cpp")
if stale_llama_cpp.exists():
    import shutil as _sh
    print(f"ℹ️  Removing stale {stale_llama_cpp}/ to avoid converter conflicts...")
    _sh.rmtree(str(stale_llama_cpp), ignore_errors=True)

print("\n⚠️  GGUF export is for llama.cpp serving, NOT vLLM.")
print("   For vLLM, use: --method safetensors (default)\n")

try:
    from unsloth import FastLanguageModel
    print("\n1. Loading model with Unsloth...")
    umodel, utokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    print(f"   ✓ Model loaded")

    methods = []
    if args.method == "both":
        methods = ["q4_k_m", "q8_0"]
    else:
        methods = [args.method]

    # Unsloth save_pretrained_gguf needs a dir with config.json (the model dir).
    # We save into model_path first, then move GGUF files to output_path.
    import shutil, glob

    for method in methods:
        print(f"\n2. Exporting GGUF ({method.upper()})...")
        umodel.save_pretrained_gguf(
            str(model_path),       # must contain config.json
            utokenizer,
            quantization_method=method,
        )
        # Move generated .gguf files from model_path to output_path
        for gguf_file in glob.glob(str(model_path / "*.gguf")):
            dest = output_path / Path(gguf_file).name
            shutil.move(gguf_file, str(dest))
            print(f"   ✓ Moved {Path(gguf_file).name} → {dest}")
        print(f"   ✓ GGUF {method.upper()} saved to: {output_path}")

except ImportError:
    print("\n⚠️  Unsloth not installed. Install with: pip install unsloth")
    print("   Falling back to bf16 safetensors copy for vLLM serving...")
    from shutil import copytree, rmtree
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if output_path.exists() and any(output_path.iterdir()):
        rmtree(str(output_path))
    copytree(str(model_path), str(output_path))
    print(f"   ✓ bf16 model copied to: {output_path}")
    print("   Serve directly with vLLM (no quantization needed)")

except Exception as e:
    print(f"\n❌ Quantization failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nFallback: serve the bf16 model directly with vLLM:")
    print(f"  python scripts/08_serve_vllm.py --model-path {model_path}")
    sys.exit(1)

# ── Summary ──────────────────────────────────────────────────────────────────
try:
    model_size = sum(p.stat().st_size for p in output_path.rglob("*") if p.is_file()) / (1024**3)
except Exception:
    model_size = 0

print(f"\n{'=' * 80}")
print("QUANTIZATION COMPLETE")
print(f"{'=' * 80}")
print(f"End time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output:    {output_path}")
print(f"Size:      {model_size:.2f} GB")
print(f"\nNext steps:")
print(f"  Serve with vLLM:  python scripts/08_serve_vllm.py --model-path {output_path}")
print(f"  Or serve bf16:    python scripts/08_serve_vllm.py --model-path {model_path}")
print(f"{'=' * 80}")
