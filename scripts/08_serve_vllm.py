"""
Serve the fine-tuned KP Astrology model using vLLM's OpenAI-compatible API.

Usage:
  python scripts/08_serve_vllm.py
  python scripts/08_serve_vllm.py --model-path ./models/merged/ --port 8000

The server exposes an OpenAI-compatible API at http://host:port/v1
"""

import sys
import os
import subprocess
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/merged/")
HOST = os.environ.get("VLLM_HOST", "0.0.0.0")
PORT = os.environ.get("VLLM_PORT", "8000")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "4096")
GPU_MEM_UTIL = os.environ.get("GPU_MEM_UTIL", "0.90")
DTYPE = os.environ.get("VLLM_DTYPE", "auto")
ENABLE_APC = os.environ.get("VLLM_ENABLE_APC", "true").lower() in ("1", "true", "yes")
KV_CACHE_DTYPE = os.environ.get("VLLM_KV_CACHE_DTYPE", "auto")

# Override from CLI args
import argparse
parser = argparse.ArgumentParser(description="Serve KP Astrology model with vLLM")
parser.add_argument("--model-path", type=str, default=MODEL_PATH)
parser.add_argument("--host", type=str, default=HOST)
parser.add_argument("--port", type=str, default=PORT)
parser.add_argument("--max-model-len", type=str, default=MAX_MODEL_LEN)
parser.add_argument("--gpu-memory-utilization", type=str, default=GPU_MEM_UTIL)
parser.add_argument("--dtype", type=str, default=DTYPE,
                    choices=["auto", "bfloat16", "float16", "float32"],
                    help="Model dtype: auto (let vLLM decide), bfloat16, float16, float32")
parser.add_argument("--enable-prefix-caching", action="store_true", default=ENABLE_APC,
                    help="Enable Automatic Prefix Caching (APC) for faster repeated prompts (default: on)")
parser.add_argument("--no-prefix-caching", action="store_true", default=False,
                    help="Disable Automatic Prefix Caching")
parser.add_argument("--kv-cache-dtype", type=str, default=KV_CACHE_DTYPE,
                    choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
                    help="KV cache dtype: auto (match model), fp8 (50%% memory saving)")
args = parser.parse_args()

# Resolve APC flag (--no-prefix-caching overrides)
if args.no_prefix_caching:
    args.enable_prefix_caching = False

# ── Validate model path ───────────────────────────────────────────────────────
model_path = Path(args.model_path)
if not model_path.exists() or not any(model_path.glob("*.safetensors")):
    # Try fallback paths
    for fallback in ["./models/quantized_8bit/", "./models/merged/"]:
        fp = Path(fallback)
        if fp.exists() and any(fp.glob("*.safetensors")):
            model_path = fp
            break
    else:
        print("No model found. Run training + merge first.")
        sys.exit(1)

print("="*80)
print("vLLM INFERENCE SERVER — KP Astrology Model")
print("="*80)
print(f"  Model:      {model_path}")
print(f"  Server:     http://{args.host}:{args.port}/v1")
print(f"  Max length: {args.max_model_len}")
print(f"  Dtype:      {args.dtype}")
print(f"  APC:        {'ON' if args.enable_prefix_caching else 'OFF'}")
print(f"  KV cache:   {args.kv_cache_dtype}")
print(f"  GPU memory: {float(args.gpu_memory_utilization)*100:.0f}%")
print("="*80)
print()
print("Endpoints:")
print(f"  POST http://{args.host}:{args.port}/v1/chat/completions")
print(f"  POST http://{args.host}:{args.port}/v1/completions")
print(f"  GET  http://{args.host}:{args.port}/health")
print()
print("Example curl:")
print(f"""  curl http://localhost:{args.port}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{{"model": "{model_path}", "messages": [{{"role": "user", "content": "What is the 7th house sub-lord significance in KP astrology?"}}]}}'""")
print()
print("Press Ctrl+C to stop.")
print("="*80)

# ── Launch vLLM via CLI (most reliable across versions) ───────────────────────
cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", str(model_path),
    "--host", args.host,
    "--port", args.port,
    "--max-model-len", args.max_model_len,
    "--gpu-memory-utilization", args.gpu_memory_utilization,
    "--dtype", args.dtype,
    "--kv-cache-dtype", args.kv_cache_dtype,
    "--trust-remote-code",
    "--served-model-name", "kp-astrology-llama",
]

if args.enable_prefix_caching:
    cmd.append("--enable-prefix-caching")

try:
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)
except KeyboardInterrupt:
    print("\nServer stopped.")
    sys.exit(0)
