"""
Serve the quantized model using vLLM for production inference
Optimized for RTX 3090 (24GB VRAM)
"""

import sys
import argparse
from pathlib import Path

print("="*80)
print("vLLM INFERENCE SERVER")
print("="*80)
print("Optimized for RTX 3090 (24GB VRAM)")
print("="*80)

# Parse arguments
parser = argparse.ArgumentParser(description="Serve KP Astrology model with vLLM")
parser.add_argument("--model-path", type=str, default="./models/quantized_8bit/", help="Path to quantized model")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
parser.add_argument("--max-model-len", type=int, default=2048, help="Maximum sequence length")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization (0-1)")
args = parser.parse_args()

# Verify model exists
model_path = Path(args.model_path)
if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    print("Please run quantization first: python scripts/06_quantize_unsloth.py")
    sys.exit(1)

print(f"\n📦 Model: {model_path}")
print(f"🌐 Server: {args.host}:{args.port}")
print(f"📏 Max length: {args.max_model_len}")
print(f"💾 GPU memory: {args.gpu_memory_utilization * 100}%")

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.api_server import run_server
    print("\n✓ vLLM imported successfully")
except ImportError:
    print("\n❌ vLLM not installed")
    print("\nInstall with:")
    print("  pip install vllm")
    sys.exit(1)

# Start vLLM server
print("\n" + "="*80)
print("STARTING vLLM SERVER")
print("="*80)
print("\nServer will be available at:")
print(f"  OpenAI-compatible API: http://{args.host}:{args.port}/v1")
print(f"  Health check: http://{args.host}:{args.port}/health")
print(f"  Docs: http://{args.host}:{args.port}/docs")
print("\nPress Ctrl+C to stop the server")
print("="*80 + "\n")

try:
    # Run vLLM server with OpenAI-compatible API
    import uvicorn
    from vllm.entrypoints.openai.api_server import app, init_app_state
    
    # Initialize vLLM engine
    init_app_state(
        engine_args={
            "model": str(model_path),
            "tokenizer": str(model_path),
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": "float16",
            "quantization": "bitsandbytes",  # For 8-bit model
            "trust_remote_code": True,
        }
    )
    
    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )
    
except KeyboardInterrupt:
    print("\n\n⚠️  Server stopped by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Server failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Check model path is correct")
    print("  2. Ensure sufficient GPU memory")
    print("  3. Verify vLLM installation: pip install vllm")
    print("  4. Check port is not already in use")
    sys.exit(1)
