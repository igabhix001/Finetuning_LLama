#!/bin/bash
# Setup environment on RunPod for training
# Run this first after cloning the repository
#
# IMPORTANT: Do NOT install flash-attn from source — it uses 30-40GB system RAM
# and will OOM-kill the container. Use prebuilt wheels or skip it.

set -e  # Exit on error

echo "========================================================================"
echo "SETTING UP TRAINING ENVIRONMENT ON RUNPOD"
echo "========================================================================"

# Check if running on RunPod
if [ ! -d "/workspace" ]; then
    echo "⚠️  Warning: Not running on RunPod (/workspace not found)"
    echo "This script is designed for RunPod environment"
fi

# Step 1: Keep existing PyTorch (RunPod images ship with a working version)
echo ""
echo "1. Checking existing PyTorch..."
python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda} — keeping as-is')" 2>/dev/null || {
    echo "   PyTorch not found, installing..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

# Step 2: Install training dependencies (lightweight, no compilation)
echo ""
echo "2. Installing training dependencies..."
pip install --no-cache-dir \
    "transformers>=4.36.0" \
    "datasets>=2.16.0" \
    "accelerate>=0.25.0" \
    "peft>=0.7.1" \
    "bitsandbytes>=0.41.3" \
    "scipy>=1.11.0" \
    "sentencepiece>=0.1.99" \
    "protobuf>=4.25.0" \
    "python-dotenv>=1.0.0" \
    "pyyaml>=6.0" \
    "tqdm>=4.66.0" \
    "numpy>=1.24.0" \
    "tensorboard>=2.15.0" \
    "huggingface-hub>=0.20.0"

# Step 3: Check for flash-attn (do NOT install — source compile uses 30-40GB RAM and kills the pod)
echo ""
echo "3. Checking flash-attention..."
python -c "import flash_attn; print(f'✅ flash-attn {flash_attn.__version__} already installed')" 2>/dev/null || \
    echo "ℹ️  flash-attn not installed — training will use PyTorch SDPA attention (built-in, efficient, no extra install needed)"

# Step 4: Verify installations
echo ""
echo "4. Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')"

# Step 5: Check GPU
echo ""
echo "5. Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Step 6: Login to HuggingFace
echo ""
echo "6. Setting up HuggingFace authentication..."
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    if [ -n "$HF_TOKEN" ]; then
        huggingface-cli login --token "$HF_TOKEN"
        echo "✅ HuggingFace authentication successful"
    else
        echo "⚠️  HF_TOKEN not found in .env file"
        echo "Please add your HuggingFace token to .env"
    fi
else
    echo "⚠️  .env file not found"
    echo "Please create .env from .env.example and add your tokens"
fi

# Step 7: Create necessary directories
echo ""
echo "7. Creating directory structure..."
mkdir -p checkpoints/dapt_lora checkpoints/sft_lora
mkdir -p logs/dapt logs/sft
mkdir -p models/base models/merged models/quantized

echo ""
echo "========================================================================"
echo "ENVIRONMENT SETUP COMPLETE"
echo "========================================================================"
echo ""
echo "Memory usage:"
free -h | head -2
echo ""
echo "Next steps:"
echo "  1. Verify .env file has correct HF_TOKEN"
echo "  2. Run: python scripts/03_train_dapt.py (start DAPT training)"
echo "  3. After DAPT: python scripts/04_train_sft.py"
echo ""
echo "========================================================================"
