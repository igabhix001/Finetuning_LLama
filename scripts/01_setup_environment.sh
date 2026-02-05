#!/bin/bash
# Setup environment on RunPod for training
# Run this first after cloning the repository

set -e  # Exit on error

echo "========================================================================"
echo "SETTING UP TRAINING ENVIRONMENT ON RUNPOD"
echo "========================================================================"

# Check if running on RunPod
if [ ! -d "/workspace" ]; then
    echo "⚠️  Warning: Not running on RunPod (/workspace not found)"
    echo "This script is designed for RunPod environment"
fi

# Update pip
echo ""
echo "1. Updating pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "2. Installing PyTorch with CUDA 12.1..."
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo ""
echo "3. Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify installations
echo ""
echo "4. Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# Check GPU
echo ""
echo "5. Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Login to HuggingFace
echo ""
echo "6. Setting up HuggingFace authentication..."
if [ -f ".env" ]; then
    source .env
    if [ -n "$HF_TOKEN" ]; then
        echo "$HF_TOKEN" | huggingface-cli login --token
        echo "✅ HuggingFace authentication successful"
    else
        echo "⚠️  HF_TOKEN not found in .env file"
        echo "Please add your HuggingFace token to .env"
    fi
else
    echo "⚠️  .env file not found"
    echo "Please create .env from .env.example and add your tokens"
fi

# Create necessary directories
echo ""
echo "7. Creating directory structure..."
mkdir -p checkpoints/dapt checkpoints/sft
mkdir -p logs/dapt logs/sft
mkdir -p models/base models/dapt models/sft models/quantized

echo ""
echo "========================================================================"
echo "ENVIRONMENT SETUP COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Verify .env file has PINECONE_API_KEY and HF_TOKEN"
echo "  2. Run: python scripts/02_upload_pinecone.py (optional, for RAG)"
echo "  3. Run: python scripts/03_train_dapt.py (start DAPT training)"
echo ""
echo "========================================================================"
