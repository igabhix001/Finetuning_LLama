# Quick Start Guide - RunPod Training

**Goal:** Get training running on RunPod in under 30 minutes

---

## Prerequisites

- ✅ GitHub account
- ✅ RunPod account with payment method
- ✅ HuggingFace account with Llama 3.1 access
- ✅ Pinecone account (free tier OK)

---

## Step 1: Push to GitHub (5 minutes)

On your local machine:

```powershell
cd d:\Dataset_preprossecing_pipeline\Finetuning_LLama

git init
git add .
git commit -m "Initial commit: KP Astrology training pipeline"
git remote add origin https://github.com/YOUR_USERNAME/Finetuning_LLama.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## Step 2: Launch RunPod (5 minutes)

1. Go to https://www.runpod.io/
2. Click **"Deploy"** → **"GPU Instances"**
3. Select **RTX 6000 Ada (48GB)** - ~$0.89/hr
4. Template: **"RunPod Pytorch 2.1"** or **"Pytorch 2.4"**
5. Container Disk: **50GB**
6. Volume Disk: **100GB**
7. Click **"Deploy On-Demand"**
8. Wait 1-2 minutes for pod to start

---

## Step 3: Clone & Setup (10 minutes)

Click **"Connect"** → **"Start Web Terminal"** on your pod.

```bash
# Clone repository
cd /workspace
git clone https://github.com/YOUR_USERNAME/Finetuning_LLama.git
cd Finetuning_LLama

# Setup environment
cp .env.example .env
nano .env
```

Add your keys:
```
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=kp-astrology-kb
HF_TOKEN=your-huggingface-token-here
```

Save (Ctrl+X, Y, Enter).

```bash
# Install dependencies
bash scripts/01_setup_environment.sh
```

---

## Step 4: Verify Dataset (2 minutes)

```bash
# Check dataset files
ls -lh data/

# Should see:
# - dapt_corpus/
# - sft_train/
# - sft_validation/
# - pinecone_upsert.jsonl
```

If files are missing, they're already in your local folder. The dataset was copied when you created the repo.

---

## Step 5: Start Training (5 minutes to start)

### LoRA DAPT Training (2-4 hours)

```bash
python scripts/03_train_dapt.py
```

**Monitor progress:**
- Watch terminal output
- Or open new terminal: `tensorboard --logdir=logs/dapt/`

**What to expect:**
- Method: LoRA adapters (not full fine-tuning)
- Trainable params: ~2-3% of total
- Initial loss: ~3.5-4.0
- Final loss: ~2.5-3.0
- Time: 2-4 hours on RTX 6000 Ada

### LoRA SFT Training (6-10 hours)

After DAPT completes:

```bash
python scripts/04_train_sft.py
```

**What to expect:**
- Method: LoRA on top of merged DAPT LoRA
- Trainable params: ~2-3% of total
- Initial loss: ~2.5-3.0
- Final loss: ~1.2-1.5
- Time: 6-10 hours on RTX 6000 Ada

---

## Step 6: Post-Training (1-2 hours)

```bash
# Merge DAPT + SFT LoRA adapters (~30 mins)
python scripts/05_merge_adapters.py

# Quantize with Unsloth 8-bit (~30 mins)
python scripts/06_quantize_unsloth.py

# Test model (~5 mins)
python scripts/07_test_inference.py

# Serve with vLLM (production)
python scripts/08_serve_vllm.py --host 0.0.0.0 --port 8000
```

---

## Step 7: Download Model

```bash
# Compress quantized model
cd /workspace/Finetuning_LLama
tar -czf kp_model_quantized.tar.gz models/quantized/

# Download via RunPod UI or SCP
```

---

## Total Time & Cost

| Phase | Duration | Cost (RTX 6000 Ada @ $0.89/hr) |
|-------|----------|-------------------------------|
| Setup | 30 mins | $0.45 |
| LoRA DAPT | 2-4 hrs | $1.78-$3.56 |
| LoRA SFT | 6-10 hrs | $5.34-$8.90 |
| Merge + Quantize | 1 hr | $0.89 |
| **Total** | **10-16 hrs** | **$8-14** |

---

## Monitoring Training

### Option 1: Terminal Output
Watch the terminal for loss values and progress bars.

### Option 2: TensorBoard
```bash
# In a new terminal
tensorboard --logdir=logs/

# Access via RunPod port forwarding
```

### Option 3: Check Logs
```bash
# DAPT logs
tail -f logs/dapt/train.log

# SFT logs
tail -f logs/sft/train.log
```

---

## Troubleshooting

### Out of Memory
```bash
# Edit config files to reduce batch size
nano configs/dapt_config.yaml  # Change per_device_train_batch_size to 2
nano configs/sft_config.yaml   # Change per_device_train_batch_size to 2
```

### Training Interrupted
Training automatically saves checkpoints every 100/500 steps. Resume with:
```bash
# Training will auto-resume from last checkpoint
python scripts/03_train_dapt.py  # or 04_train_sft.py
```

### Dataset Missing
```bash
# Verify data folder
ls -lh data/

# If empty, dataset files are in your local repo
# Upload separately using SCP or Google Drive
```

### HuggingFace Access Denied
1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Request Access"
3. Wait for approval (usually instant)
4. Verify HF_TOKEN in .env is correct

---

## What's Next?

After training completes:

1. **Test Model:** Run `scripts/07_test_inference.py`
2. **Serve with vLLM:** Run `scripts/08_serve_vllm.py`
3. **RAG:** Upload OpenAI embeddings to Pinecone with `scripts/02_upload_pinecone.py`
4. **Download:** Get quantized model from RunPod (if deploying elsewhere)
5. **Deploy:** Model ready for RTX 3090 with vLLM
6. **Integrate:** Connect to OpenAI-compatible API at `http://localhost:8000/v1`

See `Finetunning_runpod.md` Section 4 for deployment instructions.

---

**You're ready to train!** 🚀

**Questions?** Check:
- `README.md` - Full documentation
- `GITHUB_SETUP.md` - GitHub instructions
- `TRAINING_GUIDE.md` - Detailed training guide
- `RUNPOD_SETUP.md` - Complete RunPod setup
