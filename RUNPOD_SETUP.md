# RunPod Setup Guide - Complete Beginner's Guide

**Target Audience:** First-time RunPod users  
**Goal:** Get your training environment ready step-by-step

---

## 🎯 What You'll Do

1. Create RunPod account
2. Add payment method
3. Select and launch GPU pod
4. Upload dataset
5. Setup training environment
6. Start training

**Estimated Time:** 30-45 minutes (excluding training)

---

## Step 1: Create RunPod Account

### 1.1 Sign Up
1. Go to https://www.runpod.io/
2. Click **"Sign Up"** (top right)
3. Use Google/GitHub or email
4. Verify your email

### 1.2 Add Payment Method
1. Go to **Billing** (left sidebar)
2. Click **"Add Payment Method"**
3. Add credit card
4. **Recommended:** Add $50-100 for training
   - DAPT: ~$5-10
   - SFT: ~$15-25
   - Total: ~$20-35

---

## Step 2: Select GPU Pod

### 2.1 Choose GPU Type
1. Click **"GPU Instances"** (left sidebar)
2. Click **"Deploy"** (top right)
3. Filter by GPU:
   - **For Training:** RTX 6000 Ada (48GB) ✅ RECOMMENDED
   - Alternative: A100 (80GB) - faster but more expensive

### 2.2 GPU Comparison

| GPU | VRAM | Price/hr | Best For |
|-----|------|----------|----------|
| RTX 6000 Ada | 48GB | ~$0.89 | ✅ Our training (perfect fit) |
| A100 40GB | 40GB | ~$1.89 | Too small for our needs |
| A100 80GB | 80GB | ~$2.49 | Overkill (faster but expensive) |
| RTX 3090 | 24GB | ~$0.34 | ❌ Too small for training |

**Choose RTX 6000 Ada** - best price/performance for our use case.

### 2.3 Select Template
1. In "Select a Template" dropdown:
   - Choose **"RunPod Pytorch 2.1"** or **"RunPod Pytorch 2.4"**
   - This includes Python, PyTorch, CUDA pre-installed
2. Alternative: **"RunPod Fast Stable Diffusion"** (has transformers)

### 2.4 Configure Pod
1. **Container Disk:** 50GB (minimum)
2. **Volume Disk:** 100GB (for datasets and checkpoints)
3. **Expose HTTP Ports:** 8888 (for Jupyter), 7860 (for Gradio testing)
4. **SSH:** Enable (for file transfer)

### 2.5 Launch Pod
1. Review pricing (should be ~$0.89/hr for RTX 6000 Ada)
2. Click **"Deploy On-Demand"** (not Spot - more reliable)
3. Wait 1-2 minutes for pod to start
4. Status will change to **"Running"**

---

## Step 3: Connect to Your Pod

### 3.1 Access Methods

You have 3 ways to access your pod:

#### Option A: Web Terminal (Easiest for beginners)
1. Click **"Connect"** button on your pod
2. Select **"Start Web Terminal"**
3. A terminal opens in your browser ✅ RECOMMENDED

#### Option B: Jupyter Notebook
1. Click **"Connect"** → **"Connect to Jupyter Lab"**
2. Opens Jupyter in browser
3. Good for interactive work

#### Option C: SSH (Advanced)
1. Click **"Connect"** → **"Connect with SSH"**
2. Copy SSH command
3. Run in your local terminal (requires SSH client)

**For this guide, use Option A (Web Terminal).**

---

## Step 4: Upload Dataset to RunPod

### 4.1 Prepare Dataset on Your Local Machine

Open PowerShell on your local machine:

```powershell
cd d:\Dataset_preprossecing_pipeline

# Create compressed archive
tar -czf kp_dataset.tar.gz `
  data/arrow/ `
  data/final/kb_chunks.jsonl `
  data/final/pinecone_upsert.jsonl `
  data/final/rule_embeddings.npy `
  data/final/rule_embeddings_metadata.json `
  Finetuning_LLama/
```

This creates `kp_dataset.tar.gz` (~50-100MB).

### 4.2 Upload to RunPod

#### Method 1: RunPod Web UI (Easiest)
1. In RunPod web terminal, run:
   ```bash
   cd /workspace
   ```
2. In your local PowerShell:
   ```powershell
   # Get your pod's SSH connection string from RunPod UI
   # It looks like: ssh root@<pod-id>.runpod.io -p <port> -i ~/.ssh/id_ed25519
   
   # Upload using SCP
   scp -P <port> kp_dataset.tar.gz root@<pod-id>.runpod.io:/workspace/
   ```

#### Method 2: Google Drive (Alternative)
1. Upload `kp_dataset.tar.gz` to Google Drive
2. Get shareable link
3. In RunPod terminal:
   ```bash
   cd /workspace
   # Install gdown
   pip install gdown
   
   # Download from Google Drive
   gdown <your-google-drive-file-id>
   ```

#### Method 3: Direct Upload (Small files only)
1. In Jupyter Lab (Option B above)
2. Use upload button
3. ⚠️ Only for files <100MB

### 4.3 Extract Dataset

In RunPod terminal:

```bash
cd /workspace
tar -xzf kp_dataset.tar.gz

# Verify extraction
ls -lh data/arrow/
ls -lh Finetuning_LLama/
```

You should see:
- `data/arrow/dapt_corpus/`
- `data/arrow/sft_train/`
- `data/arrow/sft_validation/`
- `Finetuning_LLama/` folder

---

## Step 5: Setup Training Environment

### 5.1 Install Dependencies

In RunPod terminal:

```bash
cd /workspace/Finetuning_LLama

# Update pip
pip install --upgrade pip

# Install required packages
pip install transformers==4.36.0
pip install datasets==2.16.0
pip install accelerate==0.25.0
pip install peft==0.7.1
pip install bitsandbytes==0.41.3
pip install scipy
pip install sentencepiece
pip install protobuf

# For Pinecone
pip install pinecone-client

# For quantization (later)
pip install auto-gptq
pip install optimum
```

This takes ~5-10 minutes.

### 5.2 Download Llama-3.1-8B-Instruct Base Model

```bash
# Login to HuggingFace (you need a token)
huggingface-cli login

# Enter your HF token when prompted
# Get token from: https://huggingface.co/settings/tokens

# Download model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'meta-llama/Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.save_pretrained('/workspace/Finetuning_LLama/models/base/')
model.save_pretrained('/workspace/Finetuning_LLama/models/base/')
print('Model downloaded successfully!')
"
```

⚠️ **Important:** You need HuggingFace access to Llama 3.1:
1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Request Access"
3. Wait for approval (usually instant)

### 5.3 Verify Setup

```bash
# Check GPU
nvidia-smi

# Should show RTX 6000 Ada with 48GB VRAM

# Check Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check datasets
python -c "from datasets import load_from_disk; ds = load_from_disk('/workspace/data/arrow/sft_train'); print(f'SFT examples: {len(ds)}')"
```

Expected output:
- PyTorch: 2.1.x or 2.4.x
- CUDA: True
- SFT examples: 19303

---

## Step 6: Setup Pinecone (for RAG)

### 6.1 Create Pinecone Account
1. Go to https://www.pinecone.io/
2. Sign up (free tier available)
3. Create new index:
   - **Name:** `kp-astrology-kb`
   - **Dimensions:** 3072 (for OpenAI text-embedding-3-large)
   - **Metric:** cosine
   - **Spec:** Serverless (AWS us-east-1)

### 6.2 Get API Key
1. Go to Pinecone dashboard
2. Click **"API Keys"**
3. Copy your API key

### 6.3 Upload Embeddings

Create `.env` file in `/workspace/Finetuning_LLama/`:

```bash
cat > /workspace/Finetuning_LLama/.env << 'EOF'
PINECONE_API_KEY=your-api-key-here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=kp-astrology
EOF
```

Then run upload script (we'll create this next).

---

## Step 7: Verify Everything is Ready

### Checklist

Run this verification script:

```bash
cd /workspace/Finetuning_LLama

cat > verify_setup.sh << 'EOF'
#!/bin/bash
echo "=== RunPod Setup Verification ==="
echo ""

# Check GPU
echo "1. GPU Check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check datasets
echo "2. Dataset Check:"
echo "  DAPT corpus: $(ls /workspace/data/arrow/dapt_corpus/ 2>/dev/null && echo '✅' || echo '❌')"
echo "  SFT train: $(ls /workspace/data/arrow/sft_train/ 2>/dev/null && echo '✅' || echo '❌')"
echo "  SFT validation: $(ls /workspace/data/arrow/sft_validation/ 2>/dev/null && echo '✅' || echo '❌')"
echo ""

# Check model
echo "3. Base Model Check:"
echo "  Llama 3.1 8B: $(ls /workspace/Finetuning_LLama/models/base/ 2>/dev/null && echo '✅' || echo '❌')"
echo ""

# Check Python packages
echo "4. Python Packages:"
python -c "import torch, transformers, datasets, peft, accelerate; print('  All packages: ✅')" 2>/dev/null || echo "  Missing packages: ❌"
echo ""

# Check Pinecone
echo "5. Pinecone Setup:"
[ -f .env ] && echo "  .env file: ✅" || echo "  .env file: ❌"
echo ""

echo "=== Setup Complete! ==="
echo "Ready to start training!"
EOF

chmod +x verify_setup.sh
./verify_setup.sh
```

All items should show ✅.

---

## Step 8: Start Training

### 8.1 DAPT Training

```bash
cd /workspace/Finetuning_LLama
python scripts/03_train_dapt.py
```

**Duration:** ~2-4 hours  
**Monitor:** Check `logs/dapt/` for progress

### 8.2 SFT Training

```bash
python scripts/04_train_sft.py
```

**Duration:** ~6-10 hours (3 epochs)  
**Monitor:** Check `logs/sft/` for progress

---

## 💰 Cost Management

### Monitor Spending
1. RunPod dashboard shows current cost
2. Set budget alerts in Billing settings
3. **Stop pod when not training** to avoid charges

### Stop Pod
1. Go to RunPod dashboard
2. Click **"Stop"** on your pod
3. Data in Volume Disk is preserved
4. Restart when ready to continue

### Estimated Costs
- DAPT: 2-4 hours × $0.89/hr = **$1.78-$3.56**
- SFT: 6-10 hours × $0.89/hr = **$5.34-$8.90**
- **Total:** ~$7-12 for complete training

---

## 🆘 Common Issues

### Issue 1: "Out of Memory"
**Solution:**
- Reduce batch size in config
- Enable gradient checkpointing
- Use smaller LoRA rank

### Issue 2: "Model download failed"
**Solution:**
- Check HuggingFace token
- Verify Llama 3.1 access approved
- Try downloading again

### Issue 3: "Dataset not found"
**Solution:**
- Verify extraction: `ls /workspace/data/arrow/`
- Re-extract: `tar -xzf kp_dataset.tar.gz`

### Issue 4: "Pod is expensive"
**Solution:**
- Use RTX 6000 Ada (~$0.89/hr) not A100
- Stop pod when not training
- Use Spot instances (cheaper but can be interrupted)

---

## 📞 Next Steps

Once setup is complete:
1. Follow `TRAINING_GUIDE.md` for detailed training steps
2. Monitor training in `logs/` directory
3. Test model after each phase
4. Deploy quantized model on RTX 3090

---

**You're ready to train!** 🚀

If you get stuck, check:
- RunPod documentation: https://docs.runpod.io/
- Our training guide: `TRAINING_GUIDE.md`
- Logs: `/workspace/Finetuning_LLama/logs/`
