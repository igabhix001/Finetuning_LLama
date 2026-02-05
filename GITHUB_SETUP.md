# GitHub Repository Setup Guide

This guide explains how to push the `Finetuning_LLama` folder to GitHub and clone it on RunPod.

---

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `Finetuning_LLama`
3. Description: "KP Astrology Llama 3.1 8B Fine-tuning Pipeline"
4. Visibility: **Private** (recommended for production code)
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

---

## Step 2: Push to GitHub (Local Machine)

Open PowerShell in your local machine:

```powershell
# Navigate to the Finetuning_LLama folder
cd d:\Dataset_preprossecing_pipeline\Finetuning_LLama

# Initialize git repository
git init

# Add all files (respects .gitignore)
git add .

# Commit
git commit -m "Initial commit: Complete training pipeline for KP Astrology LLM"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/Finetuning_LLama.git

# Push to GitHub
git push -u origin main
```

If you get an error about 'master' vs 'main', run:
```powershell
git branch -M main
git push -u origin main
```

---

## Step 3: Verify Upload

1. Go to https://github.com/YOUR_USERNAME/Finetuning_LLama
2. You should see:
   - ✅ All scripts in `scripts/` folder
   - ✅ All configs in `configs/` folder
   - ✅ README.md, requirements.txt, .gitignore
   - ✅ .env.example (NOT .env - that's gitignored)
   - ❌ NO `data/` folder contents (gitignored - too large)
   - ❌ NO `models/`, `checkpoints/`, `logs/` contents (gitignored)

**Important:** The actual dataset files in `data/` are gitignored because they're too large for GitHub. They're already in the repo locally and will be available when you clone.

---

## Step 4: Clone on RunPod

Once you've launched your RunPod instance:

```bash
# On RunPod terminal
cd /workspace

# Clone repository (replace YOUR_USERNAME)
git clone https://github.com/YOUR_USERNAME/Finetuning_LLama.git

# Navigate to folder
cd Finetuning_LLama

# Verify structure
ls -la
```

You should see:
- ✅ scripts/, configs/, README.md, requirements.txt
- ✅ Empty folders: data/, models/, checkpoints/, logs/
- ✅ .env.example

---

## Step 5: Setup on RunPod

### 5.1 Configure Environment

```bash
# Copy .env.example to .env
cp .env.example .env

# Edit .env with your keys
nano .env
```

Add your actual keys:
```bash
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=kp-astrology-kb
HF_TOKEN=your-huggingface-token-here
```

Save and exit (Ctrl+X, Y, Enter).

### 5.2 Verify Dataset Files

The dataset files are already in the `data/` folder from your local copy. Verify:

```bash
ls -lh data/
```

You should see:
- `dapt_corpus/` - DAPT training data
- `sft_train/` - SFT training data (19,303 examples)
- `sft_validation/` - Validation data (398 examples)
- `pinecone_upsert.jsonl` - Embeddings for RAG
- `kb_chunks.jsonl` - RAG chunks

If files are missing, you'll need to upload them separately (see below).

---

## Step 6: Upload Dataset (If Needed)

If dataset files are not in the repo, upload them separately:

### Option A: Using SCP (from local machine)

```powershell
# On local machine
cd d:\Dataset_preprossecing_pipeline

# Compress dataset
tar -czf finetuning_data.tar.gz Finetuning_LLama/data/

# Upload to RunPod (get SSH details from RunPod UI)
scp -P <PORT> finetuning_data.tar.gz root@<POD_ID>.runpod.io:/workspace/
```

Then on RunPod:
```bash
cd /workspace
tar -xzf finetuning_data.tar.gz
mv Finetuning_LLama/data/* Finetuning_LLama/data/
```

### Option B: Using Google Drive

1. Upload `finetuning_data.tar.gz` to Google Drive
2. Get shareable link
3. On RunPod:
```bash
pip install gdown
gdown <FILE_ID>
tar -xzf finetuning_data.tar.gz
```

---

## Step 7: Install Dependencies

```bash
cd /workspace/Finetuning_LLama

# Run setup script
bash scripts/01_setup_environment.sh
```

This will:
- Install all Python packages
- Setup HuggingFace authentication
- Verify GPU
- Create directory structure

---

## Step 8: Start Training

```bash
# DAPT training (2-4 hours)
python scripts/03_train_dapt.py

# SFT training (6-10 hours)
python scripts/04_train_sft.py

# Merge LoRA weights
python scripts/05_merge_lora.py

# Quantize for deployment
python scripts/06_quantize_model.py

# Test inference
python scripts/07_test_inference.py
```

---

## Updating Code

If you make changes locally and want to update RunPod:

### On Local Machine:
```powershell
cd d:\Dataset_preprossecing_pipeline\Finetuning_LLama
git add .
git commit -m "Update training scripts"
git push
```

### On RunPod:
```bash
cd /workspace/Finetuning_LLama
git pull
```

---

## Important Notes

1. **Dataset Size:** The full dataset is ~500MB-1GB. GitHub has a 100MB file limit, so large files are gitignored.

2. **Model Files:** Trained models (checkpoints, final models) are gitignored because they're huge (16GB+).

3. **Secrets:** Never commit `.env` file with real API keys. Always use `.env.example` as template.

4. **Git LFS:** If you need to store large files in GitHub, consider using Git LFS (Large File Storage).

---

## Troubleshooting

### Issue: Dataset files missing after clone
**Solution:** Upload dataset separately using SCP or Google Drive (see Step 6)

### Issue: Git push fails with "file too large"
**Solution:** Check `.gitignore` is properly configured. Large files should be excluded.

### Issue: Permission denied on RunPod
**Solution:** Ensure you're in `/workspace` directory and have write permissions

### Issue: Git authentication fails
**Solution:** Use personal access token instead of password:
1. Go to GitHub Settings → Developer Settings → Personal Access Tokens
2. Generate new token with `repo` scope
3. Use token as password when pushing

---

**You're ready to train on RunPod!** 🚀
