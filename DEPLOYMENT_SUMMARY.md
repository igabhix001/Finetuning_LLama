# Finetuning_LLama - Ready for GitHub & RunPod

**Status:** ✅ Production-ready for Part 2 training  
**Date:** 2026-02-06  
**Hardware:** RTX 6000 Ada (48GB) → RTX 3090 (24GB)  
**Embeddings:** OpenAI text-embedding-3-large (3072-dim)

---

## 📦 What's Included

### Training Scripts (Production-Ready)
- ✅ `scripts/01_setup_environment.sh` - Environment setup
- ✅ `scripts/02_upload_pinecone.py` - RAG embeddings upload
- ✅ `scripts/03_train_dapt.py` - DAPT with LoRA (2-4 hours)
- ✅ `scripts/04_train_sft.py` - SFT with LoRA on DAPT (6-10 hours)
- ✅ `scripts/05_merge_adapters.py` - Merge DAPT + SFT LoRA
- ✅ `scripts/06_quantize_unsloth.py` - Quantize 8-bit with Unsloth
- ✅ `scripts/07_test_inference.py` - Test model
- ✅ `scripts/08_serve_vllm.py` - vLLM production server

### Configuration Files
- ✅ `configs/dapt_config.yaml` - DAPT training config
- ✅ `configs/dapt_lora_config.yaml` - DAPT LoRA parameters
- ✅ `configs/sft_config.yaml` - SFT training config
- ✅ `configs/lora_config.yaml` - SFT LoRA parameters

### Dataset Files (Already Copied)
- ✅ `data/dapt_corpus/` - 654 chunks, ~1.19M tokens
- ✅ `data/sft_train/` - 19,303 training examples
- ✅ `data/sft_validation/` - 398 validation examples
- ✅ `data/pinecone_upsert.jsonl` - 1,207 OpenAI embeddings (3072-dim)
- ✅ `data/kb_chunks.jsonl` - RAG chunks

### Documentation
- ✅ `README.md` - Complete overview
- ✅ `QUICKSTART.md` - 30-minute setup guide
- ✅ `GITHUB_SETUP.md` - GitHub push instructions
- ✅ `RUNPOD_SETUP.md` - Detailed RunPod guide
- ✅ `TRAINING_GUIDE.md` - Step-by-step training

### Environment Files
- ✅ `.env` - Your API keys (**gitignored**, not pushed to GitHub)
- ✅ `.env.example` - Template for RunPod (includes OpenAI key placeholder)
- ✅ `.gitignore` - Excludes models, checkpoints, logs, .env
- ✅ `requirements.txt` - All dependencies (incl. openai, vllm, unsloth)

---

## 🚀 Quick Start (3 Steps)

### 1. Push to GitHub
```powershell
cd d:\Dataset_preprossecing_pipeline\Finetuning_LLama
git init
git add .
git commit -m "Initial commit: KP Astrology training pipeline"
git remote add origin https://github.com/YOUR_USERNAME/Finetuning_LLama.git
git push -u origin main
```

### 2. Clone on RunPod
```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/Finetuning_LLama.git
cd Finetuning_LLama
cp .env.example .env
nano .env  # Add your keys
bash scripts/01_setup_environment.sh
```

### 3. Start Training
```bash
# LoRA DAPT
python scripts/03_train_dapt.py  # 2-4 hours

# LoRA SFT (on top of DAPT)
python scripts/04_train_sft.py   # 6-10 hours

# Merge adapters
python scripts/05_merge_adapters.py

# Quantize with Unsloth (8-bit)
python scripts/06_quantize_unsloth.py

# Test
python scripts/07_test_inference.py

# Serve with vLLM
python scripts/08_serve_vllm.py
```

---

## 📊 Training Pipeline

**Architecture:** Base Llama-3.1-8B-Instruct → LoRA DAPT → LoRA SFT → Merge adapters → Quantize (8-bit Unsloth) → vLLM serve

| Phase | Script | Duration | Output |
|-------|--------|----------|--------|
| **LoRA DAPT** | `03_train_dapt.py` | 2-4 hrs | DAPT LoRA adapters |
| **LoRA SFT** | `04_train_sft.py` | 6-10 hrs | SFT LoRA adapters |
| **Merge** | `05_merge_adapters.py` | 30 mins | Full merged model |
| **Quantize** | `06_quantize_unsloth.py` | 30 mins | 8-bit quantized model |
| **Test** | `07_test_inference.py` | 5 mins | Quality validation |
| **Serve** | `08_serve_vllm.py` | - | vLLM production server |

**Total:** ~10-16 hours, ~$8-15 on RTX 6000 Ada

---

## 🔧 Key Features

### Optimized for RTX 6000 Ada
- **LoRA-based training** (not full fine-tuning)
- FP16 training (not BF16)
- Gradient checkpointing enabled
- Memory-efficient optimizer (paged_adamw_32bit)
- Batch size: 4 with gradient accumulation 4 (effective: 16)
- DAPT LoRA + SFT LoRA stacked approach

### Production-Ready Code
- Comprehensive error handling
- Progress monitoring with TensorBoard
- Automatic checkpoint saving
- Resume capability on interruption
- Detailed logging

### Quality Assurance
- Validation during training
- Early stopping support
- Test inference script
- Quality metrics tracking

---

## 📁 Directory Structure

```
Finetuning_LLama/
├── README.md                    # Main documentation
├── QUICKSTART.md               # 30-min setup guide
├── GITHUB_SETUP.md             # GitHub instructions
├── RUNPOD_SETUP.md             # RunPod detailed guide
├── TRAINING_GUIDE.md           # Training instructions
├── DEPLOYMENT_SUMMARY.md       # This file
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions
├── .env                        # Your API keys (gitignored)
├── .env.example                # Template
├── configs/                    # Training configurations
│   ├── dapt_config.yaml
│   ├── sft_config.yaml
│   └── lora_config.yaml
├── scripts/                    # Training scripts
│   ├── 01_setup_environment.sh
│   ├── 02_upload_pinecone.py
│   ├── 03_train_dapt.py
│   ├── 04_train_sft.py
│   ├── 05_merge_lora.py
│   ├── 06_quantize_model.py
│   └── 07_test_inference.py
├── data/                       # Datasets (gitignored contents)
│   ├── dapt_corpus/
│   ├── sft_train/
│   ├── sft_validation/
│   ├── pinecone_upsert.jsonl
│   └── kb_chunks.jsonl
├── models/                     # Model storage (gitignored)
├── checkpoints/                # Training checkpoints (gitignored)
└── logs/                       # Training logs (gitignored)
```

---

## 🔑 API Keys Required

On RunPod, copy `.env.example` to `.env` and add your keys:

```bash
# Pinecone (for RAG)
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=kp-astrology-kb

# OpenAI (for RAG query embeddings)
OPENAI_API_KEY=your-openai-key

# HuggingFace (for Llama 3.1 access)
HF_TOKEN=your-hf-token
```

> **Note:** `.env` is gitignored. You must create it on RunPod from `.env.example`.

---

## 💰 Cost Estimate

### Training on RTX 6000 Ada (~$0.89/hr)
- LoRA DAPT: 2-4 hours = $1.78-$3.56
- LoRA SFT: 6-10 hours = $5.34-$8.90
- Merge + Quantize: 1 hour = $0.89
- **Total: $8-14**

### Deployment on RTX 3090 (~$0.34/hr)
- Quantized model: ~8-10GB (8-bit Unsloth)
- vLLM serving: 50-100 tokens/sec
- OpenAI-compatible API
- Cost-effective for production

---

## ✅ Pre-Push Checklist

Before pushing to GitHub:

- [x] All scripts created and tested
- [x] Configuration files complete
- [x] Dataset files copied to data/
- [x] .gitignore configured correctly
- [x] .env.example created (without real keys)
- [x] Documentation complete
- [x] requirements.txt includes all dependencies
- [x] Directory structure clean

---

## 🎯 What Happens on RunPod

1. **Clone:** Get all scripts and configs from GitHub
2. **Setup:** Install dependencies, configure environment
3. **Train DAPT:** Adapt model to KP astrology (2-4 hrs)
4. **Train SFT:** Instruction tuning with LoRA (6-10 hrs)
5. **Merge:** Combine LoRA with base model (30 mins)
6. **Quantize:** Compress to 4-bit for RTX 3090 (1 hr)
7. **Test:** Validate model quality (5 mins)
8. **Download:** Get quantized model for deployment

---

## 📚 Documentation Guide

- **New to RunPod?** → Start with `QUICKSTART.md`
- **Need detailed setup?** → Read `RUNPOD_SETUP.md`
- **Want step-by-step training?** → Follow `TRAINING_GUIDE.md`
- **GitHub questions?** → Check `GITHUB_SETUP.md`
- **Quick reference?** → See `README.md`

---

## 🆘 Support

### Common Issues

**Out of Memory:**
- Reduce batch size in config files
- Enable gradient checkpointing (already enabled)
- Use smaller LoRA rank

**Training Interrupted:**
- Training auto-resumes from checkpoints
- Checkpoints saved every 100/500 steps

**Dataset Missing:**
- Dataset files are in local repo
- Upload separately if needed (see GITHUB_SETUP.md)

**HuggingFace Access Denied:**
- Request access at huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Verify HF_TOKEN in .env

---

## 🎓 Training Metrics to Monitor

### DAPT
- **Loss:** Should decrease from ~3.5 to ~2.5
- **Perplexity:** Should reduce by 30-40%

### SFT
- **Train Loss:** Should decrease from ~2.5 to ~1.2
- **Eval Loss:** Should plateau around 1.3-1.6
- **Quality:** Check rule citations and KP terminology

---

## 🚀 Ready to Deploy!

**Everything is set up and ready for GitHub push.**

Next steps:
1. Review `QUICKSTART.md` for 30-minute setup
2. Push to GitHub using commands above
3. Launch RunPod RTX 6000 Ada pod
4. Clone and start training

**Total time to trained model: ~10-16 hours**  
**Total cost: ~$8-15**

---

**Good luck with training!** 🎯
