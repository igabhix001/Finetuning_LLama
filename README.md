# Finetuning_LLama - Part 2: Training Pipeline

**Purpose:** Complete training pipeline for KP Astrology Llama 3.1 8B model  
**Hardware:** RTX 6000 Ada (48GB VRAM) for training → RTX 3090 (24GB VRAM) for deployment  
**Repository:** Clone this repo on RunPod to get all scripts and datasets

---

## 📁 Folder Structure

```
Finetuning_LLama/
├── README.md                    # This file
├── QUICKSTART.md               # 30-minute quick start
├── RUNPOD_SETUP.md             # Beginner-friendly RunPod guide
├── TRAINING_GUIDE.md           # Step-by-step training instructions
├── configs/                    # Training configurations
│   ├── dapt_config.yaml        # DAPT training config
│   ├── dapt_lora_config.yaml   # DAPT LoRA parameters
│   ├── sft_config.yaml         # SFT training config
│   └── lora_config.yaml        # SFT LoRA parameters
├── scripts/                    # Training scripts
│   ├── 01_setup_environment.sh # Environment setup
│   ├── 02_upload_pinecone.py   # Upload embeddings to Pinecone
│   ├── 03_train_dapt.py        # LoRA DAPT training
│   ├── 04_train_sft.py         # LoRA SFT training
│   ├── 05_merge_adapters.py    # Merge DAPT + SFT LoRA
│   ├── 06_quantize_unsloth.py  # Quantize with Unsloth (8-bit)
│   ├── 07_test_inference.py    # Test model
│   └── 08_serve_vllm.py        # vLLM production server
├── checkpoints/                # Model checkpoints (gitignored)
│   ├── dapt/
│   ├── sft/
│   └── merged/
├── logs/                       # Training logs
│   ├── dapt/
│   └── sft/
├── models/                     # Final models (gitignored)
│   ├── base/                   # Llama 3.1 8B Instruct base
│   ├── dapt/                   # After DAPT
│   ├── sft/                    # After SFT
│   └── quantized/              # Quantized for deployment
└── data/                       # Symlinks to dataset
    ├── dapt_corpus/            → ../../data/arrow/dapt_corpus/
    ├── sft_train/              → ../../data/arrow/sft_train/
    ├── sft_validation/         → ../../data/arrow/sft_validation/
    └── embeddings/             → ../../data/final/pinecone_upsert.jsonl
```

---

## 🎯 Training Pipeline Overview

**Architecture:** Base Llama-3.1-8B-Instruct → LoRA DAPT → LoRA SFT → Merge adapters → Quantize (8-bit Unsloth) → vLLM serve

### Phase 1: LoRA DAPT (Domain-Adaptive Pre-Training)
- **Dataset:** 654 chunks, ~1.19M tokens from 6 KP books
- **Duration:** ~2-4 hours on RTX 6000 Ada
- **Method:** LoRA adapters (not full fine-tuning)
- **Purpose:** Adapt Llama to KP astrology terminology
- **Output:** DAPT LoRA adapters

### Phase 2: LoRA SFT (Supervised Fine-Tuning)
- **Dataset:** 19,303 Q&A examples (15k English + 4.3k Hinglish)
- **Duration:** ~6-10 hours on RTX 6000 Ada (3 epochs)
- **Method:** LoRA on top of merged DAPT LoRA
- **Purpose:** Teach reasoning, persona, and answer format
- **Output:** SFT LoRA adapters

### Phase 3: Merge Adapters
- **Merge:** Combine DAPT + SFT LoRA with base model
- **Duration:** ~30 minutes
- **Output:** Full merged model (~16GB)

### Phase 4: Quantize with Unsloth
- **Method:** 8-bit quantization using Unsloth
- **Duration:** ~30 minutes
- **Output:** Production-ready model (~8-10GB)

### Phase 5: vLLM Serving
- **Upload:** 1,207 OpenAI text-embedding-3-large embeddings (3072-dim) to Pinecone
- **Setup:** vLLM inference server with OpenAI-compatible API
- **Deployment:** RTX 3090 (24GB VRAM)
- **Performance:** 50-100 tokens/sec

---

## 🚀 Quick Start

### 1. Setup RunPod
1. Create RunPod account and launch RTX 6000 Ada pod
2. See `RUNPOD_SETUP.md` for detailed setup guide

### 2. Clone Repository on RunPod
```bash
# On RunPod terminal
cd /workspace
git clone https://github.com/YOUR_USERNAME/Finetuning_LLama.git
cd Finetuning_LLama
```

### 3. Configure Environment
```bash
# On RunPod
cd /workspace/Finetuning_LLama

# Create .env file with your keys
cp .env.example .env
nano .env  # Add your Pinecone, OpenAI, and HuggingFace keys

# Install dependencies
pip install -r requirements.txt
```

### 4. Run Training
```bash
# On RunPod
cd /workspace/Finetuning_LLama

# Setup environment
bash scripts/01_setup_environment.sh

# LoRA DAPT
python scripts/03_train_dapt.py

# LoRA SFT (on DAPT)
python scripts/04_train_sft.py

# Merge adapters
python scripts/05_merge_adapters.py

# Quantize with Unsloth (8-bit)
python scripts/06_quantize_unsloth.py

# Test
python scripts/07_test_inference.py

# Serve with vLLM
python scripts/08_serve_vllm.py --host 0.0.0.0 --port 8000
```

---


## 📊 Expected Results

### LoRA DAPT Metrics
- Perplexity reduction: ~30-40%
- Loss: Should decrease from ~3.5 to ~2.5
- Trainable parameters: ~2-3% of total

### LoRA SFT Metrics
- Validation loss: Should plateau around 1.2-1.5
- Answer quality: Rule citations, proper KP terminology
- Bilingual: Natural code-mixing in Hinglish
- Trainable parameters: ~2-3% of total

### Final Model
- Size: ~8-10GB (8-bit quantized with Unsloth)
- Inference: ~50-100 tokens/sec on RTX 3090
- Quality: Grounded answers with rule citations
- API: OpenAI-compatible via vLLM

---

## 🔧 Hardware Requirements

### Training (RTX 6000 Ada - 48GB)
- LoRA DAPT: ~20-25GB VRAM (LoRA is memory-efficient)
- LoRA SFT: ~25-30GB VRAM
- Batch size: 4 with gradient accumulation 4 (effective: 16)

### Deployment (RTX 3090 - 24GB)
- Quantized model: ~8-10GB (8-bit Unsloth)
- vLLM serving: ~15-18GB VRAM total
- Context window: 2K tokens (configurable)
- Throughput: 50-100 tokens/sec

---

## 📝 Important Notes

1. **LoRA-based:** Both DAPT and SFT use LoRA (not full fine-tuning)
2. **Stacked approach:** SFT LoRA is applied on top of merged DAPT LoRA
3. **8-bit quantization:** Using Unsloth (not GPTQ 4-bit)
4. **vLLM serving:** OpenAI-compatible API for production
5. **DPO Deferred:** Not included in this phase (requires human feedback)
6. **Product SKUs:** Deferred to post-training business logic layer
7. **Pinecone API:** Keys in `.env` file
8. **Checkpointing:** Enabled - can resume if interrupted
9. **Logging:** All metrics logged to `logs/` directory

---

## 🆘 Troubleshooting

### Out of Memory
- Reduce batch size in config
- Enable gradient checkpointing
- Use smaller LoRA rank (r=8 instead of r=16)

### Slow Training
- Check GPU utilization (`nvidia-smi`)
- Verify data loading (should be from Arrow format)
- Enable mixed precision (fp16/bf16)

### Poor Quality
- Check validation loss (should decrease)
- Sample predictions during training
- Verify dataset quality (no corruption)

---

## 📚 References

- HLA: `../Dataset_Preparation.md` (Part 1)
- Training specs: `../Finetunning_runpod.md` (Part 2)
- Datasets: `../data/arrow/`
- Embeddings: `../data/final/pinecone_upsert.jsonl`

---

**Ready to train!** Follow `RUNPOD_SETUP.md` for step-by-step instructions.
