# Training Pipeline Architecture

**Complete LoRA-based training pipeline for KP Astrology Llama 3.1 8B**

---

## 🏗️ Architecture Flow

```
Base Model: Llama-3.1-8B-Instruct
         ↓
    LoRA DAPT (Domain Adaptation)
         ↓
    LoRA SFT (Instruction Tuning)
         ↓
  Merge Adapters (DAPT + SFT)
         ↓
 Quantize (8-bit Unsloth)
         ↓
    vLLM Serve (Production)
```

---

## 📋 Detailed Pipeline

### Stage 1: Base Model
- **Model:** `meta-llama/Llama-3.1-8B-Instruct`
- **Size:** ~16GB (FP16)
- **Parameters:** 8 billion
- **Source:** HuggingFace (requires access token)

### Stage 2: LoRA DAPT
- **Method:** Low-Rank Adaptation (LoRA)
- **Target:** All linear layers (q, k, v, o, gate, up, down projections)
- **Rank:** 16
- **Alpha:** 32
- **Trainable:** ~2-3% of total parameters (~160-240M params)
- **Dataset:** 654 chunks, ~1.19M tokens from 6 KP books
- **Purpose:** Adapt model to KP astrology domain
- **Output:** DAPT LoRA adapters (~500MB)

### Stage 3: LoRA SFT
- **Method:** LoRA on top of merged DAPT LoRA
- **Process:**
  1. Load base model
  2. Load DAPT LoRA adapters
  3. Merge DAPT LoRA into base
  4. Apply new SFT LoRA on merged model
- **Target:** Same linear layers
- **Rank:** 16
- **Alpha:** 32
- **Trainable:** ~2-3% of total parameters
- **Dataset:** 19,303 Q&A examples (15k English + 4.3k Hinglish)
- **Purpose:** Instruction tuning for Q&A format
- **Output:** SFT LoRA adapters (~500MB)

### Stage 4: Merge Adapters
- **Process:**
  1. Load base Llama 3.1 8B
  2. Load and merge DAPT LoRA
  3. Load and merge SFT LoRA
  4. Save full merged model
- **Output:** Full FP16 model (~16GB)
- **Contains:** Base + DAPT knowledge + SFT instructions

### Stage 5: Quantize with Unsloth
- **Method:** 8-bit quantization using Unsloth
- **Fallback:** bitsandbytes 8-bit if Unsloth unavailable
- **Input:** Merged FP16 model (~16GB)
- **Output:** 8-bit quantized model (~8-10GB)
- **Quality:** Minimal degradation with 8-bit
- **Target:** RTX 3090 (24GB VRAM)

### Stage 6: vLLM Serving
- **Engine:** vLLM (optimized inference)
- **API:** OpenAI-compatible REST API
- **Features:**
  - Continuous batching
  - PagedAttention
  - Optimized CUDA kernels
- **Endpoints:**
  - `/v1/completions` - Text completion
  - `/v1/chat/completions` - Chat format
  - `/health` - Health check
  - `/docs` - API documentation
- **Performance:** 50-100 tokens/sec on RTX 3090

---

## 🔬 Why This Architecture?

### LoRA Benefits
1. **Memory Efficient:** Only train 2-3% of parameters
2. **Faster Training:** Reduced computation
3. **Modular:** Can swap/combine adapters
4. **Quality:** Comparable to full fine-tuning
5. **Cost Effective:** Lower GPU requirements

### Stacked LoRA Approach
1. **DAPT First:** Learn domain knowledge
2. **SFT Second:** Learn task format
3. **Separate Concerns:** Domain vs. instruction
4. **Better Generalization:** Each stage focused

### 8-bit Quantization
1. **Size Reduction:** 16GB → 8-10GB
2. **Speed:** Faster inference
3. **Quality:** <1% degradation with 8-bit
4. **Deployment:** Fits RTX 3090 comfortably

### vLLM Serving
1. **Performance:** 2-3x faster than vanilla transformers
2. **Batching:** Efficient multi-request handling
3. **Standard API:** OpenAI-compatible
4. **Production Ready:** Battle-tested

---

## 📊 Training Statistics

### LoRA DAPT
- **Epochs:** 1
- **Batch Size:** 4 (effective: 16 with grad accumulation)
- **Learning Rate:** 2e-4
- **Duration:** 2-4 hours on RTX 6000 Ada
- **VRAM Usage:** ~20-25GB
- **Cost:** ~$1.78-$3.56

### LoRA SFT
- **Epochs:** 3
- **Batch Size:** 4 (effective: 16 with grad accumulation)
- **Learning Rate:** 2e-4
- **Duration:** 6-10 hours on RTX 6000 Ada
- **VRAM Usage:** ~25-30GB
- **Cost:** ~$5.34-$8.90

### Total Training
- **Time:** 10-16 hours
- **Cost:** $8-14 on RTX 6000 Ada
- **Output:** Production-ready 8-bit model

---

## 🎯 Quality Metrics

### DAPT Success Criteria
- ✅ Loss decreases from ~3.5 to ~2.5
- ✅ Perplexity reduction of 30-40%
- ✅ Model uses KP terminology naturally

### SFT Success Criteria
- ✅ Validation loss plateaus around 1.2-1.5
- ✅ Answers cite specific KP rules
- ✅ Proper answer format (instruction → output)
- ✅ Confidence levels included
- ✅ Hinglish code-mixing natural

### Deployment Success Criteria
- ✅ Model loads on RTX 3090 (24GB)
- ✅ Inference speed: 50-100 tokens/sec
- ✅ API responses match OpenAI format
- ✅ Quality comparable to pre-quantization

---

## 🔧 Technical Details

### LoRA Configuration
```yaml
# DAPT LoRA
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

# SFT LoRA
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

### Training Configuration
```yaml
# Optimization
fp16: true
gradient_checkpointing: true
optim: paged_adamw_32bit
lr_scheduler_type: cosine

# Memory
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
max_seq_length: 2048
```

### vLLM Configuration
```python
# Server settings
max_model_len: 2048
gpu_memory_utilization: 0.9
dtype: float16
quantization: bitsandbytes  # For 8-bit
trust_remote_code: True
```

---

## 🚀 Deployment Flow

1. **Training on RunPod (RTX 6000 Ada)**
   - Train DAPT LoRA
   - Train SFT LoRA
   - Merge adapters
   - Quantize to 8-bit

2. **Download Model**
   - Compressed: ~8-10GB
   - Transfer to deployment server

3. **Deploy on RTX 3090**
   - Load 8-bit model
   - Start vLLM server
   - Upload Pinecone embeddings (RAG)

4. **Production Serving**
   - OpenAI-compatible API
   - RAG-enhanced responses
   - Monitor performance

---

## 📚 References

- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **Llama 3.1:** https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- **Unsloth:** https://github.com/unslothai/unsloth
- **vLLM:** https://github.com/vllm-project/vllm
- **PEFT Library:** https://github.com/huggingface/peft

---

**This architecture ensures efficient training, high quality, and production-ready deployment.** 🎯
