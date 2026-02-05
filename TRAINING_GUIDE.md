# Training Guide - Step-by-Step Instructions

**Prerequisites:** Complete `RUNPOD_SETUP.md` first

---

## 📋 Training Overview

### Pipeline Stages
1. **DAPT** (2-4 hours) - Domain adaptation
2. **SFT** (6-10 hours) - Instruction tuning
3. **Merge** (30 mins) - Combine LoRA weights
4. **Quantize** (1 hour) - Prepare for deployment
5. **Test** (15 mins) - Validate model

**Total Time:** ~10-16 hours  
**Total Cost:** ~$7-12 on RTX 6000 Ada

---

## Stage 1: DAPT Training

### What is DAPT?
Domain-Adaptive Pre-Training teaches the model KP astrology terminology and concepts before instruction tuning.

### Dataset
- **File:** `/workspace/data/arrow/dapt_corpus/`
- **Size:** 654 chunks, ~1.19M tokens
- **Content:** Raw text from 6 KP astrology books

### Configuration

Create `/workspace/Finetuning_LLama/configs/dapt_config.yaml`:

```yaml
# DAPT Training Configuration
model_name: "/workspace/Finetuning_LLama/models/base/"
output_dir: "/workspace/Finetuning_LLama/checkpoints/dapt/"
logging_dir: "/workspace/Finetuning_LLama/logs/dapt/"

# Dataset
train_data: "/workspace/data/arrow/dapt_corpus/"

# Training parameters
num_train_epochs: 1
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1e-5
warmup_steps: 100
logging_steps: 10
save_steps: 100
save_total_limit: 3

# Optimization
fp16: true
gradient_checkpointing: true
optim: "adamw_torch"
lr_scheduler_type: "cosine"

# Context
max_seq_length: 2048
```

### Training Script

Create `/workspace/Finetuning_LLama/scripts/03_train_dapt.py`:

```python
"""
DAPT Training Script
"""

import os
import yaml
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk

# Load config
with open('configs/dapt_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("="*70)
print("DAPT TRAINING - Domain Adaptive Pre-Training")
print("="*70)

# Load model and tokenizer
print("\n1. Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    torch_dtype="auto",
    device_map="auto"
)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

print(f"✓ Model loaded: {model.num_parameters():,} parameters")

# Load dataset
print("\n2. Loading DAPT corpus...")
dataset = load_from_disk(config['train_data'])
print(f"✓ Dataset loaded: {len(dataset)} examples")

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=config['max_seq_length'],
        padding='max_length'
    )

print("\n3. Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)
print(f"✓ Tokenization complete")

# Training arguments
print("\n4. Setting up training...")
training_args = TrainingArguments(
    output_dir=config['output_dir'],
    num_train_epochs=config['num_train_epochs'],
    per_device_train_batch_size=config['per_device_train_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    learning_rate=config['learning_rate'],
    warmup_steps=config['warmup_steps'],
    logging_dir=config['logging_dir'],
    logging_steps=config['logging_steps'],
    save_steps=config['save_steps'],
    save_total_limit=config['save_total_limit'],
    fp16=config['fp16'],
    gradient_checkpointing=config['gradient_checkpointing'],
    optim=config['optim'],
    lr_scheduler_type=config['lr_scheduler_type'],
    report_to="tensorboard"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train
print("\n5. Starting DAPT training...")
print(f"   Epochs: {config['num_train_epochs']}")
print(f"   Batch size: {config['per_device_train_batch_size']}")
print(f"   Gradient accumulation: {config['gradient_accumulation_steps']}")
print(f"   Effective batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
print(f"   Learning rate: {config['learning_rate']}")
print("\n" + "="*70)

trainer.train()

# Save final model
print("\n6. Saving DAPT model...")
output_path = Path(config['output_dir']) / "final"
trainer.save_model(str(output_path))
tokenizer.save_pretrained(str(output_path))

print(f"\n{'='*70}")
print("DAPT TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Model saved to: {output_path}")
print(f"Logs saved to: {config['logging_dir']}")
print(f"\nNext step: Run SFT training")
print(f"  python scripts/04_train_sft.py")
print(f"{'='*70}\n")
```

### Run DAPT

```bash
cd /workspace/Finetuning_LLama
python scripts/03_train_dapt.py
```

### Monitor Progress

```bash
# In another terminal
tensorboard --logdir=/workspace/Finetuning_LLama/logs/dapt/

# Or check logs
tail -f /workspace/Finetuning_LLama/logs/dapt/train.log
```

### Expected Metrics
- **Initial loss:** ~3.5-4.0
- **Final loss:** ~2.5-3.0
- **Perplexity reduction:** 30-40%

---

## Stage 2: SFT Training

### What is SFT?
Supervised Fine-Tuning teaches the model to answer questions in the correct format with proper reasoning.

### Dataset
- **Training:** `/workspace/data/arrow/sft_train/` (19,303 examples)
- **Validation:** `/workspace/data/arrow/sft_validation/` (398 examples)

### LoRA Configuration

Create `/workspace/Finetuning_LLama/configs/lora_config.yaml`:

```yaml
# LoRA Configuration
r: 16
lora_alpha: 32
target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
lora_dropout: 0.05
bias: "none"
task_type: "CAUSAL_LM"
```

### SFT Configuration

Create `/workspace/Finetuning_LLama/configs/sft_config.yaml`:

```yaml
# SFT Training Configuration
model_name: "/workspace/Finetuning_LLama/checkpoints/dapt/final/"
output_dir: "/workspace/Finetuning_LLama/checkpoints/sft/"
logging_dir: "/workspace/Finetuning_LLama/logs/sft/"

# Dataset
train_data: "/workspace/data/arrow/sft_train/"
eval_data: "/workspace/data/arrow/sft_validation/"

# Training parameters
num_train_epochs: 3
per_device_train_batch_size: 4
per_device_eval_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
warmup_ratio: 0.03
logging_steps: 10
eval_steps: 100
save_steps: 500
save_total_limit: 3
evaluation_strategy: "steps"
load_best_model_at_end: true
metric_for_best_model: "eval_loss"

# Optimization
fp16: true
gradient_checkpointing: true
optim: "paged_adamw_32bit"
lr_scheduler_type: "cosine"

# Context
max_seq_length: 2048
```

### Training Script

Create `/workspace/Finetuning_LLama/scripts/04_train_sft.py`:

```python
"""
SFT Training Script with LoRA
"""

import os
import yaml
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load configs
with open('configs/sft_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open('configs/lora_config.yaml', 'r') as f:
    lora_config_dict = yaml.safe_load(f)

print("="*70)
print("SFT TRAINING - Supervised Fine-Tuning with LoRA")
print("="*70)

# Load model and tokenizer
print("\n1. Loading DAPT model...")
tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    torch_dtype="auto",
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

print(f"✓ Model loaded: {model.num_parameters():,} parameters")

# Prepare for LoRA
print("\n2. Preparing model for LoRA...")
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=lora_config_dict['r'],
    lora_alpha=lora_config_dict['lora_alpha'],
    target_modules=lora_config_dict['target_modules'],
    lora_dropout=lora_config_dict['lora_dropout'],
    bias=lora_config_dict['bias'],
    task_type=lora_config_dict['task_type']
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load datasets
print("\n3. Loading SFT datasets...")
train_dataset = load_from_disk(config['train_data'])
eval_dataset = load_from_disk(config['eval_data'])
print(f"✓ Train: {len(train_dataset)} examples")
print(f"✓ Eval: {len(eval_dataset)} examples")

# Format prompt
def format_prompt(example):
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
    return {"text": prompt}

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=config['max_seq_length'],
        padding='max_length'
    )

print("\n4. Formatting and tokenizing...")
train_dataset = train_dataset.map(format_prompt)
eval_dataset = eval_dataset.map(format_prompt)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Training arguments
print("\n5. Setting up training...")
training_args = TrainingArguments(
    output_dir=config['output_dir'],
    num_train_epochs=config['num_train_epochs'],
    per_device_train_batch_size=config['per_device_train_batch_size'],
    per_device_eval_batch_size=config['per_device_eval_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    learning_rate=config['learning_rate'],
    warmup_ratio=config['warmup_ratio'],
    logging_dir=config['logging_dir'],
    logging_steps=config['logging_steps'],
    eval_steps=config['eval_steps'],
    save_steps=config['save_steps'],
    save_total_limit=config['save_total_limit'],
    evaluation_strategy=config['evaluation_strategy'],
    load_best_model_at_end=config['load_best_model_at_end'],
    metric_for_best_model=config['metric_for_best_model'],
    fp16=config['fp16'],
    gradient_checkpointing=config['gradient_checkpointing'],
    optim=config['optim'],
    lr_scheduler_type=config['lr_scheduler_type'],
    report_to="tensorboard"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train
print("\n6. Starting SFT training...")
print(f"   Epochs: {config['num_train_epochs']}")
print(f"   Effective batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
print("\n" + "="*70)

trainer.train()

# Save
print("\n7. Saving SFT model...")
output_path = Path(config['output_dir']) / "final"
trainer.save_model(str(output_path))

print(f"\n{'='*70}")
print("SFT TRAINING COMPLETE")
print(f"{'='*70}")
print(f"LoRA adapters saved to: {output_path}")
print(f"\nNext step: Merge LoRA weights")
print(f"  python scripts/05_merge_lora.py")
print(f"{'='*70}\n")
```

### Run SFT

```bash
cd /workspace/Finetuning_LLama
python scripts/04_train_sft.py
```

### Expected Metrics
- **Initial loss:** ~2.5-3.0 (from DAPT)
- **Final loss:** ~1.2-1.5
- **Eval loss:** Should plateau around 1.3-1.6

---

## Stage 3: Merge LoRA Weights

Combine LoRA adapters with base model for full model.

Create `/workspace/Finetuning_LLama/scripts/05_merge_lora.py`:

```python
"""
Merge LoRA weights with base model
"""

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("="*70)
print("MERGING LORA WEIGHTS")
print("="*70)

# Paths
base_model_path = "/workspace/Finetuning_LLama/checkpoints/dapt/final/"
lora_path = "/workspace/Finetuning_LLama/checkpoints/sft/final/"
output_path = "/workspace/Finetuning_LLama/models/sft/"

# Load base model
print("\n1. Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Load LoRA
print("\n2. Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, lora_path)

# Merge
print("\n3. Merging weights...")
model = model.merge_and_unload()

# Save
print("\n4. Saving merged model...")
Path(output_path).mkdir(parents=True, exist_ok=True)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"\n{'='*70}")
print("MERGE COMPLETE")
print(f"{'='*70}")
print(f"Merged model saved to: {output_path}")
print(f"\nNext step: Quantize for deployment")
print(f"  python scripts/06_quantize_model.py")
print(f"{'='*70}\n")
```

Run:
```bash
python scripts/05_merge_lora.py
```

---

## Stage 4: Quantize for Deployment

Quantize to 4-bit for RTX 3090 deployment.

Create `/workspace/Finetuning_LLama/scripts/06_quantize_model.py`:

```python
"""
Quantize model for deployment
"""

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

print("="*70)
print("QUANTIZING MODEL")
print("="*70)

# Paths
model_path = "/workspace/Finetuning_LLama/models/sft/"
output_path = "/workspace/Finetuning_LLama/models/quantized/"

# Load model
print("\n1. Loading merged model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Quantize
print("\n2. Quantizing to 4-bit...")
quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)

# Save
print("\n3. Saving quantized model...")
Path(output_path).mkdir(parents=True, exist_ok=True)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"\n{'='*70}")
print("QUANTIZATION COMPLETE")
print(f"{'='*70}")
print(f"Quantized model saved to: {output_path}")
print(f"Model size: ~5GB (ready for RTX 3090)")
print(f"\nNext step: Test inference")
print(f"  python scripts/07_test_inference.py")
print(f"{'='*70}\n")
```

Run:
```bash
python scripts/06_quantize_model.py
```

---

## Stage 5: Test Inference

Test the final model.

Create `/workspace/Finetuning_LLama/scripts/07_test_inference.py`:

```python
"""
Test model inference
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("="*70)
print("TESTING MODEL INFERENCE")
print("="*70)

# Load model
model_path = "/workspace/Finetuning_LLama/models/quantized/"

print("\n1. Loading quantized model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7
)

# Test questions
test_questions = [
    "What does the 7th house sub-lord signify in marriage timing?",
    "Explain the role of Venus in KP astrology for relationships.",
    "How do I calculate the ruling planets for a horary question?"
]

print("\n2. Testing predictions...\n")

for i, question in enumerate(test_questions, 1):
    print(f"{'='*70}")
    print(f"Test {i}: {question}")
    print(f"{'='*70}")
    
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    response = pipe(prompt)[0]['generated_text']
    answer = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    
    print(f"\nAnswer:\n{answer}\n")

print(f"{'='*70}")
print("TESTING COMPLETE")
print(f"{'='*70}\n")
```

Run:
```bash
python scripts/07_test_inference.py
```

---

## ✅ Training Complete!

Your model is now ready for deployment on RTX 3090.

### Next Steps
1. Download quantized model from RunPod
2. Deploy on RTX 3090 with vLLM
3. Integrate Pinecone RAG
4. Connect to client API

See `../Finetunning_runpod.md` Section 4 for deployment instructions.
