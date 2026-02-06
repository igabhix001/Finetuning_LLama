"""
DAPT Training Script - Domain Adaptive Pre-Training with LoRA
Optimized for RTX 6000 Ada (48GB VRAM)
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load configs
config_path = Path("configs/dapt_config.yaml")
lora_config_path = Path("configs/dapt_lora_config.yaml")

if not config_path.exists():
    print(f"❌ Config file not found: {config_path}")
    sys.exit(1)

if not lora_config_path.exists():
    print(f"❌ LoRA config not found: {lora_config_path}")
    sys.exit(1)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

with open(lora_config_path, 'r') as f:
    lora_config_dict = yaml.safe_load(f)

print("="*80)
print("DAPT TRAINING - Domain Adaptive Pre-Training with LoRA")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"CUDA version: {torch.version.cuda}")
print("="*80)

# Verify HF token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN not found in .env file")
    print("Please add your HuggingFace token to access Llama 3.1")
    sys.exit(1)

# Load model and tokenizer
print("\n1. Loading Llama 3.1 8B Instruct base model...")
print(f"   Model: {config['model_name']}")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        token=hf_token,
        trust_remote_code=True
    )
    
    # Select dtype: bf16 preferred on Ada GPUs, fp16 fallback
    if config.get('bf16', False):
        model_dtype = torch.bfloat16
    elif config.get('fp16', False):
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        token=hf_token,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # Saves VRAM on Ada GPUs
    )
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("\nTroubleshooting:")
    print("  1. Verify HF_TOKEN is correct")
    print("  2. Check Llama 3.1 access approval at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    sys.exit(1)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

print(f"✓ Model loaded: {model.num_parameters():,} parameters")
print(f"✓ Model dtype: {model.dtype}")
print(f"✓ Device map: {model.hf_device_map}")

# Prepare for LoRA
print("\n2. Preparing model for LoRA training...")
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(
    r=lora_config_dict['r'],
    lora_alpha=lora_config_dict['lora_alpha'],
    target_modules=lora_config_dict['target_modules'],
    lora_dropout=lora_config_dict['lora_dropout'],
    bias=lora_config_dict['bias'],
    task_type=lora_config_dict['task_type'],
    inference_mode=lora_config_dict.get('inference_mode', False)
)

model = get_peft_model(model, lora_config)
print(f"✓ LoRA applied to DAPT")
model.print_trainable_parameters()

# Load dataset
print("\n3. Loading DAPT corpus...")
dataset_path = Path(config['train_data'])

if not dataset_path.exists():
    print(f"❌ Dataset not found: {dataset_path}")
    print("Please ensure data/dapt_corpus/ exists")
    sys.exit(1)

try:
    dataset = load_from_disk(str(dataset_path))
    print(f"✓ Dataset loaded: {len(dataset)} examples")
    print(f"✓ Dataset features: {dataset.column_names}")
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    sys.exit(1)

# Tokenize function
def tokenize_function(examples):
    """Tokenize text for DAPT."""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=config['max_seq_length'],
        padding='max_length',
        return_tensors=None
    )

print("\n4. Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)
print(f"✓ Tokenization complete")

# Training arguments
print("\n5. Setting up training configuration...")
output_dir = Path(config['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

logging_dir = Path(config['logging_dir'])
logging_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=config['num_train_epochs'],
    per_device_train_batch_size=config['per_device_train_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    learning_rate=config['learning_rate'],
    warmup_steps=config['warmup_steps'],
    weight_decay=config['weight_decay'],
    logging_dir=str(logging_dir),
    logging_steps=config['logging_steps'],
    save_steps=config['save_steps'],
    save_total_limit=config['save_total_limit'],
    fp16=config['fp16'],
    bf16=config.get('bf16', False),
    gradient_checkpointing=config['gradient_checkpointing'],
    optim=config['optim'],
    lr_scheduler_type=config['lr_scheduler_type'],
    max_grad_norm=config['max_grad_norm'],
    dataloader_num_workers=config.get('dataloader_num_workers', 4),
    dataloader_pin_memory=config.get('dataloader_pin_memory', True),
    report_to=config.get('report_to', 'tensorboard'),
    logging_first_step=True,
    save_safetensors=True,
    gradient_checkpointing_kwargs=config.get('gradient_checkpointing_kwargs', {})
)

print(f"✓ Training configuration:")
print(f"   Epochs: {config['num_train_epochs']}")
print(f"   Batch size per device: {config['per_device_train_batch_size']}")
print(f"   Gradient accumulation: {config['gradient_accumulation_steps']}")
print(f"   Effective batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
print(f"   Learning rate: {config['learning_rate']}")
print(f"   FP16: {config['fp16']}")
print(f"   Gradient checkpointing: {config['gradient_checkpointing']}")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)

# Trainer
print("\n6. Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Calculate training stats
total_steps = len(tokenized_dataset) // (config['per_device_train_batch_size'] * config['gradient_accumulation_steps']) * config['num_train_epochs']
print(f"✓ Trainer initialized")
print(f"   Total training steps: {total_steps}")
print(f"   Estimated time: ~2-4 hours on RTX 6000 Ada")

# Train
print("\n7. Starting DAPT training with LoRA...")
print("="*80)
print("Training in progress... Monitor with:")
print(f"  tensorboard --logdir={logging_dir}")
print("="*80)

try:
    trainer.train()
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("Saving checkpoint...")
    trainer.save_model(str(output_dir / "interrupted"))
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    sys.exit(1)

# Save final LoRA adapters
print("\n8. Saving DAPT LoRA adapters...")
final_output = output_dir / "final"
final_output.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(final_output))
tokenizer.save_pretrained(str(final_output))

print(f"\n{'='*80}")
print("DAPT TRAINING COMPLETE")
print(f"{'='*80}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"DAPT LoRA adapters saved to: {final_output}")
print(f"Logs saved to: {logging_dir}")
print(f"\nNext step: Run SFT training with LoRA")
print(f"  python scripts/04_train_sft.py")
print(f"{'='*80}\n")
