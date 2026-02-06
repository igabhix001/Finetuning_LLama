"""
SFT Training Script - Supervised Fine-Tuning with LoRA
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
sft_config_path = Path("configs/sft_config.yaml")
lora_config_path = Path("configs/lora_config.yaml")

if not sft_config_path.exists():
    print(f"❌ SFT config not found: {sft_config_path}")
    sys.exit(1)

if not lora_config_path.exists():
    print(f"❌ LoRA config not found: {lora_config_path}")
    sys.exit(1)

with open(sft_config_path, 'r') as f:
    config = yaml.safe_load(f)

with open(lora_config_path, 'r') as f:
    lora_config_dict = yaml.safe_load(f)

print("="*80)
print("SFT TRAINING - Supervised Fine-Tuning with LoRA (on DAPT LoRA)")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"CUDA version: {torch.version.cuda}")
print("="*80)

# Verify HF token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN not found in .env file")
    sys.exit(1)

# Load base model
print("\n1. Loading base Llama 3.1 8B Instruct model...")
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
    print(f"❌ Failed to load base model: {e}")
    sys.exit(1)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

print(f"✓ Base model loaded: {model.num_parameters():,} parameters")

# Load DAPT LoRA adapters
print("\n2. Loading DAPT LoRA adapters...")
dapt_lora_path = Path(config['dapt_lora_path'])

if not dapt_lora_path.exists():
    print(f"❌ DAPT LoRA adapters not found: {dapt_lora_path}")
    print("Please run DAPT training first: python scripts/03_train_dapt.py")
    sys.exit(1)

try:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(dapt_lora_path))
    print(f"✓ DAPT LoRA adapters loaded")
    
    # Merge DAPT LoRA into base model for SFT
    print("   Merging DAPT LoRA into base model...")
    model = model.merge_and_unload()
    print(f"✓ DAPT LoRA merged")
except Exception as e:
    print(f"❌ Failed to load DAPT LoRA: {e}")
    sys.exit(1)

# Prepare for SFT LoRA
print("\n3. Preparing model for SFT LoRA training...")
model = prepare_model_for_kbit_training(model)

# LoRA configuration
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
print(f"✓ SFT LoRA applied")
model.print_trainable_parameters()

# Load datasets
print("\n4. Loading SFT datasets...")
train_path = Path(config['train_data'])
eval_path = Path(config['eval_data'])

if not train_path.exists():
    print(f"❌ Training dataset not found: {train_path}")
    sys.exit(1)

if not eval_path.exists():
    print(f"❌ Validation dataset not found: {eval_path}")
    sys.exit(1)

try:
    train_dataset = load_from_disk(str(train_path))
    eval_dataset = load_from_disk(str(eval_path))
    print(f"✓ Train dataset: {len(train_dataset)} examples")
    print(f"✓ Eval dataset: {len(eval_dataset)} examples")
except Exception as e:
    print(f"❌ Failed to load datasets: {e}")
    sys.exit(1)

# Format prompt function
def format_prompt(example):
    """Format example into Llama 3.1 chat format."""
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
    return {"text": prompt}

# Tokenize function
def tokenize_function(examples):
    """Tokenize formatted prompts."""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=config['max_seq_length'],
        padding='max_length',
        return_tensors=None
    )

print("\n5. Formatting and tokenizing datasets...")
train_dataset = train_dataset.map(format_prompt, desc="Formatting train")
eval_dataset = eval_dataset.map(format_prompt, desc="Formatting eval")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[col for col in train_dataset.column_names if col != 'text'],
    desc="Tokenizing train"
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[col for col in eval_dataset.column_names if col != 'text'],
    desc="Tokenizing eval"
)

print(f"✓ Datasets prepared")

# Training arguments
print("\n6. Setting up training configuration...")
output_dir = Path(config['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

logging_dir = Path(config['logging_dir'])
logging_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=config['num_train_epochs'],
    per_device_train_batch_size=config['per_device_train_batch_size'],
    per_device_eval_batch_size=config['per_device_eval_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    learning_rate=config['learning_rate'],
    warmup_ratio=config['warmup_ratio'],
    weight_decay=config['weight_decay'],
    logging_dir=str(logging_dir),
    logging_steps=config['logging_steps'],
    eval_steps=config['eval_steps'],
    save_steps=config['save_steps'],
    save_total_limit=config['save_total_limit'],
    evaluation_strategy=config['evaluation_strategy'],
    load_best_model_at_end=config['load_best_model_at_end'],
    metric_for_best_model=config['metric_for_best_model'],
    greater_is_better=config.get('greater_is_better', False),
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
print(f"   LoRA rank: {lora_config_dict['r']}")
print(f"   LoRA alpha: {lora_config_dict['lora_alpha']}")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
print("\n7. Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

total_steps = len(train_dataset) // (config['per_device_train_batch_size'] * config['gradient_accumulation_steps']) * config['num_train_epochs']
print(f"✓ Trainer initialized")
print(f"   Total training steps: {total_steps}")
print(f"   Estimated time: ~6-10 hours on RTX 6000 Ada")

# Train
print("\n8. Starting SFT training with LoRA...")
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

# Save final SFT LoRA adapters
print("\n9. Saving SFT LoRA adapters...")
final_output = output_dir / "final"
final_output.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(final_output))
tokenizer.save_pretrained(str(final_output))

print(f"\n{'='*80}")
print("SFT TRAINING COMPLETE")
print(f"{'='*80}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"SFT LoRA adapters saved to: {final_output}")
print(f"Logs saved to: {logging_dir}")
print(f"\nNext step: Merge all LoRA adapters")
print(f"  python scripts/05_merge_adapters.py")
print(f"{'='*80}\n")
