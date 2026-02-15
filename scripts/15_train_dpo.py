"""
DPO Training Script — Stage 3: Preference Optimization
========================================================
Trains a NEW LoRA on the merged DAPT+SFT model using DPO.

Correct staging approach:
  Stage 1 (DAPT): Base → LoRA → merge
  Stage 2 (SFT):  Merged → LoRA → merge
  Stage 3 (DPO):  Merged → NEW LoRA (this script)

Uses TRL's DPOTrainer with LoRA for memory-efficient training.

Usage:
  python scripts/15_train_dpo.py
  python scripts/15_train_dpo.py --config configs/dpo_config.yaml
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Suppress known false-positive tokenizer regex warning ─────────────────────
# transformers incorrectly flags Llama 3.1 tokenizer as having a Mistral regex issue.
# See: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84
import warnings
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")

# ── Load configs ──────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser(description="DPO Training with LoRA")
parser.add_argument("--config", type=str, default="configs/dpo_config.yaml")
parser.add_argument("--lora-config", type=str, default="configs/dpo_lora_config.yaml")
cli_args = parser.parse_args()

with open(cli_args.config, "r") as f:
    config = yaml.safe_load(f)

with open(cli_args.lora_config, "r") as f:
    lora_config_dict = yaml.safe_load(f)

print("=" * 80)
print("DPO TRAINING — Stage 3: Preference Optimization with LoRA")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 80)

# ── Import TRL (must be pre-installed — see requirements.txt) ────────────────
try:
    from trl import DPOTrainer, DPOConfig
except ImportError:
    print("❌ TRL package not installed. Install it with:")
    print(f"   {sys.executable} -m pip install trl>=0.9.0")
    print("   Or: pip install -r requirements.txt")
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk


class DPOHealthCallback(TrainerCallback):
    """Stop training if DPO margins blow up (sign of collapse/overfitting)."""
    def __init__(self, max_margin=3.0, min_loss=0.05):
        self.max_margin = max_margin
        self.min_loss = min_loss

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        margin = logs.get("rewards/margins", 0)
        loss = logs.get("loss", 1.0)
        if margin > self.max_margin:
            print(f"\n⚠️  EARLY STOP: margins={margin:.2f} exceeded {self.max_margin}. Stopping to prevent collapse.")
            control.should_training_stop = True
        if loss < self.min_loss and state.global_step > 10:
            print(f"\n⚠️  EARLY STOP: loss={loss:.4f} < {self.min_loss}. Model is overfitting.")
            control.should_training_stop = True

# ── Load merged DAPT+SFT model ───────────────────────────────────────────────
model_path = config["model_name"]
print(f"\n1. Loading merged DAPT+SFT model from: {model_path}")

hf_token = os.getenv("HF_TOKEN")

# Determine dtype
if config.get("bf16", False):
    model_dtype = torch.bfloat16
elif config.get("fp16", False):
    model_dtype = torch.float16
else:
    model_dtype = torch.float32

# Check if model_path is local or HF hub
if Path(model_path).exists():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=hf_token,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

print(f"   ✓ Model loaded: {model.num_parameters():,} parameters")

# ── Load reference model (CRITICAL for DPO) ─────────────────────────────────
print("\n2. Loading reference model (frozen copy for DPO)...")
# DPO requires a separate reference model to compute KL divergence
# Without this, policy == reference → zero preference signal → loss stuck at 0.6931
if Path(model_path).exists():
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=hf_token,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
ref_model.eval()  # Freeze reference model
for param in ref_model.parameters():
    param.requires_grad = False
print(f"   ✓ Reference model loaded and frozen")

# ── Apply DPO LoRA to policy model ───────────────────────────────────────────
print("\n3. Applying DPO LoRA adapter to policy model...")
lora_config = LoraConfig(
    r=lora_config_dict["r"],
    lora_alpha=lora_config_dict["lora_alpha"],
    target_modules=lora_config_dict["target_modules"],
    lora_dropout=lora_config_dict["lora_dropout"],
    bias=lora_config_dict["bias"],
    task_type=lora_config_dict["task_type"],
    inference_mode=False,
)

model = get_peft_model(model, lora_config)
print("   ✓ DPO LoRA applied to policy model")
model.print_trainable_parameters()

# ── Load DPO dataset ─────────────────────────────────────────────────────────
print("\n4. Loading DPO dataset...")
train_path = Path(config["train_data"])
eval_path = Path(config["eval_data"])

if not train_path.exists():
    print(f"❌ Train dataset not found: {train_path}")
    print("Run 14_prepare_dpo_dataset.py first")
    sys.exit(1)

train_dataset = load_from_disk(str(train_path))
eval_dataset = load_from_disk(str(eval_path)) if eval_path.exists() else None

print(f"   ✓ Train: {len(train_dataset)} pairs")
if eval_dataset:
    print(f"   ✓ Eval: {len(eval_dataset)} pairs")

# ── DPO Training config ──────────────────────────────────────────────────────
print("\n5. Setting up DPO training configuration...")
output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

logging_dir = Path(config["logging_dir"])
logging_dir.mkdir(parents=True, exist_ok=True)

max_steps_cfg = config.get("max_steps", -1)

training_args = DPOConfig(
    output_dir=str(output_dir),
    num_train_epochs=config["num_train_epochs"],
    max_steps=max_steps_cfg,
    per_device_train_batch_size=config["per_device_train_batch_size"],
    per_device_eval_batch_size=config["per_device_eval_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    learning_rate=config["learning_rate"],
    warmup_ratio=config["warmup_ratio"],
    weight_decay=config["weight_decay"],
    logging_dir=str(logging_dir),
    logging_steps=config["logging_steps"],
    eval_steps=config["eval_steps"],
    save_steps=config["save_steps"],
    save_total_limit=config["save_total_limit"],
    eval_strategy=config.get("evaluation_strategy", "steps"),
    load_best_model_at_end=config["load_best_model_at_end"],
    metric_for_best_model=config["metric_for_best_model"],
    greater_is_better=config.get("greater_is_better", False),
    fp16=config["fp16"],
    bf16=config.get("bf16", False),
    gradient_checkpointing=config["gradient_checkpointing"],
    optim=config["optim"],
    lr_scheduler_type=config["lr_scheduler_type"],
    max_grad_norm=config["max_grad_norm"],
    report_to=config.get("report_to", "tensorboard"),
    logging_first_step=True,
    save_safetensors=True,
    gradient_checkpointing_kwargs=config.get("gradient_checkpointing_kwargs", {}),
    # DPO-specific
    beta=config.get("beta", 0.1),
    loss_type=config.get("loss_type", "sigmoid"),
    label_smoothing=config.get("label_smoothing", 0.0),
    max_length=config.get("max_length", 1024),
    max_prompt_length=config.get("max_prompt_length", 512),
)

print(f"   ✓ DPO Config:")
print(f"     Beta: {config.get('beta', 0.1)}")
print(f"     Loss type: {config.get('loss_type', 'sigmoid')}")
print(f"     Epochs: {config['num_train_epochs']}")
print(f"     Batch size: {config['per_device_train_batch_size']}")
print(f"     Grad accumulation: {config['gradient_accumulation_steps']}")
print(f"     Effective batch: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
print(f"     Learning rate: {config['learning_rate']}")
print(f"     Label smoothing: {config.get('label_smoothing', 0.0)}")
print(f"     LoRA rank: {lora_config_dict['r']}")
print(f"     Max length: {config.get('max_length', 1024)}")

es_patience = config.get("early_stopping_patience", 2)
print(f"     Early stopping: patience={es_patience} evals on eval_loss")
print(f"     Health guard: stop if margins > 3.0 or loss < 0.05")

# ── Initialize DPO Trainer ────────────────────────────────────────────────────
print("\n6. Initializing DPO Trainer...")
callbacks = [
    EarlyStoppingCallback(early_stopping_patience=es_patience),
    DPOHealthCallback(max_margin=3.0, min_loss=0.05),
]

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # CRITICAL: Pass separate reference model
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    callbacks=callbacks,
)

total_steps = (
    len(train_dataset)
    // (config["per_device_train_batch_size"] * config["gradient_accumulation_steps"])
    * config["num_train_epochs"]
)
print(f"   ✓ Trainer initialized")
print(f"     Total training steps: ~{total_steps}")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n7. Starting DPO training...")
print("=" * 80)
print("Training in progress... Monitor with:")
print(f"  tensorboard --logdir={logging_dir}")
print("=" * 80)

try:
    trainer.train()
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("Saving checkpoint...")
    trainer.save_model(str(output_dir / "interrupted"))
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Save final DPO LoRA ──────────────────────────────────────────────────────
print("\n8. Saving DPO LoRA adapters...")
final_output = output_dir / "final"
final_output.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(final_output))
tokenizer.save_pretrained(str(final_output))

print(f"\n{'=' * 80}")
print("DPO TRAINING COMPLETE")
print(f"{'=' * 80}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"DPO LoRA adapters saved to: {final_output}")
print(f"Logs: {logging_dir}")
print(f"\nTraining pipeline complete:")
print(f"  ✓ Stage 1: DAPT LoRA (domain adaptation)")
print(f"  ✓ Stage 2: SFT LoRA (instruction tuning)")
print(f"  ✓ Stage 3: DPO LoRA (preference optimization)")
print(f"\nNext step: Merge DPO LoRA into final model")
print(f"  python scripts/16_merge_dpo_lora.py")
print(f"{'=' * 80}")
