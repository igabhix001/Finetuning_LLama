"""
Prepare DPO Dataset for Training
=================================
Converts raw JSONL DPO pairs into HuggingFace Dataset format
compatible with TRL's DPOTrainer.

Expected input format (JSONL from 13_generate_dpo_dataset.py):
  {"prompt": "...", "chart_yaml": "...", "chosen": "...", "rejected": "...", "category": "..."}

Also loads combos.json (if present) to reconstruct full YAML chart context for prompts.

Output format (HuggingFace Dataset with columns):
  - prompt: formatted as Llama 3.1 chat template (system + user + YAML chart)
  - chosen: assistant response (ideal pandit)
  - rejected: assistant response (bad robotic)

Usage:
  python scripts/14_prepare_dpo_dataset.py
  python scripts/14_prepare_dpo_dataset.py --input data/dpo/dpo_pairs.jsonl --split 0.1
"""

import json
import argparse
import random
from pathlib import Path
from datasets import Dataset, DatasetDict

parser = argparse.ArgumentParser(description="Prepare DPO dataset for training")
parser.add_argument("--input", type=str, default="data/dpo/dpo_pairs.jsonl", help="Input JSONL file")
parser.add_argument("--output-dir", type=str, default="data/dpo/prepared", help="Output directory")
parser.add_argument("--split", type=float, default=0.1, help="Validation split ratio")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

random.seed(args.seed)

print("=" * 80)
print("PREPARING DPO DATASET FOR TRAINING")
print("=" * 80)

# ── Load raw pairs ────────────────────────────────────────────────────────────
input_path = Path(args.input)
if not input_path.exists():
    print(f"❌ Input file not found: {input_path}")
    print("Run 13_generate_dpo_dataset.py first")
    exit(1)

raw_pairs = []
with open(input_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            pair = json.loads(line)
            raw_pairs.append(pair)
        except json.JSONDecodeError as e:
            print(f"  ⚠️ Skipping line {line_num}: {e}")

print(f"✓ Loaded {len(raw_pairs)} raw pairs from {input_path}")

# ── Category distribution ─────────────────────────────────────────────────────
cat_counts = {}
for p in raw_pairs:
    cat = p.get("category", "unknown")
    cat_counts[cat] = cat_counts.get(cat, 0) + 1
print(f"\nCategory distribution:")
for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

# ── Format into Llama 3.1 chat template ──────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Jyotish, a warm and confident KP astrologer — like a trusted family pandit.\n\n"
    "LANGUAGE RULE (HIGHEST PRIORITY): Match the user's language exactly. "
    "English question → 100% English answer. Hindi/Hinglish question → Hindi/Hinglish answer.\n\n"
    "RULES: Answer DIRECTLY with specific Mon YYYY dates from dasha table. "
    "Simple questions = 1 sentence. Timing = 2-3 sentences. MAX 4 sentences. "
    "Cite cusp sub-lord + house numbers. No markdown, no headers, no bullets. "
    "Address as '[Name] ji'. Products ONLY when user asks for remedies. "
    "Read today_date from YAML for correct tense.\n\n"
    "SAFETY: For death/health fear queries, respond with compassion — reassure, "
    "redirect to medical professionals, never scare. "
    "EMOTIONAL: For distress queries, lead with empathy before astrological analysis."
)

# ── Load combos.json for full YAML chart context reconstruction ──────────────
combos_path = Path(args.input).parent / "combos.json"
combos_map = {}  # question+chart_name -> full yaml
if combos_path.exists():
    with open(combos_path, "r", encoding="utf-8") as f:
        combos_list = json.load(f)
    for c in combos_list:
        key = (c.get("question", ""), c.get("chart_name", ""))
        combos_map[key] = c.get("chart_yaml", "")
    print(f"\u2713 Loaded {len(combos_map)} combos for YAML context reconstruction")

def format_prompt(pair: dict) -> dict:
    """Format a DPO pair into Llama 3.1 chat template with YAML chart context."""
    question = pair.get("prompt", "")
    chart_yaml = pair.get("chart_yaml", "")
    chart_name = pair.get("chart_name", "")
    chosen = pair.get("chosen", "")
    rejected = pair.get("rejected", "")
    category = pair.get("category", "")

    # Try to get full YAML from combos_map if chart_yaml is truncated
    if chart_name and question:
        full_yaml = combos_map.get((question, chart_name), "")
        if full_yaml and len(full_yaml) > len(chart_yaml):
            chart_yaml = full_yaml

    # Build user message — YAML format for chart context
    if chart_yaml and len(chart_yaml) > 50:
        user_msg = f"Chart context (YAML):\n{chart_yaml}\n\nQuestion: {question}"
    elif category == "general":
        user_msg = question
    else:
        user_msg = f"Question: {question}"

    # Format as Llama 3.1 chat template
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "category": category,
        "chart_name": chart_name,
    }

# ── Process all pairs ─────────────────────────────────────────────────────────
print(f"\nFormatting {len(raw_pairs)} pairs into Llama 3.1 chat template...")
formatted = [format_prompt(p) for p in raw_pairs]

# Validate
valid = []
for i, f_pair in enumerate(formatted):
    if len(f_pair["chosen"]) < 30:
        print(f"  ⚠️ Skipping pair {i}: chosen too short ({len(f_pair['chosen'])} chars)")
        continue
    if len(f_pair["rejected"]) < 30:
        print(f"  ⚠️ Skipping pair {i}: rejected too short ({len(f_pair['rejected'])} chars)")
        continue
    if f_pair["chosen"] == f_pair["rejected"]:
        print(f"  ⚠️ Skipping pair {i}: chosen == rejected")
        continue
    valid.append(f_pair)

print(f"✓ {len(valid)} valid pairs after filtering")

# ── Train/validation split ────────────────────────────────────────────────────
random.shuffle(valid)
split_idx = int(len(valid) * (1 - args.split))
train_data = valid[:split_idx]
eval_data = valid[split_idx:]

print(f"✓ Train: {len(train_data)} pairs")
print(f"✓ Eval: {len(eval_data)} pairs")

# ── Save as HuggingFace Dataset ───────────────────────────────────────────────
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

train_ds = Dataset.from_list(train_data)
eval_ds = Dataset.from_list(eval_data)

dataset_dict = DatasetDict({
    "train": train_ds,
    "test": eval_ds,
})

dataset_dict.save_to_disk(str(output_dir))

# Also save a sample for inspection
sample_path = output_dir / "sample_pairs.json"
with open(sample_path, "w", encoding="utf-8") as f:
    json.dump(valid[:5], f, indent=2, ensure_ascii=False)

print(f"\n✓ Dataset saved to: {output_dir}")
print(f"✓ Sample saved to: {sample_path}")
print(f"\n{'=' * 80}")
print("DPO DATASET PREPARATION COMPLETE")
print(f"{'=' * 80}")
print(f"Train: {len(train_data)} | Eval: {len(eval_data)}")
print(f"\nNext step: Merge SFT LoRA into base model, then train DPO")
print(f"  python scripts/05b_merge_sft_lora.py")
print(f"  python scripts/15_train_dpo.py")
print(f"{'=' * 80}")
