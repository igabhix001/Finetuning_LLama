"""
Merge DPO Datasets
==================
Merges the new V2 fixes dataset with the existing dataset.
Filters out the 68 past/future confusion pairs from the old dataset.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_jsonl(filepath):
    """Load JSONL file"""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs

def save_jsonl(pairs, filepath):
    """Save to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

def is_past_future_confused(pair):
    """Check if pair has past/future confusion"""
    prompt = pair.get('prompt', '')
    chosen = pair.get('chosen', '')
    
    # Past tense question patterns
    past_patterns = [
        'what happened',
        'did i',
        'when did',
        'was there',
        'were there',
        'in 2020',
        'in 2021',
        'in 2022',
        'in 2023',
        'in 2024',
        'in 2025',
    ]
    
    is_past_question = any(p in prompt.lower() for p in past_patterns)
    
    if is_past_question:
        # Check if response uses future years (2025+)
        import re
        future_years = re.findall(r'\b(202[5-9]|203\d)\b', chosen)
        if future_years:
            return True
    
    return False

def main():
    print("=" * 80)
    print("Merging DPO Datasets")
    print("=" * 80)
    
    data_dir = Path("data/dpo")
    
    # Load existing dataset
    print("\n📂 Loading existing dataset...")
    existing_file = data_dir / "dpo_pairs.jsonl"
    if not existing_file.exists():
        print(f"❌ File not found: {existing_file}")
        return
    
    existing_pairs = load_jsonl(existing_file)
    print(f"✓ Loaded {len(existing_pairs)} existing pairs")
    
    # Filter out past/future confused pairs
    print("\n🔍 Filtering out past/future confused pairs...")
    filtered_pairs = []
    confused_count = 0
    for pair in existing_pairs:
        if is_past_future_confused(pair):
            confused_count += 1
        else:
            filtered_pairs.append(pair)
    
    print(f"✓ Removed {confused_count} confused pairs")
    print(f"✓ Kept {len(filtered_pairs)} clean pairs")
    
    # Load new V2 dataset
    print("\n📂 Loading new V2 fixes dataset...")
    v2_file = data_dir / "dpo_pairs_v2_fixes.jsonl"
    if not v2_file.exists():
        print(f"⚠️  File not found: {v2_file}")
        print("   Run: python generate_dpo_v2_sync.py --count 500")
        new_pairs = []
    else:
        new_pairs = load_jsonl(v2_file)
        print(f"✓ Loaded {len(new_pairs)} new pairs")
    
    # Merge
    print("\n🔀 Merging datasets...")
    merged_pairs = filtered_pairs + new_pairs
    print(f"✓ Total pairs: {len(merged_pairs)}")
    
    # Save merged dataset
    output_file = data_dir / "dpo_pairs_merged.jsonl"
    save_jsonl(merged_pairs, output_file)
    print(f"\n💾 Saved merged dataset to: {output_file}")
    
    # Backup old dataset
    backup_file = data_dir / "dpo_pairs_backup.jsonl"
    if not backup_file.exists():
        import shutil
        shutil.copy(existing_file, backup_file)
        print(f"💾 Backed up original to: {backup_file}")
    
    # Replace original
    save_jsonl(merged_pairs, existing_file)
    print(f"💾 Updated original file: {existing_file}")
    
    # Stats
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Original pairs: {len(existing_pairs)}")
    print(f"Removed (confused): {confused_count}")
    print(f"New pairs added: {len(new_pairs)}")
    print(f"Final total: {len(merged_pairs)}")
    
    # Quality check
    print("\n📊 Quality Check:")
    chart_names = defaultdict(int)
    for pair in merged_pairs:
        name = pair.get('chart_name', 'Unknown')
        chart_names[name] += 1
    
    print(f"Unique charts: {len(chart_names)}")
    print(f"Pairs per chart (avg): {len(merged_pairs) / len(chart_names):.1f}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Prepare dataset for training:")
    print("   python scripts/14_prepare_dpo_dataset.py")
    print("\n2. Upload to RunPod and train:")
    print("   python scripts/15_train_dpo.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
