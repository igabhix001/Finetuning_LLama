import os
import shutil

# Files to keep
keep_files = {
    'dpo_pairs_FINAL_COMPLETE.jsonl',  # Final merged dataset
    'batch_meta.json',  # Batch tracking metadata
    'combos.json',  # Question-chart combinations
}

# Directories to keep
keep_dirs = {
    'prepared'  # Prepared training data
}

# List all files in data/dpo
dpo_dir = 'data/dpo'
all_items = os.listdir(dpo_dir)

files_to_remove = []
for item in all_items:
    item_path = os.path.join(dpo_dir, item)
    
    # Skip if it's a directory we want to keep
    if os.path.isdir(item_path) and item in keep_dirs:
        continue
    
    # Skip if it's a file we want to keep
    if os.path.isfile(item_path) and item in keep_files:
        continue
    
    # Mark for removal
    files_to_remove.append(item)

print(f"Found {len(files_to_remove)} files/dirs to clean up:")
for item in sorted(files_to_remove):
    print(f"  - {item}")

print(f"\nKeeping {len(keep_files)} essential files:")
for item in sorted(keep_files):
    print(f"  ✓ {item}")

# Ask for confirmation
print(f"\nThis will remove {len(files_to_remove)} items. Proceed? (y/n)")
response = input().strip().lower()

if response == 'y':
    removed_count = 0
    for item in files_to_remove:
        item_path = os.path.join(dpo_dir, item)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
            removed_count += 1
            print(f"  ✓ Removed: {item}")
        except Exception as e:
            print(f"  ✗ Failed to remove {item}: {e}")
    
    print(f"\n✅ Cleanup complete! Removed {removed_count}/{len(files_to_remove)} items")
    print(f"✅ data/dpo now contains only essential files")
else:
    print("Cleanup cancelled.")
