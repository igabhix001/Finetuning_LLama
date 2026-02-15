from datasets import load_from_disk
import json

# Load train dataset
ds = load_from_disk('data/dpo/prepared/train')

# Check first 3 samples
print('Checking first 3 training samples:')
print('='*80)
for i in range(min(3, len(ds))):
    sample = ds[i]
    print(f'\nSample {i+1}:')
    print(f'Prompt length: {len(sample["prompt"])} chars')
    print(f'Chosen length: {len(sample["chosen"])} chars')
    print(f'Rejected length: {len(sample["rejected"])} chars')
    print(f'\nChosen preview: {sample["chosen"][:200]}...')
    print(f'\nRejected preview: {sample["rejected"][:200]}...')
    print('='*80)

# Check if chosen is consistently shorter (would indicate swap)
chosen_shorter = 0
rejected_shorter = 0
for sample in ds:
    if len(sample["chosen"]) < len(sample["rejected"]):
        chosen_shorter += 1
    else:
        rejected_shorter += 1

print(f'\nLength comparison across {len(ds)} samples:')
print(f'Chosen shorter: {chosen_shorter} ({chosen_shorter/len(ds)*100:.1f}%)')
print(f'Rejected shorter: {rejected_shorter} ({rejected_shorter/len(ds)*100:.1f}%)')

if chosen_shorter > len(ds) * 0.8:
    print('\n✅ GOOD: Chosen is consistently shorter (as designed)')
elif rejected_shorter > len(ds) * 0.8:
    print('\n❌ PROBLEM: Rejected is consistently shorter - LABELS MAY BE SWAPPED!')
else:
    print('\n⚠️ MIXED: Length varies - this is normal for diverse dataset')
