import json
import os

# The three correct batch outputs (from the user's check)
correct_batch_outputs = [
    'data/dpo/batch_output.jsonl',        # batch_6991fde594f081909ed566431afe5d1b (400 pairs)
    'data/dpo/batch_output_chunk1.jsonl', # batch_6991fdf775848190b197db91fc62bcc7 (400 pairs)  
    'data/dpo/batch_output_chunk2.jsonl'  # batch_6991fe047e788190bd01b77d7e3ce7c3 (200 pairs)
]

all_pairs = []

# Read all batch outputs
for batch_file in correct_batch_outputs:
    if os.path.exists(batch_file):
        print(f"Reading {batch_file}...")
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Parse batch response format
                    if 'custom_id' in data and 'response' in data:
                        custom_id = data['custom_id']
                        response = data['response']['body']['choices'][0]['message']['content']
                        
                        # Extract metadata from custom_id format: chosen_00800_past_events or rejected_00800_past_events
                        parts = custom_id.split('_')
                        pair_type = parts[0]  # 'chosen' or 'rejected'
                        # The middle part is a 5-digit number: first 3=chart_idx, last 2=question_idx
                        number = parts[1]
                        chart_idx = int(number[:3])
                        question_idx = int(number[3:])
                        
                        all_pairs.append({
                            'custom_id': custom_id,
                            'chart_idx': chart_idx,
                            'question_idx': question_idx,
                            'pair_type': pair_type,
                            'response': response
                        })

print(f"Total responses read: {len(all_pairs)}")

# Group by chart_idx and question_idx to create pairs
pairs_dict = {}
for item in all_pairs:
    key = (item['chart_idx'], item['question_idx'])
    if key not in pairs_dict:
        pairs_dict[key] = {}
    pairs_dict[key][item['pair_type']] = item['response']

# Create final DPO pairs
final_pairs = []
for (chart_idx, question_idx), responses in pairs_dict.items():
    if 'chosen' in responses and 'rejected' in responses:
        final_pairs.append({
            'chart_idx': chart_idx,
            'question_idx': question_idx,
            'chosen': responses['chosen'],
            'rejected': responses['rejected']
        })

# Save merged pairs
with open('data/dpo/dpo_pairs_v4_correct.jsonl', 'w', encoding='utf-8') as f:
    for pair in final_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + '\n')

print(f"\n✓ Merged {len(final_pairs)} complete DPO pairs")
print(f"✓ Saved to: data/dpo/dpo_pairs_v4_correct.jsonl")

# Show category distribution if we have combos.json
if os.path.exists('data/dpo/combos.json'):
    with open('data/dpo/combos.json', 'r', encoding='utf-8') as f:
        combos = json.load(f)
    
    # Count categories from merged pairs
    category_counts = {}
    for i, pair in enumerate(final_pairs):
        if i < len(combos):
            cat = combos[i].get('category', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\nCategory distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
