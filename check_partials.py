import json
data = json.load(open('test_results_30q.json', encoding='utf-8'))
for r in data:
    if 'PARTIAL' in r['status'] or 'FAIL' in r['status']:
        print(f"[{r['id']}] {r['question']}")
        print(f"  Response: {r['response'][:350]}")
        print()
