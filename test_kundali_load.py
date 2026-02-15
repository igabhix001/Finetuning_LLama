import sys
sys.path.insert(0, 'scripts')
from chart_preprocessor import chart_to_yaml
import glob
import json

files = glob.glob('sample_kundali/kundali_*.json')
print(f'Testing {len(files)} kundali files...\n')

errors = []
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        yaml = chart_to_yaml(json.dumps(data))
        name = data.get('name', '?')
        fname = f.replace('\\', '/').split('/')[-1]
        print(f'✓ {fname}: {name} ({len(yaml)} chars)')
    except Exception as e:
        errors.append((f, str(e)))
        print(f'✗ {f}: {e}')

print(f'\n{"="*60}')
print(f'Result: {len(files)-len(errors)}/{len(files)} loaded successfully')
if errors:
    print(f'\nERRORS:')
    for f, e in errors:
        print(f'  {f}: {e}')
