import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/Phase_3_Healthcare_Risk_Decision_Support.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get('cells', [])

for idx in range(16, 45):
    c = cells[idx]
    src = "".join(c.get('source', []))
    if any(k in src for k in ['R1_', 'R2_', 'R3_', 'R4_', 'risk_score', 'risk_category', 'rules', 'consistency']):
        print(f"Cell {idx}:")
        print(src)
        print("="*50)
