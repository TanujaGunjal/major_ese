import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/Phase_3_Healthcare_Risk_Decision_Support.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get('cells', [])

for idx in [61, 62, 63, 64, 65, 66, 67, 78, 80, 81, 82, 88, 89, 90, 91, 92, 94, 95]:
    c = cells[idx]
    src = "".join(c.get('source', []))
    print(f"--- CELL {idx} ({c.get('cell_type')}) ---")
    print(src)
