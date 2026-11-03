import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/Phase_3_Healthcare_Risk_Decision_Support.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get('cells', [])

for idx, c in enumerate(cells):
    src = "".join(c.get('source', []))
    if "joblib.dump" in src or "pickle.dump" in src or ".pkl" in src:
        print(f"Cell {idx}:")
        print(src)
        print("="*50)
