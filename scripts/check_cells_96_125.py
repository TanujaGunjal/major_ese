import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/Phase_3_Healthcare_Risk_Decision_Support.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get('cells', [])

for idx in range(96, 125):
    c = cells[idx]
    src = "".join(c.get('source', []))
    if any(k in src for k in ['fit(', 'RandomForestClassifier', 'predict_proba', 'train_test_split', 'X =', 'dump']):
        print(f"Cell {idx}:")
        print(src)
        print("="*50)
