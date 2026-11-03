import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/Phase_3_Healthcare_Risk_Decision_Support.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get('cells', [])

for idx, c in enumerate(cells):
    src = "".join(c.get('source', []))
    if "%%writefile" in src or "streamlit as st" in src or "duckdb.connect" in src:
        print(f"Cell {idx} ({c.get('cell_type')}):")
        print(src[:400])
        print("...\n" + "-"*40)
