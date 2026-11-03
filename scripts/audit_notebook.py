import json
import os

notebook_path = "notebooks/Phase_3_Healthcare_Risk_Decision_Support.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in notebook: {len(nb.get('cells', []))}")

cells = nb.get('cells', [])
code_cells = [c for c in cells if c.get('cell_type') == 'code']
markdown_cells = [c for c in cells if c.get('cell_type') == 'markdown']

print(f"Code cells: {len(code_cells)}, Markdown cells: {len(markdown_cells)}")

# Print markdown headings
print("\n--- KEY HEADINGS ---")
for i, c in enumerate(markdown_cells):
    src = "".join(c.get('source', []))
    lines = [line.strip() for line in src.split("\n") if line.strip().startswith("#")]
    for line in lines:
        print(f"Cell {i}: {line}")

# Search for specific keywords in code cells
keywords = ['NFHS', 'iair', 'glucose', 'hypertension', 'consistency', 'rules', 'PotentialFraud', 'provider_level_features', 'RandomForest', 'XGB', 'shap', 'streamlit', 'duckdb', 'threshold']

print("\n--- CODE CELL HIGHLIGHTS ---")
for idx, c in enumerate(code_cells):
    src = "".join(c.get('source', []))
    matched = [k for k in keywords if k.lower() in src.lower()]
    if matched:
        first_line = src.strip().split("\n")[0] if src.strip() else ""
        # print first few lines of important cells
        print(f"Code Cell {idx} matches {matched}: {first_line[:80]}")
