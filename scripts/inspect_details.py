import json

with open("notebooks/Phase_3_Healthcare_Risk_Decision_Support.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get('cells', [])

def print_cell(idx):
    c = cells[idx]
    print(f"==================== CELL {idx} ({c.get('cell_type')}) ====================")
    print("".join(c.get('source', [])))
    print("\n")

print("--- NFHS RULES DEFINITIONS ---")
for i in [13, 14, 15, 17, 24, 28, 42]:
    print_cell(i)

print("--- MEDICARE FEATURE ENGINEERING ---")
for i in [58, 59, 60, 61, 62, 63]:
    print_cell(i)

print("--- MODEL TRAINING & EVALUATION ---")
for i in [65, 79, 81, 82, 105, 112]:
    print_cell(i)

print("--- BACKEND / DUCKDB & APP ATTEMPTS ---")
for i in [147, 159, 186]:
    print_cell(i)
