import json
import sys
import pandas as pd
import glob
import os

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/Phase_3_Healthcare_Risk_Decision_Support.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get('cells', [])

print("=== CELL 226 (investigation_backend.py) ===")
print("".join(cells[226].get('source', [])))

print("\n=== CELL 272 (latest app.py attempt) ===")
print("".join(cells[272].get('source', []))[:2000])

print("\n=== CHECKING MEDICARE DATASETS IN data/medicare ===")
medicare_files = glob.glob("data/medicare/*.csv")
for f in medicare_files:
    fname = os.path.basename(f)
    try:
        df = pd.read_csv(f, nrows=5)
        # get line count quickly
        with open(f, 'rb') as fp:
            num_lines = sum(1 for _ in fp)
        print(f"\nFile: {fname} | Rows: {num_lines - 1:,} | Cols: {df.shape[1]}")
        print("Columns:", list(df.columns)[:15], ("..." if len(df.columns) > 15 else ""))
    except Exception as e:
        print(f"File: {fname} Error: {e}")
