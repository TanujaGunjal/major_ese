import zipfile
import os
from pathlib import Path

output_zip = "healthcare_claims_fraud_phase4.zip"
print(f"Creating project distribution ZIP: {output_zip}...")

base_dir = Path(".")
excluded_dirs = {".venv", ".pytest_cache", "__pycache__", ".git"}
excluded_extensions = {".pyc", ".pyo", ".log"}

with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(base_dir):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith(".")]

        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(base_dir)

            # Skip the output zip itself and excluded extensions
            if file == output_zip or file_path.suffix in excluded_extensions:
                continue

            # Skip raw medicare large CSVs to keep zip manageable, but include Parquet, DuckDB, models, code
            if "data\\medicare" in str(rel_path) or "data/medicare" in str(rel_path):
                if file.endswith(".csv") and "final_model_comparison" not in file and "provider_level_features" not in file:
                    continue

            zipf.write(file_path, arcname=str(rel_path))

zip_size_mb = Path(output_zip).stat().st_size / (1024 * 1024)
print(f"Successfully created {output_zip} ({zip_size_mb:.2f} MB)")
