"""
Inspect Vault Dataset Structure & Counts
========================================
Analyzes the standardized PM export dataset to report:
  1. Total vault entries per Password Manager (Bitwarden, KeePassXC, 1Password)
  2. Total vault entries per Export Format (CSV, Bitwarden JSON, KeePassXC XML, 1PUX)
  3. Total unique structural attribute columns (one-hot features)
  4. Attribute breakdown per Password Manager
"""

import os
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).parent / "standardized_pm_vault_features.csv"


def inspect_csv():
    if not os.path.exists(DATA_PATH):
        print(f"[INFO] Dataset file not found at {DATA_PATH}. Running builder...")
        from build_standardized_dataset import main as build_main
        build_main()

    df = pd.read_csv(DATA_PATH)
    print("==========================================================")
    print("      PM VAULT STANDARDIZED DATASET METRICS SUMMARY       ")
    print("==========================================================")
    print(f"Total Combined Entries (Rows): {len(df):,}")
    print(f"Total Feature Columns: {df.shape[1]:,}")
    
    attr_cols = [c for c in df.columns if c.startswith("attr_")]
    print(f"Total One-Hot Attribute Columns (attr_*): {len(attr_cols):,}")
    
    print("\n--- Breakdown by Password Manager ---")
    print(df["pm"].value_counts().to_string())

    print("\n--- Breakdown by Export Format ---")
    print(df["export_format"].value_counts().to_string())

    print("\n--- Breakdown by PM & Export Format ---")
    print(df.groupby(["pm", "export_format"]).size().to_string())

    print("\n--- Unique Attributes List ---")
    for idx, col in enumerate(attr_cols, 1):
        clean_name = col.replace("attr_", "")
        print(f"  {idx:2d}. {clean_name}")

    print("\n==========================================================")


if __name__ == "__main__":
    inspect_csv()
