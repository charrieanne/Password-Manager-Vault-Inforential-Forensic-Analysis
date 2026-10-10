"""
Exploratory Data Analysis (EDA) - PM Structural Comparison
==========================================================
Generates dataset-level visualizations:
  (a) Field Count Distribution by PM (Boxplot)
  (b) Dominant Field-Naming Convention by PM (Stacked Bar Chart)

Note: This analysis is MODEL-INDEPENDENT (evaluates raw dataset properties).
"""

import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. STRUCTURAL PARSERS
# ---------------------------------------------------------------------------

def naming_style(field_names):
    snake = sum(bool(re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", f)) for f in field_names)
    camel = sum(bool(re.fullmatch(r"[a-z]+([A-Z][a-z0-9]*)+", f)) for f in field_names)
    pascal = sum(bool(re.fullmatch(r"([A-Z][a-z0-9]*)+", f)) for f in field_names)
    title = sum(bool(re.fullmatch(r"[A-Z][a-zA-Z0-9]*( [A-Z][a-zA-Z0-9]*)*", f)) for f in field_names)
    counts = {"snake_case": snake, "camelCase": camel, "PascalCase": pascal, "Title Case": title}
    return max(counts, key=counts.get)


def parse_csv_generic(path, pm_label):
    entries = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        sniffer = csv.Sniffer()
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = sniffer.sniff(sample, delimiters=",;\t")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        for _ in reader:
            entries.append({"field_sequence": list(fieldnames), "pm": pm_label})
    return entries


def parse_bitwarden_json(path, pm_label):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    entries = []
    for item in items:
        field_seq = list(item.keys())
        login = item.get("login", {}) if isinstance(item.get("login"), dict) else {}
        flat_login_keys = list(login.keys())
        entries.append({
            "field_sequence": field_seq + ["login." + k for k in flat_login_keys],
            "pm": pm_label,
        })
    return entries


def parse_1password_1pux(path, pm_label):
    import zipfile
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        data_name = next((n for n in names if n.endswith("export.data")), None)
        if data_name is None:
            raise ValueError(f"No export.data found inside {path}")
        with zf.open(data_name) as f:
            data = json.load(f)

    entries = []
    for account in data.get("accounts", []):
        for vault in account.get("vaults", []):
            for item in vault.get("items", []):
                overview = item.get("overview", {}) if isinstance(item.get("overview"), dict) else {}
                details = item.get("details", {}) if isinstance(item.get("details"), dict) else {}
                top_seq = list(item.keys())
                overview_seq = ["overview." + k for k in overview.keys()]
                details_top_seq = ["details." + k for k in details.keys() if k != "sections"]
                sections = details.get("sections", []) if isinstance(details.get("sections"), list) else []
                section_field_types = []
                for sec in sections:
                    for fld in sec.get("fields", []) if isinstance(sec.get("fields"), list) else []:
                        ftype = fld.get("fieldType") or fld.get("t") or "field"
                        section_field_types.append(f"section.{ftype}")
                entries.append({
                    "field_sequence": top_seq + overview_seq + details_top_seq + section_field_types,
                    "pm": pm_label,
                })
    return entries


def parse_keepassxc_xml(path, pm_label):
    tree = ET.parse(path)
    root = tree.getroot()
    entries = []
    for entry_el in root.iter("Entry"):
        field_seq = []
        string_keys = []
        for child in entry_el:
            field_seq.append(child.tag)
            if child.tag == "String":
                key_el = child.find("Key")
                if key_el is not None and key_el.text:
                    string_keys.append(key_el.text)
        entries.append({
            "field_sequence": field_seq + string_keys,
            "pm": pm_label,
        })
    return entries


def load_all(file_map):
    all_entries = []
    for path, pm, fmt in file_map:
        if fmt == "csv":
            all_entries.extend(parse_csv_generic(path, pm))
        elif fmt == "bitwarden_json":
            all_entries.extend(parse_bitwarden_json(path, pm))
        elif fmt == "keepassxc_xml":
            all_entries.extend(parse_keepassxc_xml(path, pm))
        elif fmt == "1password_1pux":
            all_entries.extend(parse_1password_1pux(path, pm))
        else:
            raise ValueError(f"Unknown format: {fmt}")
    return all_entries

# ---------------------------------------------------------------------------
# 2. FEATURE EXTRACTION & PLOTTING
# ---------------------------------------------------------------------------

def plot_pm_comparison(df, out_path="pm_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pms = sorted(df["pm"].unique())
    data = [df.loc[df["pm"] == pm, "n_fields"] for pm in pms]
    
    # Subplot 1: Field Count Distribution
    axes[0].boxplot(data, tick_labels=pms)
    axes[0].set_ylabel("Number of Fields per Entry")
    axes[0].set_title("Field Count Distribution by PM")
    axes[0].grid(alpha=0.3)

    # Subplot 2: Dominant Field-Naming Convention
    style_counts = df.groupby(["pm", "naming_style"]).size().unstack(fill_value=0)
    style_counts = style_counts.reindex(pms)
    style_counts_prop = style_counts.div(style_counts.sum(axis=1), axis=0)
    bottom = np.zeros(len(pms))
    for style in style_counts_prop.columns:
        vals = style_counts_prop[style].values
        axes[1].bar(pms, vals, bottom=bottom, label=style)
        bottom += vals
    axes[1].set_ylabel("Proportion of Entries")
    axes[1].set_title("Dominant Field-Naming Convention by PM")
    axes[1].legend(title="Naming style", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[EDA] PM structural comparison plot saved to {out_path}")


def main(file_map):
    all_entries = load_all(file_map)
    print(f"[EDA] Total entries loaded: {len(all_entries)}")
    rows = []
    for e in all_entries:
        seq = e["field_sequence"]
        rows.append({
            "pm": e["pm"],
            "n_fields": len(seq),
            "naming_style": naming_style(seq),
        })
    df = pd.DataFrame(rows)
    plot_pm_comparison(df, "pm_comparison.png")


if __name__ == "__main__":
    FILES = [
        ("C:\\Users\\Lenovo\\Desktop\\Password Assessment\\Bitwarden\\bitwarden_5000.csv", "Bitwarden", "csv"),
        ("C:\\Users\\Lenovo\\Desktop\\Password Assessment\\Bitwarden\\bitwarden_5000_encrypted.json", "Bitwarden", "bitwarden_json"),
        ("C:\\Users\\Lenovo\\Desktop\\Password Assessment\\KeyPass\\keepassxc_5000_export.csv", "KeePassXC", "csv"),
        ("C:\\Users\\Lenovo\\Desktop\\Password Assessment\\KeyPass\\keepassxc_5000_export.xml", "KeePassXC", "keepassxc_xml"),
        ("C:\\Users\\Lenovo\\Desktop\\Password Assessment\\1Password\\1PasswordExport.csv", "1Password", "csv"),
        ("C:\\Users\\Lenovo\\Desktop\\Password Assessment\\1Password\\1PasswordExport.1pux", "1Password", "1password_1pux"),
    ]
    main(FILES)
