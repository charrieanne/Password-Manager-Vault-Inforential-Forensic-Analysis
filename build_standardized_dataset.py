"""
Build Standardized Dataset for 30,000 Individual Vault Export Files
===================================================================
Scans 30,000 generated vault export files across Bitwarden, KeePassXC, and 1Password in:
  C:\\Users\\Lenovo\\Desktop\\Password Assessment\\Generate Vault\\output

Unit of Analysis: 1 Row = 1 Exported Vault File (Total 30,000 files)
"""

import csv
import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

BASE_OUTPUT_DIR = Path("C:/Users/Lenovo/Desktop/Password Assessment/Generate Vault/output")
OUT_CSV = Path(__file__).parent / "standardized_pm_vault_features.csv"

VAULT_FOLDERS = [
    (BASE_OUTPUT_DIR / "Bitwarden" / "csv", "Bitwarden", "csv", ".csv"),
    (BASE_OUTPUT_DIR / "Bitwarden" / "encrypted_json", "Bitwarden", "bitwarden_json", ".json"),
    (BASE_OUTPUT_DIR / "KeyPassXC" / "csv", "KeePassXC", "csv", ".csv"),
    (BASE_OUTPUT_DIR / "KeyPassXC" / "xml", "KeePassXC", "keepassxc_xml", ".xml"),
    (BASE_OUTPUT_DIR / "1Password" / "csv", "1Password", "csv", ".csv"),
    (BASE_OUTPUT_DIR / "1Password" / "1pux", "1Password", "1password_1pux", ".1pux"),
]


def naming_style(field_names):
    if not field_names:
        return "unknown"
    snake = sum(bool(re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", f)) for f in field_names)
    camel = sum(bool(re.fullmatch(r"[a-z]+([A-Z][a-z0-9]*)+", f)) for f in field_names)
    pascal = sum(bool(re.fullmatch(r"([A-Z][a-z0-9]*)+", f)) for f in field_names)
    title = sum(bool(re.fullmatch(r"[A-Z][a-zA-Z0-9]*( [A-Z][a-zA-Z0-9]*)*", f)) for f in field_names)
    counts = {"snake_case": snake, "camelCase": camel, "PascalCase": pascal, "Title Case": title}
    return max(counts, key=counts.get)


def parse_csv_file(path):
    all_fields = []
    n_entries = 0
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
            n_entries += 1
            all_fields.extend(fieldnames)
    return n_entries, all_fields


def parse_bitwarden_json_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    all_fields = []
    for item in items:
        field_seq = list(item.keys())
        login = item.get("login", {}) if isinstance(item.get("login"), dict) else {}
        flat_login_keys = ["login." + k for k in login.keys()]
        all_fields.extend(field_seq + flat_login_keys)
    return len(items), all_fields


def parse_1password_1pux_file(path):
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        data_name = next((n for n in names if n.endswith("export.data")), None)
        if data_name is None:
            return 0, []
        with zf.open(data_name) as f:
            data = json.load(f)

    all_fields = []
    n_entries = 0
    for account in data.get("accounts", []):
        for vault in account.get("vaults", []):
            for item in vault.get("items", []):
                n_entries += 1
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
                all_fields.extend(top_seq + overview_seq + details_top_seq + section_field_types)
    return n_entries, all_fields


def parse_keepassxc_xml_file(path):
    tree = ET.parse(path)
    root = tree.getroot()
    all_fields = []
    n_entries = 0
    for entry_el in root.iter("Entry"):
        n_entries += 1
        field_seq = []
        string_keys = []
        for child in entry_el:
            field_seq.append(child.tag)
            if child.tag == "String":
                key_el = child.find("Key")
                if key_el is not None and key_el.text:
                    string_keys.append(key_el.text)
        all_fields.extend(field_seq + string_keys)
    return n_entries, all_fields


def parse_single_vault_file(file_args):
    file_path, pm_label, fmt = file_args
    try:
        if fmt == "csv":
            n_entries, all_fields = parse_csv_file(file_path)
        elif fmt == "bitwarden_json":
            n_entries, all_fields = parse_bitwarden_json_file(file_path)
        elif fmt == "keepassxc_xml":
            n_entries, all_fields = parse_keepassxc_xml_file(file_path)
        elif fmt == "1password_1pux":
            n_entries, all_fields = parse_1password_1pux_file(file_path)
        else:
            return None

        unique_fields_in_file = set(all_fields)
        total_fields = len(all_fields)
        avg_fields = (total_fields / n_entries) if n_entries > 0 else 0.0

        return {
            "file_name": os.path.basename(file_path),
            "pm": pm_label,
            "export_format": fmt,
            "n_entries": n_entries,
            "total_fields": total_fields,
            "avg_fields_per_entry": round(avg_fields, 2),
            "n_unique_fields": len(unique_fields_in_file),
            "naming_style": naming_style(all_fields),
            "all_fields_set": unique_fields_in_file,
        }
    except Exception as err:
        print(f"[WARNING] Failed to parse {file_path}: {err}")
        return None


def collect_all_vault_files():
    file_tasks = []
    for folder, pm, fmt, ext in VAULT_FOLDERS:
        if not os.path.exists(folder):
            print(f"[WARNING] Directory does not exist: {folder}")
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)]
        print(f"[INFO] Discovered {len(files)} {ext} files in {folder} ({pm} - {fmt})")
        for f in files:
            file_tasks.append((f, pm, fmt))

    print(f"\n[INFO] Total vault export files discovered: {len(file_tasks):,}")
    return file_tasks


def main():
    print("=== Standardized 30,000 Vault Export File Dataset Builder ===")
    file_tasks = collect_all_vault_files()
    if not file_tasks:
        print("[ERROR] No vault files found!")
        return

    print("\n[INFO] Extracting features from all 30,000 vault files using multi-threading...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(parse_single_vault_file, file_tasks))

    valid_records = [r for r in results if r is not None]
    print(f"[INFO] Successfully parsed {len(valid_records):,} vault export files.")

    all_global_attributes = sorted(list(set(attr for r in valid_records for attr in r["all_fields_set"])))
    print(f"\n[INFO] Discovered {len(all_global_attributes)} total unique attributes.")

    rows = []
    for r in valid_records:
        file_attr_set = r["all_fields_set"]
        row = {
            "file_name": r["file_name"],
            "pm": r["pm"],
            "export_format": r["export_format"],
            "n_entries": r["n_entries"],
            "total_fields": r["total_fields"],
            "avg_fields_per_entry": r["avg_fields_per_entry"],
            "n_unique_fields": r["n_unique_fields"],
            "naming_style": r["naming_style"],
        }
        for attr in all_global_attributes:
            row[f"attr_{attr}"] = attr in file_attr_set

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[INFO] Final Standardized DataFrame shape: {df.shape} (30,000 vault files preserved)")
    print(f"[INFO] PM distribution:\n{df['pm'].value_counts()}")

    df.to_csv(OUT_CSV, index=False)
    print(f"\n[SUCCESS] Standardized dataset saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
