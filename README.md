# Inferential Forensic Analysis of Structural Fingerprints in Password Manager Vaults

Official research repository and source code for the paper:  
**"Inferential Forensic Analysis of Structural Fingerprints in Password Manager Vaults"**  
*Charrie Anne A. Rafanan and James Patrick A. Acang*

---

## 📌 Project Overview

Password Management Systems (Bitwarden, KeePassXC, 1Password) enforce strong symmetric cryptography (AES-256, ChaCha20-Poly1305) and Key Derivation Functions (PBKDF2-HMAC-SHA256, Argon2id) to protect stored user credentials. However, upon export, password managers produce distinct, unencrypted structural schema footprints (CSV column headers, JSON key trees, XML tags, and archive directory layouts).

This repository contains the synthetic dataset generation tools, multi-threaded metadata parsers, feature extraction modules, and machine learning models used to demonstrate that **exported vault files leak structural fingerprints capable of identifying the target password manager with up to 100% accuracy using as few as 30 vault files**.

---

## 📂 Repository Structure

```
├── 1password_export_generator/
│   └── generate_1password_exports.py      # Synthetic 1Password CSV & 1PUX export generator
├── bitwarden_export_generator/
│   └── generate_bitwarden_exports.py      # Synthetic Bitwarden CSV & Encrypted JSON generator
├── build_standardized_dataset.py          # Multi-threaded structural feature extraction pipeline
├── standardized_pm_vault_features.csv     # Extracted 70-dimensional boolean feature dataset (30,000 samples)
├── fingerprint_categorical_nb.py          # Categorical Naïve Bayes classification pipeline
├── fingerprint_decision_tree.py           # Decision Tree classification pipeline
├── fingerprint_random_forest.py           # Random Forest classification pipeline
├── run_top_features_and_stacked_area_experiment.py # Feature importance & subsample learning curve experiments
├── eda_pm_comparison.py                   # Exploratory data analysis & summary statistics
├── inspect_vault_dataset.py               # Dataset sanity check & structural verification
└── figures/                               # Visualizations (Learning Curves, Feature Importance, Confusion Matrices)
```

---

## 📊 Dataset Description

The primary feature matrix (`standardized_pm_vault_features.csv`) contains **30,000 synthetic vault exports** (5,000 per format across 6 formats):
- **Bitwarden:** Plaintext CSV, Encrypted JSON
- **KeePassXC:** Plaintext CSV, Hierarchical XML
- **1Password:** Plaintext CSV, 1PUX Archive (Compressed JSON)

### Target Leakage Protection
To evaluate genuine side-channel metadata vulnerabilities, explicit format indicators (`File Extension` such as `.1pux`, `.xml` and vendor `Export Representation`) are **excluded** during model training. Classification is conducted strictly on the 70 binary attribute presence indicators (`attr_*`) derived from unencrypted JSON key paths, XML element tags, and CSV field headers.

---

## 🚀 Key Experimental Findings

1. **High Sample Efficiency:** Tree-based models (Random Forest, Decision Tree) achieve **100.0% accuracy (95% CI: [100.0%, 100.0%])** with a minimal training sample size of **$N=30$ total vault files** (10 per application).
2. **Top Features:** Machine learning feature importance reveals that key predictors include `n_entries`, `naming_style` (`camelCase`, `snake_case`, `PascalCase`, `TitleCase`), `avg_fields_per_entry`, `n_unique_fields`, and `total_fields`.
3. **Encrypted Vault Vulnerability:** Structural leaks persist even in encrypted export modes (e.g., Bitwarden Encrypted JSON), where unencrypted schema keys (`items`, `login.uris`, `folderId`, `uuid`) remain exposed.

---

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.8+
- `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

### Setup Environment
```bash
git clone https://github.com/charrieanne/Password-Manager-Vault-Inforential-Forensic-Analysis.git
cd Password-Manager-Vault-Inforential-Forensic-Analysis
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Reproduce Experiments

1. **Run Feature Extraction / Dataset Inspection:**
   ```bash
   python inspect_vault_dataset.py
   python eda_pm_comparison.py
   ```

2. **Evaluate Machine Learning Classifiers:**
   ```bash
   python fingerprint_categorical_nb.py
   python fingerprint_decision_tree.py
   python fingerprint_random_forest.py
   ```

3. **Run Incremental Subsample Learning Curves & Feature Importance:**
   ```bash
   python run_top_features_and_stacked_area_experiment.py
   ```

---

## 📄 Citation

If you use this dataset or codebase in your research, please cite:

```bibtex
@inproceedings{rafanan2026inferential,
  title={Inferential Forensic Analysis of Structural Fingerprints in Password Manager Vaults},
  author={Rafanan, Charrie Anne A. and Acang, James Patrick A.},
  year={2026}
}
```

---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
