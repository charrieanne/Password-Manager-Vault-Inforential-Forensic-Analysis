"""
Stage 1: PM Vault Fingerprinting - Categorical Naive Bayes Model (Grouped Feature Importance)
=============================================================================================
- Unit of Analysis: 1 Row = 1 Exported Vault File (30,000 Total Vault Files)
- Grouped Feature Importance: Aggregates 70+ attributes into 5 functional categories to save plot space
- Learning Curve: Evaluates up to 30,000 files, plots N = 10 to 100
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "standardized_pm_vault_features.csv"

# Steps evaluated up to 30,000 entries
DEFAULT_STEPS = (
    list(range(10, 101, 10)) +          # 10, 20, 30, ..., 100
    list(range(150, 501, 50)) +         # 150, 200, ..., 500
    list(range(600, 1001, 100)) +       # 600, 700, ..., 1000
    list(range(1500, 5001, 500)) +      # 1500, 2000, ..., 5000
    list(range(6000, 30001, 1000))      # 6000, 7000, ..., 30000
)

# Attribute Grouping Definition to save plot space
ATTRIBUTE_GROUPS = {
    "Field Naming Convention": ["naming_style"],
    "Auth Credentials": ["login.username", "login.password", "String.UserName", "String.Password", "section.concealed", "username", "password"],
    "Web/Target Identifiers": ["login.uris", "String.URL", "overview.url", "overview.title", "String.Title", "urls", "title", "url"],
    "2FA / OTP Indicators": ["login.totp", "TOTP", "String.TOTP", "OTP", "section.otp", "totp"],
    "Vault Organization": ["folderId", "folder", "Group", "details.sections", "section.email", "section.phone", "section.address", "tags", "favorite", "type"],
    "System Metadata": ["updatedAt", "uuid", "Times", "AutoType", "state", "reprompt", "notesPlain", "notes", "String.Notes", "Icon"],
    "Vault Size & Metrics": ["is_high_n_entries", "is_high_total_fields", "is_high_avg_fields_per_entry", "is_high_n_unique_fields"]
}


def load_standardized_data(data_path=DATA_PATH):
    if not os.path.exists(data_path):
        print(f"[INFO] Standardized dataset file not found at {data_path}. Building it now...")
        from build_standardized_dataset import main as build_main
        build_main()

    df = pd.read_csv(data_path)
    print(f"[CategoricalNB] Loaded dataset shape: {df.shape}")
    return df


def prepare_features(df):
    target_col = "pm"
    attr_cols = [c for c in df.columns if c.startswith("attr_")]
    meta_cols = ["n_entries", "total_fields", "avg_fields_per_entry", "n_unique_fields"]
    
    print(f"[INFO] Found {len(attr_cols)} one-hot boolean attribute columns.")

    X_attr = df[attr_cols].astype(int).values

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    naming_encoded = enc.fit_transform(df[["naming_style"]]).astype(int)

    X_num = df[meta_cols].values
    X_num_bin = (X_num > np.median(X_num, axis=0)).astype(int)

    X_all = np.hstack([X_attr, naming_encoded, X_num_bin])
    n_categories = (X_all.max(axis=0) + 1).astype(int).tolist()
    
    clean_attr_names = [c.replace("attr_", "") for c in attr_cols]
    feature_names = clean_attr_names + ["naming_style"] + [f"is_high_{c}" for c in meta_cols]
    y = df[target_col].values

    return X_all, y, feature_names, n_categories


def plot_grouped_feature_importance(importances_mean, feature_names, out_path="cnb_grouped_feature_importance.png"):
    group_scores = {grp: 0.0 for grp in ATTRIBUTE_GROUPS}
    group_scores["Other Attributes"] = 0.0

    for feat_name, imp in zip(feature_names, importances_mean):
        matched = False
        for grp, keywords in ATTRIBUTE_GROUPS.items():
            if feat_name in keywords or any(kw.lower() in feat_name.lower() for kw in keywords):
                group_scores[grp] += max(0, imp)
                matched = True
                break
        if not matched:
            group_scores["Other Attributes"] += max(0, imp)

    # Filter out non-zero groups and sort
    sorted_groups = sorted([(k, v) for k, v in group_scores.items() if v >= 0.0001 or k == "Field Naming Convention"], key=lambda x: x[1])
    
    labels = [g[0] for g in sorted_groups]
    values = [g[1] for g in sorted_groups]

    plt.figure(figsize=(8, 4))
    bars = plt.barh(labels, values, color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.xlabel("Aggregated Feature Importance (Mean Accuracy Drop)", fontsize=10, fontweight="bold")
    plt.title("CategoricalNB: Grouped Vault Attribute Importance", fontsize=11, fontweight="bold")
    
    for bar in bars:
        w = bar.get_width()
        if w > 0.0001:
            plt.text(w + 0.001, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=9)

    plt.xlim(0, max(values) * 1.18 if max(values) > 0 else 1)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[CategoricalNB] Grouped feature importance saved to {out_path}")


def plot_learning_curve(X_all, y_all, n_categories, steps=DEFAULT_STEPS, n_repeats=5, display_max_x=100, out_path="cnb_learning_curve.png"):
    total_n = len(X_all)
    steps = [s for s in steps if s <= total_n]

    print(f"\n=== [CategoricalNB] Stratified Subsampling (Evaluates up to 30,000, Plots up to {display_max_x}) ===")

    accuracies = []
    for step_n in steps:
        scores = []
        for rep in range(n_repeats):
            seed = RANDOM_STATE + rep * 100 + step_n
            
            if step_n == total_n:
                X_sub, y_sub = X_all, y_all
            else:
                X_sub, _, y_sub, _ = train_test_split(
                    X_all, y_all, train_size=step_n, random_state=seed, stratify=y_all
                )

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub, y_sub, test_size=0.40, random_state=seed, stratify=y_sub
            )

            clf = CategoricalNB(min_categories=n_categories, alpha=1.0)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            scores.append(accuracy_score(y_te, preds))
        
        mean_acc = float(np.mean(scores))
        accuracies.append(mean_acc)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, accuracies, marker="o", color="#1f77b4", label="CategoricalNB", linewidth=2.5)
    
    plt.xlabel("Total Number of Vault Export Files (Subsample Size N)", fontsize=11, fontweight="bold")
    plt.ylabel("Testing Accuracy (40% Holdout Set)", fontsize=11, fontweight="bold")
    plt.title(f"CategoricalNB: Learning Curve (Vault Files N = 10 to {display_max_x})", fontsize=12, fontweight="bold")
    
    if display_max_x is not None:
        plot_steps = [s for s in steps if s <= display_max_x]
        plt.xticks(plot_steps)
        plt.xlim(5, display_max_x + 5)

    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[CategoricalNB] Learning curve plot saved to {out_path}")


def plot_confusion_matrix(y_test, preds, labels, out_path="cnb_confusion_matrix.png"):
    cm = confusion_matrix(y_test, preds, labels=labels)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("CategoricalNB Confusion Matrix")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted PM")
    plt.ylabel("Actual PM")
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[CategoricalNB] Confusion matrix saved to {out_path}")


def main():
    df = load_standardized_data()
    X, y, feature_names, n_categories = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )

    print(f"[CategoricalNB] Full Train size: {len(X_train)} (60%), Full Test size: {len(X_test)} (40%)")

    clf = CategoricalNB(min_categories=n_categories, alpha=1.0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"[CategoricalNB] Full Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    labels = sorted(list(set(y)))
    plot_confusion_matrix(y_test, preds, labels, "cnb_confusion_matrix.png")
    
    # Compute permutation importance and generate Grouped Feature Importance Plot
    perm_res = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE)
    plot_grouped_feature_importance(perm_res.importances_mean, feature_names, "cnb_grouped_feature_importance.png")
    
    plot_learning_curve(X, y, n_categories, steps=DEFAULT_STEPS, n_repeats=5, display_max_x=100, out_path="cnb_learning_curve.png")


if __name__ == "__main__":
    main()
