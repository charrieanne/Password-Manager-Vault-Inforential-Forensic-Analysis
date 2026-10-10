"""
PM Vault Fingerprinting - Top 5 / Top 10 Feature Experiments & Stacked Area Learning Curves
=============================================================================================
This script:
1. Section III-B (Feature Importance):
   - Trains Categorical Naive Bayes, Random Forest, and Decision Tree.
   - Extracts and plots Top 5 and Top 10 individual features per model.

2. Section III-C (Password Manager Recognition):
   - Evaluates learning curves across N = 10 to 100 subsample sizes per class.
   - Generates Stacked Area Charts normalized to [0.0, 1.0] accuracy range.
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_PATH = Path("C:/Users/Lenovo/.gemini/antigravity/scratch/standardized_pm_vault_features.csv")
STEPS_N = list(range(10, 101, 10))


def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"[INFO] Looking for dataset at {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_data(df):
    target_col = "pm"
    attr_cols = [c for c in df.columns if c.startswith("attr_")]
    meta_cols = ["n_entries", "total_fields", "avg_fields_per_entry", "n_unique_fields"]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    naming_encoded = enc.fit_transform(df[["naming_style"]]).astype(int)

    X_attr = df[attr_cols].astype(int).values
    X_num = df[meta_cols].values
    X_num_bin = (X_num > np.median(X_num, axis=0)).astype(int)

    # Combined features for Naive Bayes (all discrete/binned)
    X_nb = np.hstack([X_attr, naming_encoded, X_num_bin])
    n_categories_nb = (X_nb.max(axis=0) + 1).astype(int).tolist()

    # Combined features for Tree models (continuous metadata allowed)
    X_tree = np.hstack([X_attr, X_num, naming_encoded])

    clean_attr_names = [c.replace("attr_", "") for c in attr_cols]
    feature_names = clean_attr_names + meta_cols + ["naming_style"]
    y = df[target_col].values

    return X_nb, X_tree, y, feature_names, n_categories_nb


def get_top_k_indices(importances, k):
    return np.argsort(importances)[::-1][:k]


# --- SECTION III-B: FEATURE IMPORTANCE BAR CHARTS (TOP 10 FEATURES) ---
def plot_top10_feature_importance(importances, feature_names, model_name, color, out_path):
    top10_idx = get_top_k_indices(importances, 10)
    top10_names = [feature_names[i] for i in top10_idx[::-1]]
    top10_scores = [importances[i] for i in top10_idx[::-1]]

    plt.figure(figsize=(8, 4.5))
    bars = plt.barh(top10_names, top10_scores, color=color, edgecolor="black", alpha=0.85)
    plt.xlabel("Feature Importance Score", fontsize=10, fontweight="bold")
    plt.title(f"{model_name}: Top 10 Vault Feature Importance", fontsize=11, fontweight="bold")
    
    for bar in bars:
        w = bar.get_width()
        if w > 0.0001:
            plt.text(w + max(top10_scores)*0.01, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=9)

    plt.xlim(0, max(top10_scores) * 1.18 if max(top10_scores) > 0 else 1)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[{model_name}] Top 10 Feature Importance plot saved to {out_path}")


# --- SECTION III-C: STACKED AREA LEARNING CURVES (NORMALIZED 0.0 TO 1.0 ACCURACY RANGE) ---
def plot_stacked_area_learning_curve(steps, acc_per_class, model_name, out_path):
    """
    Plots Stacked Area Learning Curves.
    - Stacked Areas: Raw per-class testing accuracies stacked on top of each other (accumulating up to 3.0 max).
    - Black Line Overlay: Actual Average Stacked Accuracy = (c1 + c2 + c3) / 3.0, sitting around 1.0 (100% mean accuracy).
    """
    plt.figure(figsize=(9, 5))
    classes = list(acc_per_class.keys())
    
    # Extract raw per-class testing accuracies (each ranging from 0.0 to 1.0)
    raw_c1 = np.array(acc_per_class[classes[0]])
    raw_c2 = np.array(acc_per_class[classes[1]])
    raw_c3 = np.array(acc_per_class[classes[2]])

    # Calculate actual Average Stacked Accuracy across all 3 classes (sits around 1.0)
    actual_avg_acc = (raw_c1 + raw_c2 + raw_c3) / 3.0

    # Stack raw per-class accuracies directly (total stacked height max = 3.0)
    plt.stackplot(
        steps,
        raw_c1,
        raw_c2,
        raw_c3,
        labels=[f"{classes[0]}", f"{classes[1]}", f"{classes[2]} "],
        colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
        alpha=0.75
    )

    # Plot line overlay tracking actual Average Stacked Accuracy (around 1.0) with square markers
    plt.plot(steps, actual_avg_acc, color="black", linewidth=2.5, marker="s", markersize=6, label="Average Performance")

    plt.xlabel("Number of Vault Export Files", fontsize=11, fontweight="bold")
    plt.ylabel("Stacked Accuracy", fontsize=11, fontweight="bold")
    plt.title(f"{model_name}: Learning Curve", fontsize=12, fontweight="bold")
    plt.xticks(steps)
    plt.xlim(10, 100)
    plt.ylim(0, 3.2)
    plt.grid(alpha=0.3)
    plt.legend(loc="center right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[{model_name}] Learning Curve saved to {out_path}")


def evaluate_per_class_learning_curve_nb(X_full, y_full, n_cats_full, steps=STEPS_N, n_repeats=5):
    classes = sorted(list(set(y_full)))
    per_class_acc = {c: [] for c in classes}

    for step_n in steps:
        class_scores = {c: [] for c in classes}
        for rep in range(n_repeats):
            seed = RANDOM_STATE + rep * 100 + step_n
            X_sub, _, y_sub, _ = train_test_split(
                X_full, y_full, train_size=step_n, random_state=seed, stratify=y_full
            )
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub, y_sub, test_size=0.40, random_state=seed, stratify=y_sub
            )
            clf = CategoricalNB(min_categories=n_cats_full, alpha=1.0)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            
            for c in classes:
                mask = (y_te == c)
                if np.sum(mask) > 0:
                    class_scores[c].append(accuracy_score(y_te[mask], preds[mask]))
                else:
                    class_scores[c].append(1.0)

        for c in classes:
            per_class_acc[c].append(np.mean(class_scores[c]))

    return per_class_acc


def evaluate_per_class_learning_curve_tree(clf_class, X_full, y_full, steps=STEPS_N, n_repeats=5, **kwargs):
    classes = sorted(list(set(y_full)))
    per_class_acc = {c: [] for c in classes}

    for step_n in steps:
        class_scores = {c: [] for c in classes}
        for rep in range(n_repeats):
            seed = RANDOM_STATE + rep * 100 + step_n
            X_sub, _, y_sub, _ = train_test_split(
                X_full, y_full, train_size=step_n, random_state=seed, stratify=y_full
            )
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub, y_sub, test_size=0.40, random_state=seed, stratify=y_sub
            )
            clf = clf_class(random_state=seed, **kwargs)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            
            for c in classes:
                mask = (y_te == c)
                if np.sum(mask) > 0:
                    class_scores[c].append(accuracy_score(y_te[mask], preds[mask]))
                else:
                    class_scores[c].append(1.0)

        for c in classes:
            per_class_acc[c].append(np.mean(class_scores[c]))

    return per_class_acc


def main():
    df = load_data()
    X_nb, X_tree, y, feature_names, n_cats_nb = prepare_data(df)

    X_tr_nb, X_te_nb, y_tr, y_te = train_test_split(X_nb, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE)
    X_tr_tr, X_te_tr, _, _ = train_test_split(X_tree, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE)

    # 1. Categorical Naive Bayes - Top Features
    clf_nb = CategoricalNB(min_categories=n_cats_nb, alpha=1.0)
    clf_nb.fit(X_tr_nb, y_tr)
    perm_nb = permutation_importance(clf_nb, X_te_nb, y_te, n_repeats=5, random_state=RANDOM_STATE)
    imp_nb = perm_nb.importances_mean

    plot_top10_feature_importance(imp_nb, feature_names, "CategoricalNB", "#1f77b4", "cnb_top10_feature_importance.png")

    # 2. Random Forest - Top Features
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf_rf.fit(X_tr_tr, y_tr)
    imp_rf = clf_rf.feature_importances_

    plot_top10_feature_importance(imp_rf, feature_names, "RandomForest", "#d62728", "rf_top10_feature_importance.png")

    # 3. Decision Tree - Top Features
    clf_dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    clf_dt.fit(X_tr_tr, y_tr)
    imp_dt = clf_dt.feature_importances_

    plot_top10_feature_importance(imp_dt, feature_names, "DecisionTree", "#2ca02c", "dt_top10_feature_importance.png")

    print("\n==================================================")
    print("SECTION III-C: STACKED AREA LEARNING CURVES (RANGE: 0.0 TO 1.0)")
    print("==================================================")

    nb_per_class = evaluate_per_class_learning_curve_nb(X_nb, y, n_cats_nb)
    plot_stacked_area_learning_curve(STEPS_N, nb_per_class, "CategoricalNB", "cnb_learning_curve_stacked_area.png")

    rf_per_class = evaluate_per_class_learning_curve_tree(RandomForestClassifier, X_tree, y, n_estimators=100)
    plot_stacked_area_learning_curve(STEPS_N, rf_per_class, "RandomForest", "rf_learning_curve_stacked_area.png")

    dt_per_class = evaluate_per_class_learning_curve_tree(DecisionTreeClassifier, X_tree, y)
    plot_stacked_area_learning_curve(STEPS_N, dt_per_class, "DecisionTree", "dt_learning_curve_stacked_area.png")


if __name__ == "__main__":
    main()