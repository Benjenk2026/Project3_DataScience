"""
pca_comparison.py
Project 3 - Section 3b Part A

Trains all classifiers on raw features and PCA-reduced features (10 components)
and compares accuracy, ROC-AUC, training time, and inference time.

Usage:
    python src/pca_comparison.py
    python src/pca_comparison.py --rows 500000
"""

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH    = Path(__file__).resolve().parent.parent / "data" / "processed" / "higgs_cleaned.csv"
OUT_DIR      = Path(__file__).resolve().parent.parent / "Analysis_and_Findings" / "pca_comparison"
LABEL_COL    = "label"
FEATURE_COLS = [f"feature_{i}" for i in range(1, 29)]
RANDOM_STATE = 42

# RBF SVM and k-NN don't scale well at large sizes so we cap them
RBF_MAX = 10_000
KNN_MAX = 200_000


def get_models():
    # Using same hyperparameters as benchmark_metrics.py
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=RANDOM_STATE),
        "k-NN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1))]),
        "Linear SVM": Pipeline([("scaler", StandardScaler()), ("model", LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE))]),
        "RBF SVM": Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", C=10.0, gamma="scale", probability=False, random_state=RANDOM_STATE))]),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0),
    }


def get_score(model, X_test):
    # Get probability scores for AUC metrics - LinearSVC uses decision_function instead
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return scores if scores.ndim == 1 else scores[:, 1]
    return model.predict(X_test).astype(float)


def run_model(name, model, X_train, X_test, y_train, y_test, n_rows, feature_set):
    # Skip models that are too slow at this sample size
    if name == "RBF SVM" and n_rows > RBF_MAX:
        print(f"  {name:<20} SKIPPED (too slow at n={n_rows:,})")
        return {"model": name, "feature_set": feature_set, "sample_size": n_rows,
                "accuracy": None, "roc_auc": None, "f1": None, "pr_auc": None,
                "train_time_s": None, "predict_time_s": None, "note": "Skipped"}
    if name == "k-NN" and n_rows > KNN_MAX:
        print(f"  {name:<20} SKIPPED (too slow at n={n_rows:,})")
        return {"model": name, "feature_set": feature_set, "sample_size": n_rows,
                "accuracy": None, "roc_auc": None, "f1": None, "pr_auc": None,
                "train_time_s": None, "predict_time_s": None, "note": "Skipped"}

    print(f"  {name:<20} ", end="", flush=True)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - t0

    y_score = get_score(model, X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    roc = roc_auc_score(y_test, y_score)
    pr  = average_precision_score(y_test, y_score)

    print(f"acc={acc:.4f}  roc={roc:.4f}  train={train_time:.2f}s  pred={pred_time:.2f}s")
    return {"model": name, "feature_set": feature_set, "sample_size": n_rows,
            "accuracy": acc, "roc_auc": roc, "f1": f1, "pr_auc": pr,
            "train_time_s": round(train_time, 2), "predict_time_s": round(pred_time, 2), "note": ""}


def plot_results(df, out_dir):
    models = df["model"].unique()
    x = np.arange(len(models))
    width = 0.35
    colors = {"Raw 28D": "#2b8cbe", "PCA 10D": "#e34a33"}

    metrics = [
        ("accuracy",       "Accuracy (higher is better)"),
        ("roc_auc",        "ROC-AUC (higher is better)"),
        ("train_time_s",   "Training Time in seconds (lower is better)"),
        ("predict_time_s", "Inference Time in seconds (lower is better)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        for j, fset in enumerate(["Raw 28D", "PCA 10D"]):
            vals = []
            for m in models:
                row = df[(df["model"] == m) & (df["feature_set"] == fset)]
                val = float(row[metric].values[0]) if len(row) and pd.notna(row[metric].values[0]) else 0
                vals.append(val)
            bars = ax.bar(x + j * width, vals, width, label=fset, color=colors[fset], edgecolor="white")
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    n = int(df["sample_size"].iloc[0])
    fig.suptitle(f"Raw Features vs PCA 10D — Model Comparison\n(sample size: {n:,})", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = out_dir / "pca_vs_raw_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {out_path}")


def main(rows, n_components):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {rows:,} rows ...")
    df = pd.read_csv(DATA_PATH, nrows=rows + 5000)
    df = df.sample(n=min(rows, len(df)), random_state=RANDOM_STATE)
    print(f"Using {len(df):,} rows\n")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df[LABEL_COL].values
    n_rows = len(df)

    # Apply StandardScaler then PCA to get reduced features
    print(f"Applying StandardScaler + PCA ({n_components} components) ...")
    pca_pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE))])
    X_pca = pca_pipe.fit_transform(X)
    explained = pca_pipe.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"PCA {n_components} components explain {explained*100:.1f}% of variance\n")

    # Same train/test split used for both feature sets so results are comparable
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_pca_tr, X_pca_te, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    records = []

    print("--- Raw 28D ---")
    for name, model in get_models().items():
        records.append(run_model(name, model, X_tr, X_te, y_tr, y_te, n_rows, "Raw 28D"))

    print(f"\n--- PCA {n_components}D ---")
    for name, model in get_models().items():
        records.append(run_model(name, model, X_pca_tr, X_pca_te, y_tr, y_te, n_rows, f"PCA {n_components}D"))

    results = pd.DataFrame(records)
    csv_path = OUT_DIR / "pca_vs_raw_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    print("\n--- Summary ---")
    print(results[["model", "feature_set", "accuracy", "roc_auc", "train_time_s", "predict_time_s"]].to_string(index=False))

    plot_results(results, OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare raw vs PCA features for classification")
    parser.add_argument("--rows", type=int, default=200_000, help="Number of rows to use")
    parser.add_argument("--components", type=int, default=10, help="Number of PCA components")
    args = parser.parse_args()
    main(args.rows, args.components)