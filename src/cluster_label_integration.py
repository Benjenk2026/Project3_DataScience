"""
cluster_label_integration.py
Project 3 - Section 3b Part B

Adds the k-Means cluster ID from Project 2 as an extra feature and retrains
all classifiers. Compares against raw features only to see if cluster
membership improves classification performance.

Usage:
    python src/cluster_label_integration.py
    python src/cluster_label_integration.py --rows 200000
"""

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

CLUSTERED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "higgs_clustered.csv"
OUT_DIR        = Path(__file__).resolve().parent.parent / "Analysis_and_Findings" / "cluster_integration"
LABEL_COL      = "label"
CLUSTER_COL    = "cluster"
FEATURE_COLS   = [f"feature_{i}" for i in range(1, 29)]
RANDOM_STATE   = 42

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

    print(f"acc={acc:.4f}  f1={f1:.4f}  roc={roc:.4f}  pr={pr:.4f}  train={train_time:.2f}s  pred={pred_time:.2f}s")
    return {"model": name, "feature_set": feature_set, "sample_size": n_rows,
            "accuracy": acc, "roc_auc": roc, "f1": f1, "pr_auc": pr,
            "train_time_s": round(train_time, 2), "predict_time_s": round(pred_time, 2), "note": ""}


def plot_results(df, out_dir):
    models = df["model"].unique()
    x = np.arange(len(models))
    width = 0.35
    colors = {"Raw 28D": "#2b8cbe", "Raw 28D + Cluster": "#e34a33"}

    metrics = [
        ("accuracy", "Accuracy (higher is better)"),
        ("roc_auc",  "ROC-AUC (higher is better)"),
        ("f1",       "F1-Score (higher is better)"),
        ("pr_auc",   "PR-AUC (higher is better)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        for j, fset in enumerate(["Raw 28D", "Raw 28D + Cluster"]):
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
    fig.suptitle(f"Raw Features vs Raw + Cluster Label — Model Comparison\n(sample size: {n:,})", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = out_dir / "cluster_integration_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {out_path}")


def main(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {rows:,} rows from {CLUSTERED_PATH} ...")
    df = pd.read_csv(CLUSTERED_PATH, nrows=rows + 5000)
    df = df.sample(n=min(rows, len(df)), random_state=RANDOM_STATE)
    print(f"Using {len(df):,} rows\n")

    if CLUSTER_COL not in df.columns:
        raise ValueError(f"Column '{CLUSTER_COL}' not found. Make sure higgs_clustered.csv has a cluster column.")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    y      = df[LABEL_COL].values
    n_rows = len(df)

    # Raw 28 features only
    X_raw = df[feature_cols].values

    # Raw 28 features + cluster ID appended as a 29th column
    X_cluster = np.column_stack([X_raw, df[CLUSTER_COL].values])

    print(f"Raw feature shape:             {X_raw.shape}")
    print(f"Raw + cluster feature shape:   {X_cluster.shape}\n")

    # Same split for both so results are directly comparable
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_cl_tr, X_cl_te, _, _ = train_test_split(X_cluster, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    records = []

    print("--- Raw 28D ---")
    for name, model in get_models().items():
        records.append(run_model(name, model, X_tr, X_te, y_tr, y_te, n_rows, "Raw 28D"))

    print("\n--- Raw 28D + Cluster Label ---")
    for name, model in get_models().items():
        records.append(run_model(name, model, X_cl_tr, X_cl_te, y_tr, y_te, n_rows, "Raw 28D + Cluster"))

    results = pd.DataFrame(records)
    csv_path = OUT_DIR / "cluster_integration_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    print("\n--- Summary ---")
    print(results[["model", "feature_set", "accuracy", "roc_auc", "f1", "pr_auc"]].to_string(index=False))

    plot_results(results, OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate impact of adding cluster label as a feature")
    parser.add_argument("--rows", type=int, default=200_000, help="Number of rows to use")
    args = parser.parse_args()
    main(args.rows)