"""
benchmark_metrics.py
Full evaluation benchmark for all HIGGS classifiers.

Runs every model at 4 sample sizes (50k, 200k, 500k, 1M) and collects:
  - ROC-AUC
  - PR-AUC
  - Accuracy
  - F1-score (weighted)
  - Train time (seconds)
  - Predict time (seconds)

Fixed hyperparameters are used (no GridSearchCV) so the benchmark
completes in a reasonable time. Parameters were chosen based on the
tuning runs in the individual model scripts.

Computational caps to avoid infeasible runtimes:
  - RBF SVM  : capped at 50k  (O(n²) training)
  - k-NN     : capped at 200k (O(n·d) per prediction at inference)

Usage:
    cd src
    python benchmark_metrics.py
    python benchmark_metrics.py --data ../processed/higgs_cleaned.csv
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
DEFAULT_DATA   = BASE_DIR.parent / "processed" / "higgs_cleaned.csv"
OUT_CSV        = BASE_DIR.parent / "processed" / "benchmark_results.csv"
LABEL_COL      = "label"
FEATURE_COLS   = [f"feature_{i}" for i in range(1, 29)]
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
SAMPLE_SIZES   = [50_000, 200_000, 500_000, 1_000_000]

# Computational limits
KNN_MAX_SAMPLE     = 200_000
# RBF SVM libsvm backend is O(n²) in memory. At n=40k training rows the
# kernel-cache alone requires ~12 GB. Cap at 10k for a feasible quality demo;
# all required benchmark sizes (50k+) are marked SKIPPED.
RBF_SVM_MAX_SAMPLE = 10_000


# ── Model factory ─────────────────────────────────────────────────────────────
def make_models() -> dict:
    """
    Return a fresh dict of model_name -> estimator on every call so each
    sample-size run starts with an untrained model.

    Hyperparameters are fixed to values that generalise well on HIGGS based
    on earlier GridSearchCV tuning runs (see individual model scripts).
    """
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "k-NN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=7, weights="distance", n_jobs=-1
            )),
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearSVC(
                C=1.0, max_iter=5000, random_state=RANDOM_STATE
            )),
        ]),
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            # probability=False: avoids Platt-scaling (5-fold internal CV).
            # decision_function scores are used directly for ROC-AUC / PR-AUC,
            # which is a valid and common approach.
            ("model", SVC(
                kernel="rbf", C=10.0, gamma="scale",
                probability=False, random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def get_score(model, X_test: np.ndarray) -> np.ndarray:
    """Return a continuous score for the positive class (for AUC metrics)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    # LinearSVC (and its Pipeline) expose decision_function
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        # decision_function may return shape (n,) or (n, classes)
        return raw if raw.ndim == 1 else raw[:, 1]
    return model.predict(X_test).astype(float)


def evaluate(model, X_train, X_test, y_train, y_test) -> dict:
    """Fit model, predict, and return all metrics."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    y_score = get_score(model, X_test)

    return {
        "accuracy":       accuracy_score(y_test, y_pred),
        "f1":             f1_score(y_test, y_pred, average="weighted"),
        "roc_auc":        roc_auc_score(y_test, y_score),
        "pr_auc":         average_precision_score(y_test, y_score),
        "train_time_s":   round(train_time, 2),
        "predict_time_s": round(predict_time, 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main(data_path: Path) -> None:
    # Resolve CLI-provided relative paths against the current working directory.
    data_path = data_path.expanduser().resolve()

    # Only read enough rows for the largest sample (HIGGS is pre-shuffled).
    # This avoids loading the full 5+ GB file when only 1M rows are needed.
    nrows_needed = max(SAMPLE_SIZES) + 10_000  # small buffer
    print(f"Loading up to {nrows_needed:,} rows from {data_path} ...")
    df_full = pd.read_csv(data_path, nrows=nrows_needed)
    print(f"Loaded: {len(df_full):,} rows\n")

    feature_cols = [c for c in FEATURE_COLS if c in df_full.columns]
    records = []

    # ── RBF SVM quality demo at 10k (only feasible size) ─────────────────────
    print("── RBF SVM quality demo at 10,000 rows (O(n²) makes 50k+ infeasible) ──")
    rbf_demo_n = min(RBF_SVM_MAX_SAMPLE, len(df_full))
    df_rbf = df_full.sample(n=rbf_demo_n, random_state=RANDOM_STATE)
    X_rbf = df_rbf[feature_cols].values
    y_rbf = df_rbf[LABEL_COL].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_rbf, y_rbf, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_rbf
    )
    rbf_model = make_models()["RBF SVM"]
    print(f"  RBF SVM (10k)         ", end="", flush=True)
    try:
        m = evaluate(rbf_model, X_tr, X_te, y_tr, y_te)
        print(
            f"acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
            f"roc={m['roc_auc']:.4f}  pr={m['pr_auc']:.4f}  "
            f"train={m['train_time_s']:.1f}s"
        )
        records.append({"model": "RBF SVM", "sample_size": rbf_demo_n, "note": "quality demo only", **m})
    except Exception as exc:
        print(f"ERROR: {exc}")
    print()

    for n in SAMPLE_SIZES:
        actual_n = min(n, len(df_full))
        print(f"{'='*65}")
        print(f"  Sample size: {actual_n:,}{' (dataset limit)' if actual_n < n else ''}")
        print(f"{'='*65}")

        df = df_full.sample(n=actual_n, random_state=RANDOM_STATE)
        X = df[feature_cols].values
        y = df[LABEL_COL].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y,
        )

        for model_name, model in make_models().items():

            # ── Skip computationally infeasible combos ──────────────────────
            skip_reason = None
            if model_name == "k-NN" and actual_n > KNN_MAX_SAMPLE:
                skip_reason = f"Skipped: n > {KNN_MAX_SAMPLE:,} (O(n·d) prediction)"
            if model_name == "RBF SVM" and actual_n > RBF_SVM_MAX_SAMPLE:
                skip_reason = f"Skipped: n > {RBF_SVM_MAX_SAMPLE:,} (O(n²) training)"

            if skip_reason:
                print(f"  {model_name:<20}  {skip_reason}")
                records.append({
                    "model": model_name, "sample_size": actual_n,
                    "accuracy": None, "f1": None,
                    "roc_auc": None, "pr_auc": None,
                    "train_time_s": None, "predict_time_s": None,
                    "note": skip_reason,
                })
                continue

            # ── Run model ───────────────────────────────────────────────────
            print(f"  {model_name:<20}  ", end="", flush=True)
            try:
                m = evaluate(model, X_train, X_test, y_train, y_test)
                print(
                    f"acc={m['accuracy']:.4f}  "
                    f"f1={m['f1']:.4f}  "
                    f"roc={m['roc_auc']:.4f}  "
                    f"pr={m['pr_auc']:.4f}  "
                    f"train={m['train_time_s']:.1f}s  "
                    f"pred={m['predict_time_s']:.2f}s"
                )
                records.append({
                    "model": model_name, "sample_size": actual_n,
                    "note": "", **m,
                })
            except Exception as exc:
                print(f"ERROR: {exc}")
                records.append({
                    "model": model_name, "sample_size": actual_n,
                    "accuracy": None, "f1": None,
                    "roc_auc": None, "pr_auc": None,
                    "train_time_s": None, "predict_time_s": None,
                    "note": f"Error: {exc}",
                })

        print()

    # ── Save results ──────────────────────────────────────────────────────────
    results = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    print(f"Results saved → {OUT_CSV}\n")

    # ── Pretty summary table ──────────────────────────────────────────────────
    metric_cols = ["roc_auc", "pr_auc", "accuracy", "f1", "train_time_s"]
    display = results[["model", "sample_size"] + metric_cols].copy()
    display["sample_size"] = display["sample_size"].apply(lambda x: f"{int(x):,}")

    float_cols = [c for c in metric_cols if c in display.columns]
    for col in float_cols:
        display[col] = display[col].map(
            lambda v: f"{v:.4f}" if pd.notna(v) else "N/A"
        )

    print("=" * 80)
    print("BENCHMARK SUMMARY — ROC-AUC | PR-AUC | Accuracy | F1 | Train-time(s)")
    print("=" * 80)
    print(display.to_string(index=False))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark all HIGGS models")
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="Path to higgs_cleaned.csv",
    )
    args = parser.parse_args()
    main(args.data)
