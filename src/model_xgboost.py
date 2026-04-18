"""
model_xgboost.py
Supervised Learning - Gradient Boosting (XGBoost)

Trains an XGBoost classifier on the HIGGS dataset.
Supports GPU acceleration (set --device cuda if available).
Scalable via n_jobs=-1 and tree_method='hist'.

Usage:
    python model_xgboost.py
    python model_xgboost.py --data ../data/processed/higgs_cleaned.csv
    python model_xgboost.py --data ../data/processed/higgs_cleaned.csv --n_estimators 300
    python model_xgboost.py --data ../data/processed/higgs_cleaned.csv --device cuda
"""

import argparse
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# CONFIGURATION
DEFAULT_DATA = Path("../data/processed/higgs_cleaned.csv")
LABEL_COL = "label"
FEATURE_COLS = [f"feature_{i}" for i in range(1, 29)]
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 6
DEFAULT_LEARNING_RATE = 0.1


def load_data(path: Path, sample: int | None) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)
        print(f"Using {len(df):,} rows (sampled).")
    else:
        print(f"Loaded {len(df):,} rows.")
    return df


def main(data_path: Path, sample: int | None, n_estimators: int,
         max_depth: int, learning_rate: float, device: str) -> None:
    df = load_data(data_path, sample)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",         # fast histogram method — scales to large data
        device=device,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )

    print(f"Training XGBoost (n_estimators={n_estimators}, max_depth={max_depth}, "
          f"lr={learning_rate}, device={device}) ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost on HIGGS dataset")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sample", type=int, default=None,
                        help="Random row sample size (useful for large files)")
    parser.add_argument("--n_estimators", type=int, default=DEFAULT_N_ESTIMATORS,
                        help="Number of boosting rounds (default: 200)")
    parser.add_argument("--max_depth", type=int, default=DEFAULT_MAX_DEPTH,
                        help="Maximum tree depth (default: 6)")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help="Learning rate / eta (default: 0.1)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use for training (default: cpu)")
    args = parser.parse_args()
    main(args.data, args.sample, args.n_estimators, args.max_depth,
         args.learning_rate, args.device)
