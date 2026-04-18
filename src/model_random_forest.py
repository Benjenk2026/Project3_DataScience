"""
model_random_forest.py
Supervised Learning - Random Forest

Trains a Random Forest classifier on the HIGGS dataset.
Parallelized via n_jobs=-1 for scalability.

Usage:
    python model_random_forest.py
    python model_random_forest.py --data ../data/processed/higgs_cleaned.csv
    python model_random_forest.py --data ../data/processed/higgs_cleaned.csv --n_estimators 200
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# CONFIGURATION
DEFAULT_DATA = Path("../data/processed/higgs_cleaned.csv")
LABEL_COL = "label"
FEATURE_COLS = [f"feature_{i}" for i in range(1, 29)]
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 15


def load_data(path: Path, sample: int | None) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)
        print(f"Using {len(df):,} rows (sampled).")
    else:
        print(f"Loaded {len(df):,} rows.")
    return df


def main(data_path: Path, sample: int | None, n_estimators: int, max_depth: int) -> None:
    df = load_data(data_path, sample)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth}) ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest on HIGGS dataset")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sample", type=int, default=None,
                        help="Random row sample size (useful for large files)")
    parser.add_argument("--n_estimators", type=int, default=DEFAULT_N_ESTIMATORS,
                        help="Number of trees (default: 100)")
    parser.add_argument("--max_depth", type=int, default=DEFAULT_MAX_DEPTH,
                        help="Maximum tree depth (default: 15)")
    args = parser.parse_args()
    main(args.data, args.sample, args.n_estimators, args.max_depth)
