"""
model_decision_tree.py
Supervised Learning - Decision Tree

Trains a Decision Tree classifier on the HIGGS dataset.
Scales well to large datasets; max_depth limits overfitting and memory use.

Usage:
    python model_decision_tree.py
    python model_decision_tree.py --data ../data/processed/higgs_cleaned.csv
    python model_decision_tree.py --data ../data/processed/higgs_cleaned.csv --max_depth 10
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# CONFIGURATION
DEFAULT_DATA = Path("../data/processed/higgs_cleaned.csv")
LABEL_COL = "label"
FEATURE_COLS = [f"feature_{i}" for i in range(1, 29)]
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_MAX_DEPTH = 10


def load_data(path: Path, sample: int | None) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)
        print(f"Using {len(df):,} rows (sampled).")
    else:
        print(f"Loaded {len(df):,} rows.")
    return df


def main(data_path: Path, sample: int | None, max_depth: int) -> None:
    df = load_data(data_path, sample)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)

    print(f"Training Decision Tree (max_depth={max_depth}) ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision Tree on HIGGS dataset")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sample", type=int, default=None,
                        help="Random row sample size (useful for large files)")
    parser.add_argument("--max_depth", type=int, default=DEFAULT_MAX_DEPTH,
                        help="Maximum tree depth (default: 10)")
    args = parser.parse_args()
    main(args.data, args.sample, args.max_depth)
