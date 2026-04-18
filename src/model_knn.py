"""
model_knn.py
Supervised Learning - k-Nearest Neighbors

Trains a k-NN classifier on the HIGGS dataset.
Scalable note: k-NN can be slow on very large datasets at prediction time.
Use --sample to limit training size if needed.

Usage:
    python model_knn.py
    python model_knn.py --data ../data/processed/higgs_cleaned.csv
    python model_knn.py --data ../data/processed/higgs_cleaned.csv --sample 100000 --k 5
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# CONFIGURATION
DEFAULT_DATA = Path("../data/processed/higgs_cleaned.csv")
LABEL_COL = "label"
FEATURE_COLS = [f"feature_{i}" for i in range(1, 29)]
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_K = 5


def load_data(path: Path, sample: int | None) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)
        print(f"Using {len(df):,} rows (sampled).")
    else:
        print(f"Loaded {len(df):,} rows.")
    return df


def main(data_path: Path, sample: int | None, k: int) -> None:
    df = load_data(data_path, sample)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=k, n_jobs=-1)),
    ])

    print(f"Training k-NN (k={k}) ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-NN on HIGGS dataset")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sample", type=int, default=None,
                        help="Random row sample size (recommended for large files)")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Number of neighbors (default: 5)")
    args = parser.parse_args()
    main(args.data, args.sample, args.k)
