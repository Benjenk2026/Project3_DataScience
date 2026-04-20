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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
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
DEFAULT_CV_FOLDS = 3


def load_data(path: Path, sample: int | None) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)
        print(f"Using {len(df):,} rows (sampled).")
    else:
        print(f"Loaded {len(df):,} rows.")
    return df


def main(data_path: Path, sample: int | None, k: int, cv_folds: int) -> None:
    df = load_data(data_path, sample)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_jobs=-1)),
    ])

    param_grid = {
        "model__n_neighbors": sorted(set([k, 3, 5, 7, 11, 15])),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    print(f"Tuning k-NN with {cv_folds}-fold cross-validation ...")
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")

    y_pred = search.best_estimator_.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-NN on HIGGS dataset")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sample", type=int, default=None,
                        help="Random row sample size (recommended for large files)")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Baseline number of neighbors included in tuning grid")
    parser.add_argument("--cv_folds", type=int, default=DEFAULT_CV_FOLDS,
                        help="Number of cross-validation folds (default: 3)")
    args = parser.parse_args()
    main(args.data, args.sample, args.k, args.cv_folds)
