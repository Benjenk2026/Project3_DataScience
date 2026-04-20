"""
model_linear_svm.py
Supervised Learning - Linear SVM

Trains a Linear Support Vector Machine on the HIGGS dataset.
Scalable to large datasets via LinearSVC (liblinear solver).

Usage:
    python model_linear_svm.py
    python model_linear_svm.py --data ../data/processed/higgs_cleaned.csv
    python model_linear_svm.py --data ../data/processed/higgs_cleaned.csv --sample 100000
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.svm import LinearSVC
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


def main(data_path: Path, sample: int | None, cv_folds: int) -> None:
    df = load_data(data_path, sample)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearSVC(max_iter=5000, random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__loss": ["hinge", "squared_hinge"],
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

    print(f"Tuning Linear SVM with {cv_folds}-fold cross-validation ...")
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")

    y_pred = search.best_estimator_.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear SVM on HIGGS dataset")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sample", type=int, default=None,
                        help="Random row sample size (useful for large files)")
    parser.add_argument("--cv_folds", type=int, default=DEFAULT_CV_FOLDS,
                        help="Number of cross-validation folds (default: 3)")
    args = parser.parse_args()
    main(args.data, args.sample, args.cv_folds)
