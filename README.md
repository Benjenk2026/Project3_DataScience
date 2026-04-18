# Project3_DataScience
Data Science project using supervised learning on the UCI HIGGS dataset.

## Contents
* [Go to Project Structure](#project-structure)
* [Go to Setup](#setup)
* [Go to Data Preparation](#data-preparation)
* [Go to Linear SVM](#linear-svm----model_linear_svmpy)
* [Go to RBF-Kernel SVM](#rbf-kernel-svm----model_rbf_svmpy)
* [Go to k-Nearest Neighbors](#k-nearest-neighbors----model_knnpy)
* [Go to Decision Tree](#decision-tree----model_decision_treepy)
* [Go to Random Forest](#random-forest----model_random_forestpy)
* [Go to Gradient Boosting (XGBoost)](#gradient-boosting-xgboost----model_xgboostpy)

---

## Project Structure

```
data/
  higgs/          # Raw HIGGS.csv (download separately)
  processed/      # Cleaned output from cleaning.py
src/
  cleaning.py             # Data cleaning pipeline
  eda.py                  # Exploratory data analysis
  model_linear_svm.py     # Linear SVM
  model_rbf_svm.py        # RBF-Kernel SVM
  model_knn.py            # k-Nearest Neighbors
  model_decision_tree.py  # Decision Tree
  model_random_forest.py  # Random Forest
  model_xgboost.py        # Gradient Boosting (XGBoost)
```

---

## Setup

Install dependencies:

```bash
pip install pandas scikit-learn xgboost matplotlib seaborn
```

---

## Data Preparation

### 1. Clean the raw data (cleaning.py)

Run `cleaning.py` before any model script. Use `--chunked` for the full 11M-row HIGGS file.

```bash
cd src
python cleaning.py --file higgs --chunked
```

Output: `data/processed/higgs_cleaned.csv`


### What it does to the data
1. Loads source data from CSV or JSON.
2. For HIGGS specifically, assigns explicit column names:
	- `label`
	- `feature_1` through `feature_28`
3. Standardizes all column names to `snake_case`.
4. Applies known alias mapping for location fields (for datasets that include them), such as:
	- `lat` -> `latitude`
	- `long` / `lon` -> `longitude`
	- `zip` / `postal_code` -> `zipcode`
5. Trims whitespace in text/string columns and converts string "nan" values to missing (`NA`).
6. For HIGGS, enforces that `feature_1` through `feature_28` are numeric floats (`float64`).
	- Non-numeric values are coerced to missing (`NaN`) and reported.
7. Drops rows missing critical required fields configured per dataset.
	- For HIGGS, rows missing `label` are dropped.
8. Flags outliers for HIGGS using per-feature IQR fences and adds:
	- `outlier_feature_count` (number of features flagged as outliers in that row)
	- `has_outlier` added column in processed (boolean)
9. Optionally applies dataset-specific normalization (for non-HIGGS files in this script), such as:
	- complaint category normalization
10. Deduplicates when a dataset has a configured ID key.
	- If no key is configured/found, dedup is skipped.
	- For HIGGS in this project, dedup is currently skipped (`id_col = None`).
11. Writes cleaned output to `data/processed/higgs_cleaned.csv`.

---

## Running the Models

All model scripts share these common arguments:

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to the cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Use a random sample of N rows (recommended for large files) | full dataset |

Run all scripts from the `src/` directory:

```bash
cd src
```

---

### Linear SVM — `model_linear_svm.py`

Uses `LinearSVC` (liblinear), efficient for large datasets.

```bash
python model_linear_svm.py
python model_linear_svm.py --data ../data/processed/higgs_cleaned.csv
python model_linear_svm.py --data ../data/processed/higgs_cleaned.csv --sample 500000
```

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Row sample size | full dataset |

---

### RBF-Kernel SVM — `model_rbf_svm.py`

Uses `SVC` with an RBF kernel. Training cost is quadratic — **use `--sample`** for large datasets.

```bash
python model_rbf_svm.py
python model_rbf_svm.py --data ../data/processed/higgs_cleaned.csv --sample 50000
```

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Row sample size (recommended) | full dataset |

---

### k-Nearest Neighbors — `model_knn.py`

Uses `KNeighborsClassifier` with parallel distance computation. **Use `--sample`** for large datasets.

```bash
python model_knn.py
python model_knn.py --data ../data/processed/higgs_cleaned.csv --sample 100000 --k 5
```

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Row sample size (recommended) | full dataset |
| `--k N` | Number of neighbors | `5` |

---

### Decision Tree — `model_decision_tree.py`

Uses `DecisionTreeClassifier`. `--max_depth` controls overfitting and memory usage.

```bash
python model_decision_tree.py
python model_decision_tree.py --data ../data/processed/higgs_cleaned.csv --max_depth 10
```

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Row sample size | full dataset |
| `--max_depth N` | Maximum depth of the tree | `10` |

---

### Random Forest — `model_random_forest.py`

Uses `RandomForestClassifier` with all CPU cores (`n_jobs=-1`).

```bash
python model_random_forest.py
python model_random_forest.py --data ../data/processed/higgs_cleaned.csv --n_estimators 200 --max_depth 15
```

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Row sample size | full dataset |
| `--n_estimators N` | Number of trees | `100` |
| `--max_depth N` | Maximum depth per tree | `15` |

---

### Gradient Boosting (XGBoost) — `model_xgboost.py`

Uses `XGBClassifier` with `tree_method='hist'` for scalability. Supports GPU acceleration.

```bash
python model_xgboost.py
python model_xgboost.py --data ../data/processed/higgs_cleaned.csv --n_estimators 300 --learning_rate 0.05
python model_xgboost.py --data ../data/processed/higgs_cleaned.csv --device cuda
```

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Row sample size | full dataset |
| `--n_estimators N` | Number of boosting rounds | `200` |
| `--max_depth N` | Maximum tree depth | `6` |
| `--learning_rate F` | Step size shrinkage (eta) | `0.1` |
| `--device STR` | `cpu` or `cuda` (GPU) | `cpu` |

