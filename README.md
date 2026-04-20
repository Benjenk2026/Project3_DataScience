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
4. Trims whitespace in text/string columns and converts string "nan" values to missing (`NA`).
5. For HIGGS, enforces that `feature_1` through `feature_28` are numeric floats (`float64`).
	- Non-numeric values are coerced to missing (`NaN`) and reported.
6. Drops rows missing critical required fields configured per dataset.
	- For HIGGS, rows missing `label` are dropped.
7. Flags outliers for HIGGS using per-feature IQR fences and adds:
	- `outlier_feature_count` (number of features flagged as outliers in that row)
	- `has_outlier` added column in processed (boolean)
8. Optionally applies dataset-specific normalization (for non-HIGGS files in this script), such as:
	- complaint category normalization
9. Deduplicates when a dataset has a configured ID key.
	- If no key is configured/found, dedup is skipped.
	- For HIGGS in this project, dedup is currently skipped (`id_col = None`).
10. Writes cleaned output to `data/processed/higgs_cleaned.csv`.

---

## Running the Models

All model scripts share these common arguments:

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to the cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Use a random sample of N rows (recommended for large files) | full dataset |
| `--cv_folds N` | Number of CV folds used during hyperparameter tuning | `3` |

All models now follow the same training workflow:

1. Stratified train/test split (80/20).
2. Cross-validated hyperparameter tuning on the training split.
3. Final evaluation on the held-out test split.

Feature scaling is applied where appropriate (Linear SVM, RBF SVM, and k-NN) using a `StandardScaler` inside each model pipeline.

Run all scripts from the `src/` directory:

```bash
cd src
```

---

### Linear SVM — `model_linear_svm.py`

Uses `LinearSVC` (liblinear), efficient for large datasets, with CV tuning over core margin-loss settings.

```bash
python model_linear_svm.py
python model_linear_svm.py --data ../data/processed/higgs_cleaned.csv
python model_linear_svm.py --data ../data/processed/higgs_cleaned.csv --sample 500000 --cv_folds 3
```

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Row sample size | full dataset |
| `--cv_folds N` | CV folds for grid search | `3` |

---

## Benchmark Evaluation

The project includes `benchmark_metrics.py` to compare all models on the cleaned HIGGS dataset using subsamples of 50k, 200k, 500k, and 1M rows.

### Evaluation Metrics

The benchmark reports:

1. ROC-AUC
2. PR-AUC
3. Accuracy
4. F1-score

### Practical Benchmark Notes

To make benchmarking feasible on a large dataset, the script uses a few practical constraints:

1. The script reads only the number of rows needed for the largest benchmark sample instead of loading the full 5.4 GB CSV into memory.
2. `k-NN` is evaluated up to 200k rows and skipped beyond that because prediction cost grows poorly with dataset size.
3. `RBF SVM` is evaluated only in a 10k-row quality demo and skipped for 50k+ because the `libsvm` RBF implementation has quadratic memory and time complexity.
4. The other models run across all benchmark sizes: 50k, 200k, 500k, and 1M.

### Benchmark Output

Benchmark results are saved to `processed/benchmark_results.csv`.

---

## Model Comparison

### Which Models Scale Best?

`XGBoost` scaled best overall. It completed all benchmark sizes with low training time while maintaining the strongest predictive results. Its training time increased from 0.57s at 50k rows to 5.73s at 1M rows, which was much better than `Random Forest` and still delivered stronger metrics than `Linear SVM` or `Decision Tree`.

`Linear SVM` and `Decision Tree` also scaled reasonably well in runtime, but their predictive performance was noticeably weaker. `Random Forest` scaled to 1M rows, but training time grew much faster, reaching 62.23s at 1M rows. `k-NN` and `RBF SVM` scaled poorly and had to be capped or skipped due to computational limits.

### Which Models Perform Best?

`XGBoost` performed best across the benchmark. At 1M rows, it achieved:

1. Accuracy: 0.7379
2. F1-score: 0.7377
3. ROC-AUC: 0.8186
4. PR-AUC: 0.8333

`Random Forest` was the second-best performer across all sample sizes. It consistently produced strong results, but remained slightly behind `XGBoost` in every metric. `Decision Tree` ranked third overall and improved steadily as the sample size increased, but it did not match the ensemble models. `Linear SVM` and `k-NN` showed lower performance, and `RBF SVM` did not show competitive enough results in its 10k-row demo to justify its much higher computational cost.

### Which Models Are Most Interpretable?

`Decision Tree` is the most interpretable model in this project because its decision rules can be directly inspected and explained through feature splits and thresholds.

`Linear SVM` is the next most interpretable because the learned feature weights can be examined, although the explanation is less intuitive than a tree structure. `Random Forest` has moderate interpretability through aggregate feature importance and inspection of individual trees, but the full ensemble is harder to explain clearly. `XGBoost` is less interpretable than a single tree or linear model because it combines many boosted trees. `RBF SVM` is the least interpretable because its nonlinear kernel decision boundary is difficult to explain directly.

### Summary

1. Best scaling model: `XGBoost`
2. Best performing model: `XGBoost`
3. Most interpretable model: `Decision Tree`
4. Best performance-to-scalability tradeoff: `XGBoost`

---

### RBF-Kernel SVM — `model_rbf_svm.py`

Uses `SVC` with an RBF kernel and CV tuning over `C` and `gamma`. Training cost is quadratic — **use `--sample`** for large datasets.

```bash
python model_rbf_svm.py
python model_rbf_svm.py --data ../data/processed/higgs_cleaned.csv --sample 50000 --cv_folds 3
```

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Row sample size (recommended) | full dataset |
| `--cv_folds N` | CV folds for grid search | `3` |

---

### k-Nearest Neighbors — `model_knn.py`

Uses `KNeighborsClassifier` with parallel distance computation and CV tuning over neighborhood settings. **Use `--sample`** for large datasets.

```bash
python model_knn.py
python model_knn.py --data ../data/processed/higgs_cleaned.csv --sample 100000 --k 5 --cv_folds 3
```

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Row sample size (recommended) | full dataset |
| `--k N` | Baseline neighbors included in tuning grid | `5` |
| `--cv_folds N` | CV folds for grid search | `3` |

---

### Decision Tree — `model_decision_tree.py`

Uses `DecisionTreeClassifier` with CV tuning over depth and split/leaf constraints.

```bash
python model_decision_tree.py
python model_decision_tree.py --data ../data/processed/higgs_cleaned.csv --max_depth 10 --cv_folds 3
```

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Row sample size | full dataset |
| `--max_depth N` | Baseline max depth included in tuning grid | `10` |
| `--cv_folds N` | CV folds for grid search | `3` |

---

### Random Forest — `model_random_forest.py`

Uses `RandomForestClassifier` with all CPU cores (`n_jobs=-1`) and CV tuning over tree ensemble settings.

```bash
python model_random_forest.py
python model_random_forest.py --data ../data/processed/higgs_cleaned.csv --n_estimators 200 --max_depth 15 --cv_folds 3
```

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Row sample size | full dataset |
| `--n_estimators N` | Baseline number of trees included in tuning grid | `100` |
| `--max_depth N` | Baseline max depth included in tuning grid | `15` |
| `--cv_folds N` | CV folds for grid search | `3` |

---

### Gradient Boosting (XGBoost) — `model_xgboost.py`

Uses `XGBClassifier` with `tree_method='hist'` for scalability and CV tuning over boosting settings. Supports GPU acceleration.

```bash
python model_xgboost.py
python model_xgboost.py --data ../data/processed/higgs_cleaned.csv --n_estimators 300 --learning_rate 0.05 --cv_folds 3
python model_xgboost.py --data ../data/processed/higgs_cleaned.csv --device cuda --cv_folds 3
```

| Argument | Description | Default |
|---|---|---|
| `--data PATH` | Path to cleaned CSV | `../data/processed/higgs_cleaned.csv` |
| `--sample N` | Row sample size | full dataset |
| `--n_estimators N` | Baseline boosting rounds included in tuning grid | `200` |
| `--max_depth N` | Baseline tree depth included in tuning grid | `6` |
| `--learning_rate F` | Baseline eta included in tuning grid | `0.1` |
| `--device STR` | `cpu` or `cuda` (GPU) | `cpu` |
| `--cv_folds N` | CV folds for grid search | `3` |

---

## Benchmark Evaluation

The project includes `benchmark_metrics.py` to compare all models on the cleaned HIGGS dataset using subsamples of 50k, 200k, 500k, and 1M rows.

### Evaluation Metrics

The benchmark reports:

1. ROC-AUC
2. PR-AUC
3. Accuracy
4. F1-score

### Practical Benchmark Notes

To make benchmarking feasible on a large dataset, the script uses a few practical constraints:

1. The script reads only the number of rows needed for the largest benchmark sample instead of loading the full CSV into memory.
2. `k-NN` is evaluated up to 200k rows and skipped beyond that because prediction cost grows poorly with dataset size.
3. `RBF SVM` is evaluated only in a 10k-row quality demo and skipped for 50k+ because the `libsvm` RBF implementation has quadratic memory and time complexity.
4. The other models run across all benchmark sizes: 50k, 200k, 500k, and 1M.

### Benchmark Output

Benchmark results are saved to `processed/benchmark_results.csv`.

---

## Model Comparison

### Which Models Scale Best?

`XGBoost` scaled best overall. It completed all benchmark sizes with low training time while maintaining the strongest predictive results. Its training time increased from 0.57s at 50k rows to 5.73s at 1M rows, which was much better than `Random Forest` and still delivered stronger metrics than `Linear SVM` or `Decision Tree`.

`Linear SVM` and `Decision Tree` also scaled reasonably well in runtime, but their predictive performance was noticeably weaker. `Random Forest` scaled to 1M rows, but training time grew much faster, reaching 62.23s at 1M rows. `k-NN` and `RBF SVM` scaled poorly and had to be capped or skipped due to computational limits.

### Which Models Perform Best?

`XGBoost` performed best across the benchmark. At 1M rows, it achieved:

1. Accuracy: 0.7379
2. F1-score: 0.7377
3. ROC-AUC: 0.8186
4. PR-AUC: 0.8333

`Random Forest` was the second-best performer across all sample sizes. It consistently produced strong results, but remained slightly behind `XGBoost` in every metric. `Decision Tree` ranked third overall and improved steadily as the sample size increased, but it did not match the ensemble models. `Linear SVM` and `k-NN` showed lower performance, and `RBF SVM` did not show competitive enough results in its 10k-row demo to justify its much higher computational cost.

### Which Models Are Most Interpretable?

`Decision Tree` is the most interpretable model in this project because its decision rules can be directly inspected and explained through feature splits and thresholds.

`Linear SVM` is the next most interpretable because the learned feature weights can be examined, although the explanation is less intuitive than a tree structure. `Random Forest` has moderate interpretability through aggregate feature importance and inspection of individual trees, but the full ensemble is harder to explain clearly. `XGBoost` is less interpretable than a single tree or linear model because it combines many boosted trees. `RBF SVM` is the least interpretable because its nonlinear kernel decision boundary is difficult to explain directly.

### Summary

1. Best scaling model: `XGBoost`
2. Best performing model: `XGBoost`
3. Most interpretable model: `Decision Tree`
4. Best performance-to-scalability tradeoff: `XGBoost`



## For Report 

Based on the benchmark results in benchmark_results.csv, the model that scales best overall is XGBoost. It ran successfully at 50k, 200k, 500k, and 1M, and its training time stayed relatively low: 0.57s, 1.25s, 2.97s, and 5.73s. Linear SVM and Decision Tree also scale reasonably well in terms of runtime, but they do not match XGBoost’s predictive quality. Random Forest scales to 1M too, but its cost grows much faster, reaching 62.23s at 1M. k-NN and RBF SVM do not scale well in this benchmark: k-NN was skipped beyond 200k, and RBF SVM was skipped beyond 10k because of computational limits.

The best-performing model is also XGBoost. It had the strongest results at every benchmark size across all four metrics. At 1M rows, it achieved Accuracy 0.7379, F1 0.7377, ROC-AUC 0.8186, and PR-AUC 0.8333. Random Forest was consistently second best, with strong but slightly lower numbers at every size. Decision Tree improved as sample size increased and was clearly better than Linear SVM and k-NN, but it stayed behind the two ensemble tree models. Linear SVM and k-NN were the weakest among the models that ran across larger samples. RBF SVM also underperformed in its 10k quality-demo run, so there is no evidence here that it would beat the other models even if it were computationally feasible.

For interpretability, Decision Tree is the most interpretable model. You can inspect its splits directly and explain predictions in terms of feature thresholds. Linear SVM is next most interpretable because its linear weights can be examined, although that is less intuitive than a tree. Random Forest is moderately interpretable through feature importance and per-tree inspection, but the full ensemble is harder to explain. XGBoost is less interpretable than a single tree or linear model because it combines many boosted trees, even though feature importance and SHAP-style explanations can help. k-NN is easy to describe procedurally, but not very interpretable as a global model. RBF SVM is the least interpretable because its nonlinear kernel decision boundary is difficult to explain directly.