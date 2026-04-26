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
* [Go to PCA Comparison Analysis](#pca-comparison-analysis----pca_comparisonpy)
* [Go to Cluster Label Integration Analysis](#cluster-label-integration-analysis----cluster_label_integrationpy)

---

## Project Structure

```
Analysis_and_Findings/  # Results from analysis scripts
  cluster_integration/  # Output from cluster_label_integration.py
  pca_comparison/       # Output from pca_comparison.py
data/
  higgs/          # Raw HIGGS.csv (download separately)
  processed/      # Cleaned output from cleaning.py
src/
  cleaning.py               # Data cleaning pipeline
  eda.py                    # Exploratory data analysis
  pca_comparison.py         # PCA vs Raw Features analysis
  cluster_label_integration.py  # Cluster membership feature analysis
  model_linear_svm.py       # Linear SVM
  model_rbf_svm.py          # RBF-Kernel SVM
  model_knn.py              # k-Nearest Neighbors
  model_decision_tree.py    # Decision Tree
  model_random_forest.py    # Random Forest
  model_xgboost.py          # Gradient Boosting (XGBoost)
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


---

## PCA Comparison Analysis — `pca_comparison.py`

Analyzes the impact of dimensionality reduction on model performance by comparing classifiers trained on raw 28-dimensional features versus PCA-reduced features (10 principal components).

### Purpose

Evaluate whether PCA dimensionality reduction:
- Speeds up training and inference across all classifiers
- Preserves or improves classification performance
- Provides meaningful insights into feature redundancy

### Usage

```bash
python src/pca_comparison.py
python src/pca_comparison.py --rows 500000
python src/pca_comparison.py --rows 200000 
```

| Argument | Description | Default |
|---|---|---|
| `--rows N` | Number of rows to sample for analysis | full dataset |
| `--components N` | Number of PCA components | `10` |

### What It Does

1. Loads cleaned HIGGS data
2. Applies PCA to reduce 28 features to 10 principal components
3. Trains all six classifiers on both raw and PCA-reduced feature sets using fixed benchmark hyperparameters
4. Records for each model:
   - Accuracy, ROC-AUC, PR-AUC, F1-score on both feature sets
   - Training time
   - Inference time
5. Generates comparison visualizations and results CSV

### Output

Results are saved to `Analysis_and_Findings/pca_comparison/pca_vs_raw_results.csv` with metrics for each model and feature configuration.

---

## Cluster Label Integration Analysis — `cluster_label_integration.py`

Evaluates whether k-Means cluster membership (from Project 2) acts as a useful supplementary feature for improving classification performance.

### Purpose

Test if adding cluster IDs as an additional feature helps classifiers by:
- Capturing high-level data groupings missed by individual features
- Providing non-linear feature engineering through unsupervised clustering
- Improving model generalization across cluster boundaries

### Usage

```bash
python src/cluster_label_integration.py
python src/cluster_label_integration.py --rows 200000
python src/cluster_label_integration.py --rows 500000
```

To run the full benchmark set and preserve each output with a size suffix, use:

```bash
python src/run_cluster_integration_batch.py
python src/run_cluster_integration_batch.py --data processed/higgs_clustered.csv
python src/run_cluster_integration_batch.py --sizes 50000 200000 500000 1000000
```

| Argument | Description | Default |
|---|---|---|
| `--rows N` | Number of rows to sample for analysis | full dataset |

### What It Does

1. Loads clustered HIGGS data (with pre-computed cluster assignments from Project 2)
2. Trains all six classifiers on:
   - Raw features only (baseline)
   - Raw features + cluster ID as an additional feature
  - Uses fixed benchmark hyperparameters rather than per-run CV tuning
3. Records for each model:
   - Accuracy, ROC-AUC, PR-AUC, F1-score for both configurations
   - Improvement delta between cluster-augmented and baseline
4. Generates comparison visualizations and results CSV

### Output

Results are saved to `Analysis_and_Findings/cluster_integration/cluster_integration_results.csv` with metrics comparing models with and without cluster membership as a feature.

The batch helper additionally writes size-tagged files such as `cluster_integration_results_50.csv` and `cluster_integration_comparison_50.png`.


## K-Means Clustering Pipeline (k-means.py)

The script `src/k-means.py` runs clustering on the cleaned HIGGS dataset using all 28 feature columns (`feature_1` to `feature_28`).

### Main goals
- Cluster events using the full 28-dimensional feature space.
- Support both `KMeans` and `MiniBatchKMeans`.
- Allow limiting the number of rows used for faster experiments.
- Benchmark runtime vs dataset size and save a plot.

### Default behavior
Running with no arguments:

`python src/k-means.py`

will:
- Read `data/processed/higgs_cleaned.csv`
- Validate required feature columns
- Convert features to numeric and drop invalid rows
- Standardize all 28 features
- Fit K-Means (`k=2` by default)
- Save output with cluster labels to `data/processed/higgs_clustered.csv`

### Useful options
- `--k`: number of clusters (default `2`)
- `--algorithm`: `kmeans` or `minibatch` (default `kmeans`)
- `--batch-size`: mini-batch size when using `minibatch` (default `10000`)
- `--rows`: limit the number of rows loaded from input
- `--input`: input CSV path (default `data/processed/ higgs_cleaned.csv`)
- `--output`: output CSV path (default `data/processed/higgs_clustered.csv`)
- `--random-state`: random seed (default `42`)

### Example: MiniBatch on a subset

`python src/k-means.py --algorithm minibatch --rows 500000 --k 2`


## Step-by-Step Run Order

Use this order if you want to reproduce the project from raw data to final comparisons.

### 1. Install dependencies

From the project root:

```bash
pip install pandas scikit-learn xgboost matplotlib seaborn
```

### 2. Put the raw HIGGS file in place

Download the UCI HIGGS dataset and place the raw file where `cleaning.py` expects it.

Expected path:

```text
data/higgs/HIGGS.csv
```

### 3. Clean the raw dataset

Run the cleaning pipeline before any modeling or analysis script.

```bash
cd src
python cleaning.py --file higgs --chunked
```

This creates `data/processed/higgs_cleaned.csv`.

### 4. Optionally inspect the cleaned data

If you want a quick exploratory pass before training models:

```bash
python eda.py
```

### 5. Run the individual supervised models

These scripts perform train/test split, CV tuning, and final held-out evaluation.

```bash
python model_linear_svm.py --data ../data/processed/higgs_cleaned.csv --sample 500000 --cv_folds 3
python model_rbf_svm.py --data ../data/processed/higgs_cleaned.csv --sample 50000 --cv_folds 3
python model_knn.py --data ../data/processed/higgs_cleaned.csv --sample 100000 --k 5 --cv_folds 3
python model_decision_tree.py --data ../data/processed/higgs_cleaned.csv --max_depth 10 --cv_folds 3
python model_random_forest.py --data ../data/processed/higgs_cleaned.csv --n_estimators 200 --max_depth 15 --cv_folds 3
python model_xgboost.py --data ../data/processed/higgs_cleaned.csv --n_estimators 300 --learning_rate 0.05 --cv_folds 3
```

Use smaller `--sample` values while testing, then larger values for final runs.

### 6. Run the all-model benchmark

This compares the main classifiers at 50k, 200k, 500k, and 1M rows using fixed hyperparameters.

```bash
python benchmark_metrics.py --data ../data/processed/higgs_cleaned.csv
```

This writes benchmark results to `processed/benchmark_results.csv`.

### 7. Run the PCA comparison analysis

After cleaning is complete, compare raw 28D features against PCA-reduced features.

```bash
python src/pca_comparison.py --rows 500000 --components 10 
```

Use this after the core model runs if you want dimensionality-reduction analysis.

### 8. Run the cluster label integration analysis

This depends on a clustered dataset from Project 2 named `higgs_clustered.csv`.
 
If you do not already have `higgs_clustered.csv`, run the k-means.py to get that dataset.

Once you have that file in `processed/`, you can run the command :

```bash
python cluster_label_integration.py --rows 200000
```

### 9. Review outputs in the results folders

After the runs finish, review:

1. `processed/benchmark_results.csv`
2. `Analysis_and_Findings/pca_comparison/`
3. `Analysis_and_Findings/cluster_integration/`

### Recommended practical order

If you want the shortest sensible workflow, use this sequence:

1. `cleaning.py`
2. One or more individual model scripts
3. `benchmark_metrics.py`
4. `pca_comparison.py`
5. `run_cluster_integration_batch.py` after you have `higgs_clustered.csv` from Project 2
