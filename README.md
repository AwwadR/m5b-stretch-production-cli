# Module 5 Week B Stretch — Production Model Comparison CLI

A production-style command-line script for running a full churn model comparison pipeline, validating the dataset, and saving all results to an output directory.

## Project Overview

In the original Integration 5B task, the model comparison pipeline was built as a normal Python workflow. In this stretch assignment, I refactored that work into a production-style CLI tool so that someone can run the full pipeline directly from the command line without reading the source code or using a notebook.

This script:

- loads the telecom churn dataset from a CSV file
- validates the dataset before training
- supports a `--dry-run` mode for checking configuration without fitting models
- compares 6 model configurations using stratified cross-validation
- saves all results to files instead of displaying them interactively

This makes the workflow more realistic and closer to how ML pipelines are used in professional environments.

---

## Installation

Clone the repository and move into the project folder:

```bash
git clone <your-repo-url>
cd m5b-stretch-production-cli
```

Create and activate a virtual environment:

### Git Bash on Windows
```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```


## Usage

General command:

```bash
python compare_models.py --data-path PATH_TO_CSV [--output-dir OUTPUT_DIR] [--n-folds N] [--random-seed SEED] [--dry-run]
```

### Required argument

- `--data-path`  
  Path to the input dataset CSV file.

### Optional arguments

- `--output-dir`  
  Directory where all results will be saved.  
  Default: `./output`

- `--n-folds`  
  Number of stratified cross-validation folds.  
  Default: `5`

- `--random-seed`  
  Random seed for reproducibility.  
  Default: `42`

- `--dry-run`  
  Validates the data and prints the pipeline configuration without training any models.

---

## Example Commands

### 1) Normal run

```bash
python compare_models.py --data-path data/telecom_churn.csv
```

### 2) Dry run

```bash
python compare_models.py --data-path data/telecom_churn.csv --dry-run
```

### 3) Custom output folder

```bash
python compare_models.py --data-path data/telecom_churn.csv --output-dir results
```

### 4) Different CV setting

```bash
python compare_models.py --data-path data/telecom_churn.csv --n-folds 3 --random-seed 7
```

---

## Dry Run Behavior

The `--dry-run` flag is included to make the script safer and more production-like.

When dry run is used, the script:

- loads the dataset
- checks the required columns
- reports dataset shape
- reports class distribution
- prints the full pipeline configuration
- exits without fitting any models

In my test run, the script successfully validated the dataset and reported:

- dataset shape: **(4500, 14)**
- no missing values in required columns
- class distribution:
  - class `0`: **0.836444**
  - class `1`: **0.163556**

This was useful because it confirmed that the data path, column names, and pipeline settings were correct before running the full training pipeline.

---

## Models Compared

The script compares these 6 model configurations:

1. `Dummy`
2. `LogisticRegression_Default`
3. `LogisticRegression_Balanced`
4. `DecisionTree_Default`
5. `DecisionTree_Balanced_Depth5`
6. `RandomForest_Balanced_Depth10`

The comparison is done using **5-fold Stratified Cross-Validation**.

The saved evaluation metrics are:

- accuracy
- precision
- recall
- f1
- PR-AUC

---

## Output Files

All results are saved to the output directory.

### 1) `model_comparison.csv`
Contains the full metrics table with mean and standard deviation across folds.

### 2) `comparison_summary.txt`
Contains:
- the full results table
- the best model by PR-AUC
- a short summary statement

### 3) `pr_auc_comparison.png`
Bar chart comparing the models by PR-AUC.

In `--dry-run` mode, no output files are created.

---

## Results Summary

Based on the saved output from my run, the best model by **PR-AUC** was:

- **RandomForest_Balanced_Depth10**
- PR-AUC = **0.4663**
- F1 = **0.4472**
- Recall = **0.4931**
- Accuracy = **0.8004**

Other important results:

- `DecisionTree_Balanced_Depth5` had the highest recall at **0.8016** and the highest F1 score at **0.4708**
- `LogisticRegression_Default` had the highest accuracy at **0.8518**, but its recall was much lower at **0.1630**
- `Dummy` performed worst overall, with:
  - precision = **0.0000**
  - recall = **0.0000**
  - f1 = **0.0000**
  - PR-AUC = **0.1636**

This shows why accuracy alone is not enough for an imbalanced churn problem. Even though `LogisticRegression_Default` had the highest accuracy, the best PR-AUC came from `RandomForest_Balanced_Depth10`, which made it the strongest overall model under this evaluation setup.

---
