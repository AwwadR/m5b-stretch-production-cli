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

## Repository Contents

- `compare_models.py` — main production CLI script
- `README.md` — usage and project documentation
- `requirements.txt` — Python dependencies
- `data/telecom_churn.csv` — input dataset used for testing
- `output/model_comparison.csv` — saved metrics table
- `output/comparison_summary.txt` — saved text summary
- `output/pr_auc_comparison.png` — saved comparison plot

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

## Dataset

The script expects a CSV file containing the telecom churn dataset.

The target column is:

- `churned`

The numeric feature columns used in the pipeline are:

- `tenure`
- `monthly_charges`
- `total_charges`
- `num_support_calls`
- `senior_citizen`
- `has_partner`
- `has_dependents`
- `contract_months`

During validation, the script checks that:

- the file exists
- the dataset is not empty
- all required columns are present
- the target column contains at least two classes

---

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

The script does not display plots interactively. It saves them directly, which is more appropriate for a CLI pipeline.

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

## Why PR-AUC Matters Here

This dataset is imbalanced, with only about **16.36%** positive class examples (`churned = 1`).

Because of that, PR-AUC is a more useful comparison metric than accuracy alone. A model can achieve high accuracy by predicting the majority class most of the time, but still fail to detect churners well.

That is exactly what happened with some models in this comparison:

- `LogisticRegression_Default` reached **0.8518** accuracy
- but its recall was only **0.1630**

So in this project, PR-AUC gave a more honest view of how well the models handled the minority class.

---

## Script Design

The script is organized into clear functions to make it easier to test and maintain:

- `parse_args()`
- `ensure_output_dir()`
- `load_data()`
- `validate_data()`
- `build_preprocessor()`
- `get_models()`
- `run_dry_run()`
- `train_and_evaluate()`
- `save_metrics_table()`
- `save_summary()`
- `save_plot()`
- `save_results()`
- `main()`

It also uses:

- `argparse` for CLI argument parsing
- `logging` instead of `print`
- `if __name__ == "__main__":` so the script can be imported safely

---

## Error Handling

The script exits with a non-zero exit code if:

- the input file does not exist
- the dataset is empty
- required columns are missing
- the target column is invalid
- another runtime error happens during execution

This makes the script more reliable and easier to use from the command line.

---

## Reflection

This stretch helped me practice an important engineering skill: taking a working ML workflow and turning it into a tool that another person can actually run and use.

The main improvement over a notebook-style workflow is that the pipeline is now:

- reproducible
- configurable
- easier to validate
- easier to rerun
- easier to share

Instead of relying on notebook state or manual cell execution, the whole process can now be run with one command.

---

## Author

Built as part of **Module 5 Week B — Stretch: From Notebook to Production Script**.
