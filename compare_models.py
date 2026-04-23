"""
Module 5 Week B — Stretch: From Notebook to Production Script

Production CLI tool for comparing churn classification models using
cross-validation and saving results to disk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


TARGET_COLUMN = "churned"

NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "contract_months",
]

EXPECTED_COLUMNS = NUMERIC_FEATURES + [TARGET_COLUMN]

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "pr_auc": make_scorer(average_precision_score, response_method="predict_proba"),
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a production model comparison pipeline for telecom churn prediction."
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the input dataset CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory where metrics, summaries, and plots will be saved. Default: ./output",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds. Default: 5",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and print pipeline configuration without training models.",
    )

    return parser.parse_args()


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create output directory if it does not already exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready: %s", output_path.resolve())
    return output_path


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Load dataset from CSV file."""
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded data from %s", path.resolve())
    logger.info("Dataset shape: %s", df.shape)
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Validate expected columns and target integrity."""
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must contain at least two classes."
        )

    missing_value_counts = df[EXPECTED_COLUMNS].isnull().sum()
    if (missing_value_counts > 0).any():
        logger.warning(
            "Missing values detected in required columns:\n%s",
            missing_value_counts[missing_value_counts > 0],
        )
    else:
        logger.info("No missing values detected in required columns.")

    logger.info("Validation successful.")
    logger.info(
        "Class distribution:\n%s",
        df[TARGET_COLUMN].value_counts(normalize=True).sort_index(),
    )


def build_preprocessor() -> StandardScaler:
    """Build preprocessing step for numeric features."""
    return StandardScaler()


def get_models(random_seed: int) -> dict[str, object]:
    """Return the set of models to compare."""
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression_Default": LogisticRegression(
            max_iter=1000,
            random_state=random_seed,
        ),
        "LogisticRegression_Balanced": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_seed,
        ),
        "DecisionTree_Default": DecisionTreeClassifier(
            random_state=random_seed,
        ),
        "DecisionTree_Balanced_Depth5": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=random_seed,
        ),
        "RandomForest_Balanced_Depth10": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }


def run_dry_run(
    df: pd.DataFrame,
    args: argparse.Namespace,
    output_path: Path,
    models: dict[str, object],
) -> None:
    """Validate configuration without training any models."""
    logger.info("DRY RUN mode enabled. No models will be trained.")
    logger.info("Pipeline configuration:")
    logger.info("  data_path: %s", Path(args.data_path).resolve())
    logger.info("  output_dir: %s", output_path.resolve())
    logger.info("  n_folds: %d", args.n_folds)
    logger.info("  random_seed: %d", args.random_seed)
    logger.info("  target_column: %s", TARGET_COLUMN)
    logger.info("  numeric_features: %s", NUMERIC_FEATURES)
    logger.info("  models: %s", list(models.keys()))
    logger.info("  dataset_shape: %s", df.shape)
    logger.info(
        "  class_distribution:\n%s",
        df[TARGET_COLUMN].value_counts(normalize=True).sort_index(),
    )


def train_and_evaluate(
    df: pd.DataFrame,
    n_folds: int,
    random_seed: int,
) -> pd.DataFrame:
    """Train and evaluate all models with stratified cross-validation."""
    logger.info("Starting training and evaluation.")

    X = df[NUMERIC_FEATURES]
    y = df[TARGET_COLUMN]

    preprocessor = build_preprocessor()
    models = get_models(random_seed)

    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_seed,
    )

    results = []

    for model_name, model in models.items():
        logger.info("Evaluating model: %s", model_name)

        pipeline = Pipeline(
            steps=[
                ("scaler", preprocessor),
                ("model", model),
            ]
        )

        cv_results = cross_validate(
            estimator=pipeline,
            X=X,
            y=y,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            return_train_score=False,
        )

        row = {"model": model_name}
        for metric_name in SCORING.keys():
            scores = cv_results[f"test_{metric_name}"]
            row[f"{metric_name}_mean"] = scores.mean()
            row[f"{metric_name}_std"] = scores.std()

        results.append(row)

        logger.info(
            "%s complete | accuracy=%.4f | precision=%.4f | recall=%.4f | f1=%.4f | pr_auc=%.4f",
            model_name,
            row["accuracy_mean"],
            row["precision_mean"],
            row["recall_mean"],
            row["f1_mean"],
            row["pr_auc_mean"],
        )

    results_df = pd.DataFrame(results).sort_values("pr_auc_mean", ascending=False)
    logger.info("Training and evaluation finished.")
    return results_df


def save_metrics_table(results_df: pd.DataFrame, output_path: Path) -> Path:
    """Save metrics table as CSV."""
    metrics_path = output_path / "model_comparison.csv"
    results_df.to_csv(metrics_path, index=False)
    logger.info("Saved metrics table to %s", metrics_path.resolve())
    return metrics_path


def save_summary(results_df: pd.DataFrame, output_path: Path) -> Path:
    """Save a text summary of the results."""
    summary_path = output_path / "comparison_summary.txt"

    best_model = results_df.iloc[0]

    lines = [
        "Model Comparison Summary",
        "=" * 30,
        "",
        "Results table:",
        results_df.to_string(index=False),
        "",
        "Best model by PR-AUC:",
        best_model.to_string(),
        "",
        (
            f"Top model: {best_model['model']} "
            f"(PR-AUC={best_model['pr_auc_mean']:.4f}, "
            f"F1={best_model['f1_mean']:.4f}, "
            f"Recall={best_model['recall_mean']:.4f})"
        ),
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved summary to %s", summary_path.resolve())
    return summary_path


def save_plot(results_df: pd.DataFrame, output_path: Path) -> Path:
    """Save a PR-AUC comparison plot."""
    plot_path = output_path / "pr_auc_comparison.png"

    sorted_df = results_df.sort_values("pr_auc_mean", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_df["model"], sorted_df["pr_auc_mean"])
    plt.title("Model Comparison by PR-AUC")
    plt.xlabel("Model")
    plt.ylabel("PR-AUC")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    logger.info("Saved plot to %s", plot_path.resolve())
    return plot_path


def save_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save all result artifacts."""
    save_metrics_table(results_df, output_path)
    save_summary(results_df, output_path)
    save_plot(results_df, output_path)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        if args.n_folds < 2:
            raise ValueError("--n-folds must be at least 2.")

        output_path = ensure_output_dir(args.output_dir)

        df = load_data(args.data_path)
        validate_data(df)

        models = get_models(args.random_seed)

        if args.dry_run:
            run_dry_run(df, args, output_path, models)
            logger.info("Dry run completed successfully.")
            return

        results_df = train_and_evaluate(
            df=df,
            n_folds=args.n_folds,
            random_seed=args.random_seed,
        )
        save_results(results_df, output_path)

        logger.info("Pipeline completed successfully.")

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()