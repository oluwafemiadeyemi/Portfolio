"""
End-to-end training pipeline for the Enterprise Brand Intelligence Platform.
Loads data, engineers features, trains the best sentiment classifier,
evaluates it, and saves the model artifact.

Usage:
    python src/train_pipeline.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is on the path
_BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE_DIR))

from src.data_loader import load_all_data
from src.features import build_feature_pipeline
from src.models import (
    cross_validate_model,
    evaluate_model,
    save_model,
    train_sentiment_classifier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_pipeline")


def main() -> None:
    """Run the complete training pipeline."""
    logger.info("=" * 60)
    logger.info("Enterprise Brand Intelligence — Training Pipeline")
    logger.info("=" * 60)

    # 1. Load data
    logger.info("Step 1/5 — Loading data")
    df = load_all_data(str(_BASE_DIR))
    logger.info("Loaded %d records", len(df))

    # 2. Feature engineering
    logger.info("Step 2/5 — Building feature pipeline")
    feature_matrix, enriched_df = build_feature_pipeline(df)
    logger.info("Feature matrix: %s", feature_matrix.shape)

    # 3. Prepare labels
    logger.info("Step 3/5 — Preparing labels")

    def _label(c: float) -> str:
        if c >= 0.05:
            return "positive"
        elif c <= -0.05:
            return "negative"
        return "neutral"

    y = enriched_df["compound_sentiment"].apply(_label).values
    unique, counts = np.unique(y, return_counts=True)
    logger.info("Label distribution: %s", dict(zip(unique, counts)))

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    logger.info("Train: %d | Test: %d", len(y_train), len(y_test))

    # 4. Train
    logger.info("Step 4/5 — Training models")
    model = train_sentiment_classifier(X_train, y_train)
    model_name = getattr(model, "_model_name", "Unknown")
    logger.info("Best model selected: %s", model_name)

    # Cross-validation
    cv_results = cross_validate_model(model, X_train, y_train, cv=5)
    logger.info(
        "CV Results — accuracy: %.4f±%.4f | f1_macro: %.4f±%.4f",
        cv_results["accuracy_mean"], cv_results["accuracy_std"],
        cv_results["f1_macro_mean"], cv_results["f1_macro_std"],
    )

    # 5. Evaluate
    logger.info("Step 5/5 — Evaluating on held-out test set")
    eval_results = evaluate_model(model, X_test, y_test)
    logger.info("Test accuracy:    %.4f", eval_results["accuracy"])
    logger.info("Test F1-macro:    %.4f", eval_results["f1_macro"])
    logger.info("Test F1-weighted: %.4f", eval_results["f1_weighted"])
    logger.info("\n%s", eval_results["classification_report"])

    # Save model
    model_path = _BASE_DIR / "data" / "models" / "sentiment_model.joblib"
    save_model(model, model_path)
    logger.info("Model saved to %s", model_path)

    # Save evaluation metrics
    import json
    metrics_path = _BASE_DIR / "data" / "models" / "eval_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as fh:
        json.dump(
            {
                "model": model_name,
                "test": {k: v for k, v in eval_results.items() if k != "classification_report"},
                "cv": cv_results,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "feature_matrix_shape": list(feature_matrix.shape),
            },
            fh, indent=2, default=str,
        )
    logger.info("Evaluation metrics saved to %s", metrics_path)

    logger.info("=" * 60)
    logger.info("Training pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
