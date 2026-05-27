"""
models.py
=========
Model training, evaluation and persistence for Parkinson's Disease Detection.

Models:
  - Binary classifier (PD vs. healthy) — XGBoost / Random Forest / Logistic Regression
  - UPDRS severity regressor — XGBoost
  - Multi-modal ensemble — separate per-modality models + meta-learner
  - Subject-level GroupKFold cross-validation (critical for clinical validity)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.getLogger(__name__).warning("xgboost not installed — falling back to GradientBoosting.")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classifier(model_type: str = "xgboost") -> Any:
    """Instantiate a classifier by name."""
    if model_type == "xgboost" and XGB_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    elif model_type == "logistic":
        return LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    else:
        return GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)


# ---------------------------------------------------------------------------
# Binary classifier
# ---------------------------------------------------------------------------

def train_binary_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "xgboost",
) -> Any:
    """Train a PD vs. healthy binary classifier.

    Parameters
    ----------
    X_train: Feature matrix.
    y_train: Binary target (1=PD, 0=healthy).
    model_type: One of ``'xgboost'``, ``'random_forest'``, ``'logistic'``.

    Returns
    -------
    Fitted classifier.
    """
    X = X_train.copy().fillna(X_train.median(numeric_only=True))
    y = y_train.values if hasattr(y_train, "values") else np.array(y_train)

    model = _make_classifier(model_type)
    logger.info("Training binary classifier (%s) on %d samples…", model_type, len(X))
    model.fit(X, y)
    logger.info("Binary classifier training complete.")
    return model


# ---------------------------------------------------------------------------
# UPDRS regressor
# ---------------------------------------------------------------------------

def train_updrs_regressor(
    X_train: pd.DataFrame,
    y_updrs: pd.Series,
) -> Any:
    """Train an UPDRS severity score regressor (XGBoost or GBM)."""
    X = X_train.copy().fillna(X_train.median(numeric_only=True))
    y = y_updrs.values if hasattr(y_updrs, "values") else np.array(y_updrs)

    if XGB_AVAILABLE:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)

    logger.info("Training UPDRS regressor on %d samples…", len(X))
    model.fit(X, y)
    logger.info("UPDRS regressor training complete.")
    return model


# ---------------------------------------------------------------------------
# Multi-modal ensemble
# ---------------------------------------------------------------------------

def train_multimodal_ensemble(
    X_voice: pd.DataFrame,
    X_gait: pd.DataFrame,
    X_tremor: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
) -> Dict[str, Any]:
    """Train modality-specific models and a meta-learner stacker.

    Returns
    -------
    Dict with keys: 'voice', 'gait', 'tremor', 'meta', 'scaler'
    """
    y_arr = y.values if hasattr(y, "values") else np.array(y)

    def safe_fill(X: pd.DataFrame) -> pd.DataFrame:
        return X.copy().fillna(X.median(numeric_only=True))

    models: Dict[str, Any] = {}

    logger.info("Training voice sub-model…")
    models["voice"] = _make_classifier(model_type)
    models["voice"].fit(safe_fill(X_voice), y_arr)

    logger.info("Training gait sub-model…")
    models["gait"] = _make_classifier(model_type)
    models["gait"].fit(safe_fill(X_gait), y_arr)

    logger.info("Training tremor sub-model…")
    models["tremor"] = _make_classifier(model_type)
    models["tremor"].fit(safe_fill(X_tremor), y_arr)

    # Meta-features: predicted probabilities from each sub-model
    meta_X = np.column_stack([
        models["voice"].predict_proba(safe_fill(X_voice))[:, 1],
        models["gait"].predict_proba(safe_fill(X_gait))[:, 1],
        models["tremor"].predict_proba(safe_fill(X_tremor))[:, 1],
    ])

    scaler = StandardScaler()
    meta_X_scaled = scaler.fit_transform(meta_X)

    logger.info("Training meta-learner…")
    models["meta"] = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    models["meta"].fit(meta_X_scaled, y_arr)
    models["scaler"] = scaler

    logger.info("Multi-modal ensemble training complete.")
    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Compute clinical classification metrics.

    Returns
    -------
    Dict with: auc_roc, f1, sensitivity, specificity, ppv, npv, accuracy
    """
    X = X_test.copy().fillna(X_test.median(numeric_only=True))
    y_true = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    y_pred = model.predict(X)
    y_proba = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn + 1e-9)  # recall for PD
    specificity = tn / (tn + fp + 1e-9)
    ppv = tp / (tp + fp + 1e-9)           # precision
    npv = tn / (tn + fn + 1e-9)

    metrics = {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
    }
    logger.info("Classifier evaluation: %s", metrics)
    return metrics


def evaluate_regressor(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Compute regression metrics for UPDRS prediction."""
    X = X_test.copy().fillna(X_test.median(numeric_only=True))
    y_true = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    y_pred = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0

    metrics = {"rmse": rmse, "mae": mae, "r2": r2, "updrs_correlation": corr}
    logger.info("Regressor evaluation: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Subject-level cross-validation
# ---------------------------------------------------------------------------

def subject_level_cross_validate(
    model_type: str,
    df: pd.DataFrame,
    subject_col: str,
    target_col: str,
    feature_cols: List[str],
    n_folds: int = 5,
    task: str = "classification",
) -> Dict[str, Any]:
    """GroupKFold cross-validation respecting subject boundaries.

    Critical for clinical validity: ensures all recordings from the same
    subject are in the same fold (train or test, never both).

    Parameters
    ----------
    model_type: Classifier type name.
    df: Full dataset DataFrame.
    subject_col: Column with subject IDs.
    target_col: Column with the target variable.
    feature_cols: List of feature column names.
    n_folds: Number of CV folds.
    task: ``'classification'`` or ``'regression'``.

    Returns
    -------
    Dict with per-fold metrics and mean/std summaries.
    """
    groups = df[subject_col].values
    X = df[feature_cols].copy().fillna(df[feature_cols].median(numeric_only=True))
    y = df[target_col].values

    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if task == "classification":
            model = train_binary_classifier(X_train, pd.Series(y_train), model_type=model_type)
            metrics = evaluate_classifier(model, X_test, pd.Series(y_test))
        else:
            model = train_updrs_regressor(X_train, pd.Series(y_train))
            metrics = evaluate_regressor(model, X_test, pd.Series(y_test))

        metrics["fold"] = fold_idx + 1
        fold_metrics.append(metrics)
        logger.info("Fold %d/%d — %s", fold_idx + 1, n_folds, metrics)

    results_df = pd.DataFrame(fold_metrics)
    summary: Dict[str, Any] = {"fold_results": fold_metrics}
    for col in results_df.columns:
        if col == "fold":
            continue
        summary[f"{col}_mean"] = float(results_df[col].mean())
        summary[f"{col}_std"] = float(results_df[col].std())

    logger.info(
        "GroupKFold CV complete — AUC mean=%.3f ± %.3f",
        summary.get("auc_roc_mean", summary.get("r2_mean", 0)),
        summary.get("auc_roc_std", summary.get("r2_std", 0)),
    )
    return summary


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_artifacts(
    classif_model: Any,
    regress_model: Any,
    feature_cols: List[str],
    model_dir: str | Path,
) -> None:
    """Persist models and feature column list to disk."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "classifier.pkl", "wb") as f:
        pickle.dump(classif_model, f)
    with open(model_dir / "regressor.pkl", "wb") as f:
        pickle.dump(regress_model, f)
    with open(model_dir / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    logger.info("Artifacts saved to %s", model_dir)


def load_artifacts(
    model_dir: str | Path,
) -> Tuple[Any, Any, List[str]]:
    """Load persisted models and feature column list.

    Returns
    -------
    Tuple[classifier, regressor, feature_cols]
    """
    model_dir = Path(model_dir)

    with open(model_dir / "classifier.pkl", "rb") as f:
        classif_model = pickle.load(f)
    with open(model_dir / "regressor.pkl", "rb") as f:
        regress_model = pickle.load(f)
    with open(model_dir / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    logger.info("Artifacts loaded from %s", model_dir)
    return classif_model, regress_model, feature_cols
