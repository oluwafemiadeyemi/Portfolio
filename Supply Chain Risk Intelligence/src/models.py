"""
models.py
=========
Model training, evaluation and persistence for Financial Distress Prediction.

Models:
  - XGBoost binary classifier with imbalance handling
  - LightGBM alternative
  - Cox Proportional Hazards survival model (time-to-distress)
  - Multi-horizon models (3/6/12/18 months)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.getLogger(__name__).warning("xgboost not installed — falling back to GradientBoosting.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logging.getLogger(__name__).warning("lightgbm not installed.")

try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logging.getLogger(__name__).warning("lifelines not installed — survival model unavailable.")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# XGBoost distress classifier
# ---------------------------------------------------------------------------

def train_xgboost_distress(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: Optional[float] = None,
) -> Any:
    """Train XGBoost classifier with imbalance correction.

    Parameters
    ----------
    X_train: Feature matrix.
    y_train: Binary distress target (1=distress, 0=healthy).
    scale_pos_weight: Weight for positive class. If None, computed automatically
                      from class frequencies (n_negative / n_positive).
    """
    X = X_train.copy().fillna(X_train.median(numeric_only=True))
    y = y_train.values if hasattr(y_train, "values") else np.array(y_train)

    if scale_pos_weight is None:
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = float(n_neg / max(n_pos, 1))
        logger.info("scale_pos_weight auto-computed: %.2f (n_neg=%d, n_pos=%d)", scale_pos_weight, n_neg, n_pos)

    if XGB_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )
    else:
        logger.info("Using GradientBoosting as XGBoost fallback.")
        sample_weights = compute_sample_weight("balanced", y)
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        )
        model.fit(X, y, sample_weight=sample_weights)
        return model

    logger.info("Training XGBoost distress model on %d samples…", len(X))
    model.fit(X, y)
    logger.info("XGBoost distress model training complete.")
    return model


# ---------------------------------------------------------------------------
# LightGBM distress classifier
# ---------------------------------------------------------------------------

def train_lightgbm_distress(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: Optional[float] = None,
) -> Any:
    """Train LightGBM classifier with imbalance correction."""
    X = X_train.copy().fillna(X_train.median(numeric_only=True))
    y = y_train.values if hasattr(y_train, "values") else np.array(y_train)

    if scale_pos_weight is None:
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = float(n_neg / max(n_pos, 1))

    if not LGB_AVAILABLE:
        logger.warning("LightGBM not available — falling back to XGBoost.")
        return train_xgboost_distress(X_train, y_train, scale_pos_weight)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    logger.info("Training LightGBM distress model on %d samples…", len(X))
    model.fit(X, y)
    logger.info("LightGBM distress model training complete.")
    return model


# ---------------------------------------------------------------------------
# Survival model (Cox PH)
# ---------------------------------------------------------------------------

def train_survival_model(
    X_train: pd.DataFrame,
    T_train: pd.Series,
    E_train: pd.Series,
) -> Any:
    """Fit a Cox Proportional Hazards model for time-to-distress.

    Parameters
    ----------
    X_train: Feature matrix.
    T_train: Duration (years until distress event or censoring).
    E_train: Event indicator (1=distress occurred, 0=censored/survived).

    Returns
    -------
    Fitted CoxPHFitter (from lifelines) or None if lifelines unavailable.
    """
    if not LIFELINES_AVAILABLE:
        logger.error("lifelines not installed — cannot train survival model.")
        return None

    X = X_train.copy().fillna(X_train.median(numeric_only=True))

    # CoxPH works best with a moderate number of features to avoid collinearity
    top_cols = X.columns[:min(20, len(X.columns))].tolist()
    X_sub = X[top_cols].copy()

    cox_df = X_sub.copy()
    cox_df["T"] = T_train.values if hasattr(T_train, "values") else np.array(T_train)
    cox_df["E"] = E_train.values if hasattr(E_train, "values") else np.array(E_train)

    # Remove any rows with NaN in T or E
    cox_df = cox_df.dropna(subset=["T", "E"])
    cox_df["T"] = cox_df["T"].clip(lower=0.01)  # Duration must be positive

    cph = CoxPHFitter(penalizer=0.1)
    try:
        logger.info("Fitting Cox PH model on %d observations…", len(cox_df))
        cph.fit(cox_df, duration_col="T", event_col="E")
        logger.info("Cox PH model fitted. Concordance index: %.4f", cph.concordance_index_)
    except Exception as exc:
        logger.error("Cox PH fitting failed: %s", exc)
        return None

    return cph


# ---------------------------------------------------------------------------
# Multi-horizon models
# ---------------------------------------------------------------------------

def train_multi_horizon_models(
    X_train: pd.DataFrame,
    y_dict: Dict[str, pd.Series],
    model_fn: str = "xgboost",
) -> Dict[str, Any]:
    """Train separate distress prediction models for each horizon.

    Parameters
    ----------
    X_train: Feature matrix.
    y_dict: Dict mapping horizon name → binary Series.
    model_fn: ``'xgboost'`` or ``'lightgbm'``.

    Returns
    -------
    Dict mapping horizon name → fitted model.
    """
    models: Dict[str, Any] = {}
    for horizon, y in y_dict.items():
        logger.info("Training model for horizon: %s", horizon)
        if model_fn == "lightgbm":
            models[horizon] = train_lightgbm_distress(X_train, y)
        else:
            models[horizon] = train_xgboost_distress(X_train, y)

    logger.info("Multi-horizon models trained: %s", list(models.keys()))
    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k: int = 50,
) -> Dict[str, float]:
    """Compute classification metrics for financial distress prediction.

    Returns AUC-ROC, precision@K, recall@K, average precision, F1.
    """
    X = X_test.copy().fillna(X_test.median(numeric_only=True))
    y_true = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    y_proba = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else model.predict(X).astype(float)
    )
    y_pred = (y_proba >= 0.5).astype(int)

    p_at_k = compute_precision_at_k(y_true, y_proba, k=k)
    r_at_k = _recall_at_k(y_true, y_proba, k=k)

    try:
        auc_roc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        auc_roc = 0.5

    try:
        avg_precision = float(average_precision_score(y_true, y_proba))
    except Exception:
        avg_precision = 0.0

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "average_precision": round(avg_precision, 4),
        f"precision_at_{k}": round(p_at_k, 4),
        f"recall_at_{k}": round(r_at_k, 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "distress_rate": round(float(y_true.mean()), 4),
        "n_samples": int(len(y_true)),
    }
    logger.info("Model evaluation: %s", metrics)
    return metrics


def compute_precision_at_k(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k: int = 50,
) -> float:
    """Precision@K: of the top-K highest-risk companies, what fraction actually failed?

    Parameters
    ----------
    y_true: Binary ground truth.
    y_proba: Predicted probabilities.
    k: Number of top-risk companies to consider.
    """
    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    return float(y_true[top_k_idx].mean())


def _recall_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int = 50) -> float:
    """Recall@K: of all actual distressed companies, what fraction appear in top-K?"""
    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    n_positives = y_true.sum()
    if n_positives == 0:
        return 0.0
    return float(y_true[top_k_idx].sum() / n_positives)


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_artifacts(
    models_dict: Dict[str, Any],
    feature_cols: List[str],
    model_dir: str | Path,
) -> None:
    """Persist all models and feature column list."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models_dict.items():
        if model is None:
            continue
        out_path = model_dir / f"{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(model, f)

    with open(model_dir / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    logger.info("Artifacts saved to %s: %s", model_dir, list(models_dict.keys()))


def load_artifacts(
    model_dir: str | Path,
) -> Tuple[Dict[str, Any], List[str]]:
    """Load all persisted models and feature column list.

    Returns
    -------
    Tuple[models_dict, feature_cols]
    """
    model_dir = Path(model_dir)
    models_dict: Dict[str, Any] = {}

    for pkl_file in model_dir.glob("*.pkl"):
        if pkl_file.stem == "feature_cols":
            continue
        with open(pkl_file, "rb") as f:
            models_dict[pkl_file.stem] = pickle.load(f)

    feature_cols_path = model_dir / "feature_cols.pkl"
    if feature_cols_path.exists():
        with open(feature_cols_path, "rb") as f:
            feature_cols = pickle.load(f)
    else:
        feature_cols = []

    logger.info("Artifacts loaded from %s: %s", model_dir, list(models_dict.keys()))
    return models_dict, feature_cols
