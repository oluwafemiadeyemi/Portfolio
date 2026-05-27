"""
Model Training & Evaluation Module
Real-Time Credit Risk & Fraud Detection Intelligence Platform

XGBoost + LightGBM stacked ensemble with Isolation Forest anomaly component.
Handles class imbalance, early stopping, threshold optimization, and artifact I/O.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Train an XGBoost classifier optimised for fraud detection.

    Class imbalance handled via scale_pos_weight = n_negative / n_positive.
    Uses early stopping on validation AUC-PR.

    Args:
        X_train: Training features.
        y_train: Training labels (0/1).
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Tuple of (trained XGBClassifier, eval_results dict).
    """
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = n_neg / max(n_pos, 1)

    logger.info(
        f"Training XGBoost | n_train={len(X_train):,} | "
        f"fraud={n_pos:,} ({n_pos/len(y_train):.2%}) | "
        f"scale_pos_weight={scale_pos_weight:.1f}"
    )

    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric=["aucpr", "auc"],
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=False,
        early_stopping_rounds=50,  # XGBoost >= 2.0: set in constructor, not fit()
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=100,
    )

    best_iteration = model.best_iteration
    results = model.evals_result()

    val_auc = results["validation_1"]["auc"][best_iteration]
    val_aucpr = results["validation_1"]["aucpr"][best_iteration]
    logger.info(
        f"XGBoost training complete | best_iteration={best_iteration} | "
        f"val_AUC={val_auc:.4f} | val_AUC-PR={val_aucpr:.4f}"
    )

    eval_results: Dict[str, Any] = {
        "best_iteration": best_iteration,
        "val_auc": val_auc,
        "val_aucpr": val_aucpr,
        "train_auc": results["validation_0"]["auc"][best_iteration],
        "train_aucpr": results["validation_0"]["aucpr"][best_iteration],
        "full_history": results,
    }
    return model, eval_results


# ─────────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────────

def train_lightgbm(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """
    Train a LightGBM classifier optimised for fraud detection.

    Imbalance handled via class_weight='balanced' and is_unbalance=True.
    Uses early stopping on validation AUC-PR.

    Args:
        X_train: Training features.
        y_train: Training labels (0/1).
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Tuple of (trained LGBMClassifier, eval_results dict).
    """
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    logger.info(
        f"Training LightGBM | n_train={len(X_train):,} | "
        f"fraud={n_pos:,} ({n_pos/len(y_train):.2%})"
    )

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        is_unbalance=True,
        objective="binary",
        metric=["average_precision", "auc"],
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )

    best_iteration = model.best_iteration_
    val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_proba))
    val_aucpr = float(average_precision_score(y_val, val_proba))

    logger.info(
        f"LightGBM training complete | best_iteration={best_iteration} | "
        f"val_AUC={val_auc:.4f} | val_AUC-PR={val_aucpr:.4f}"
    )

    eval_results: Dict[str, Any] = {
        "best_iteration": best_iteration,
        "val_auc": val_auc,
        "val_aucpr": val_aucpr,
    }
    return model, eval_results


# ─────────────────────────────────────────────
# Isolation Forest (Unsupervised Anomaly)
# ─────────────────────────────────────────────

def train_isolation_forest(
    X_train: pd.DataFrame | np.ndarray,
    contamination: float = 0.035,
) -> IsolationForest:
    """
    Train an Isolation Forest for unsupervised anomaly detection.

    Can be used as an additional signal or for pre-filtering.

    Args:
        X_train: Training features (labels NOT used — unsupervised).
        contamination: Expected fraction of anomalies (matches fraud rate).

    Returns:
        Trained IsolationForest model.
    """
    logger.info(
        f"Training Isolation Forest | n_train={len(X_train):,} | "
        f"contamination={contamination:.3f}"
    )

    # Fill any remaining NaN for IsolationForest (doesn't handle them)
    if isinstance(X_train, pd.DataFrame):
        X_filled = X_train.fillna(-999).values
    else:
        X_filled = np.where(np.isnan(X_train), -999, X_train)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        max_features=0.8,
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    model.fit(X_filled)

    # Quick sanity check
    scores = model.decision_function(X_filled[:1000])
    logger.info(
        f"Isolation Forest training complete | "
        f"anomaly_score range: [{scores.min():.3f}, {scores.max():.3f}]"
    )
    return model


# ─────────────────────────────────────────────
# Stacked Ensemble
# ─────────────────────────────────────────────

def train_stacked_ensemble(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    n_folds: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Two-layer stacked ensemble: XGBoost + LightGBM base learners,
    LogisticRegression meta-learner trained on out-of-fold (OOF) predictions.

    Architecture:
      Layer 1: XGBClassifier, LGBMClassifier (trained on K folds)
      Layer 2: LogisticRegression on [xgb_oof_proba, lgb_oof_proba]

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        n_folds: Number of cross-validation folds.

    Returns:
        Tuple of (ensemble_dict, eval_results):
          ensemble_dict contains keys: 'xgb', 'lgb', 'meta', 'feature_names'
    """
    X_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
    X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val

    feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else [f"f{i}" for i in range(X_arr.shape[1])]

    logger.info(f"Training stacked ensemble with {n_folds}-fold OOF strategy")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_xgb = np.zeros(len(X_arr), dtype=np.float32)
    oof_lgb = np.zeros(len(X_arr), dtype=np.float32)
    val_preds_xgb = np.zeros(len(X_val_arr), dtype=np.float32)
    val_preds_lgb = np.zeros(len(X_val_arr), dtype=np.float32)

    xgb_models: List[xgb.XGBClassifier] = []
    lgb_models: List[lgb.LGBMClassifier] = []

    for fold_idx, (train_idx, oof_idx) in enumerate(skf.split(X_arr, y_arr)):
        logger.info(f"Fold {fold_idx + 1}/{n_folds}")
        X_fold_train, X_fold_val = X_arr[train_idx], X_arr[oof_idx]
        y_fold_train, y_fold_val = y_arr[train_idx], y_arr[oof_idx]

        # XGBoost fold
        xgb_fold, _ = train_xgboost(
            pd.DataFrame(X_fold_train, columns=feature_names),
            y_fold_train,
            pd.DataFrame(X_fold_val, columns=feature_names),
            y_fold_val,
        )
        oof_xgb[oof_idx] = xgb_fold.predict_proba(X_fold_val)[:, 1]
        val_preds_xgb += xgb_fold.predict_proba(X_val_arr)[:, 1] / n_folds
        xgb_models.append(xgb_fold)

        # LightGBM fold
        lgb_fold, _ = train_lightgbm(
            pd.DataFrame(X_fold_train, columns=feature_names),
            y_fold_train,
            pd.DataFrame(X_fold_val, columns=feature_names),
            y_fold_val,
        )
        oof_lgb[oof_idx] = lgb_fold.predict_proba(X_fold_val)[:, 1]
        val_preds_lgb += lgb_fold.predict_proba(X_val_arr)[:, 1] / n_folds
        lgb_models.append(lgb_fold)

    # OOF evaluation
    oof_aucpr_xgb = float(average_precision_score(y_arr, oof_xgb))
    oof_aucpr_lgb = float(average_precision_score(y_arr, oof_lgb))
    logger.info(f"OOF AUC-PR | XGB={oof_aucpr_xgb:.4f} | LGB={oof_aucpr_lgb:.4f}")

    # Meta-learner
    meta_train = np.column_stack([oof_xgb, oof_lgb])
    meta_val = np.column_stack([val_preds_xgb, val_preds_lgb])

    meta_learner = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=42
    )
    meta_learner.fit(meta_train, y_arr)

    meta_val_proba = meta_learner.predict_proba(meta_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, meta_val_proba))
    val_aucpr = float(average_precision_score(y_val, meta_val_proba))

    logger.info(
        f"Stacked ensemble complete | "
        f"val_AUC={val_auc:.4f} | val_AUC-PR={val_aucpr:.4f}"
    )

    ensemble = {
        "xgb_models": xgb_models,
        "lgb_models": lgb_models,
        "meta": meta_learner,
        "feature_names": feature_names,
    }

    eval_results: Dict[str, Any] = {
        "oof_aucpr_xgb": oof_aucpr_xgb,
        "oof_aucpr_lgb": oof_aucpr_lgb,
        "val_auc": val_auc,
        "val_aucpr": val_aucpr,
        "n_folds": n_folds,
    }

    return ensemble, eval_results


def ensemble_predict_proba(
    ensemble: Dict[str, Any],
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    Generate ensemble probability predictions using the stacked meta-learner.

    Args:
        ensemble: Dict returned by train_stacked_ensemble.
        X: Feature matrix.

    Returns:
        Array of fraud probabilities.
    """
    feature_names = ensemble["feature_names"]
    X_arr = X.values if isinstance(X, pd.DataFrame) else X

    xgb_preds = np.mean(
        [m.predict_proba(X_arr)[:, 1] for m in ensemble["xgb_models"]], axis=0
    )
    lgb_preds = np.mean(
        [m.predict_proba(X_arr)[:, 1] for m in ensemble["lgb_models"]], axis=0
    )

    meta_input = np.column_stack([xgb_preds, lgb_preds])
    return ensemble["meta"].predict_proba(meta_input)[:, 1]


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.

    Metrics: AUC-ROC, AUC-PR, F1, Precision, Recall, Average Precision,
    Confusion Matrix, Optimal Threshold via Youden's J statistic.

    Args:
        model: Trained classifier with predict_proba method, or ensemble dict.
        X_test: Test features.
        y_test: True labels.
        threshold: Classification threshold.

    Returns:
        Dictionary of evaluation metrics.
    """
    y_arr = y_test.values if isinstance(y_test, pd.Series) else y_test

    # Handle ensemble dict vs standard sklearn model
    if isinstance(model, dict) and "xgb_models" in model:
        y_proba = ensemble_predict_proba(model, X_test)
    else:
        y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_proba >= threshold).astype(int)

    auc_roc = float(roc_auc_score(y_arr, y_proba))
    avg_precision = float(average_precision_score(y_arr, y_proba))
    f1 = float(f1_score(y_arr, y_pred, zero_division=0))
    precision = float(precision_score(y_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_arr, y_pred, zero_division=0))
    cm = confusion_matrix(y_arr, y_pred).tolist()

    optimal_threshold = find_optimal_threshold(y_arr, y_proba)
    y_pred_opt = (y_proba >= optimal_threshold).astype(int)
    f1_opt = float(f1_score(y_arr, y_pred_opt, zero_division=0))

    results: Dict[str, Any] = {
        "auc_roc": auc_roc,
        "auc_pr": avg_precision,
        "average_precision": avg_precision,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "optimal_threshold": optimal_threshold,
        "f1_at_optimal_threshold": f1_opt,
        "threshold_used": threshold,
        "n_test": len(y_arr),
        "n_fraud_test": int(np.sum(y_arr == 1)),
        "fraud_rate_test": float(np.mean(y_arr)),
    }

    logger.info(
        f"Evaluation | AUC-ROC={auc_roc:.4f} | AUC-PR={avg_precision:.4f} | "
        f"F1={f1:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | "
        f"OptimalThreshold={optimal_threshold:.4f}"
    )
    return results


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """
    Find the probability threshold that maximises F1-score.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities.

    Returns:
        Optimal threshold as float.
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)

    # F1 = 2 * P * R / (P + R)
    denom = precision_arr + recall_arr
    f1_scores = np.where(denom > 0, 2 * precision_arr * recall_arr / denom, 0.0)

    # thresholds has len = len(precision) - 1
    best_idx = int(np.argmax(f1_scores[:-1]))
    optimal = float(thresholds[best_idx])

    logger.debug(
        f"Optimal threshold (max F1): {optimal:.4f} → "
        f"F1={f1_scores[best_idx]:.4f}"
    )
    return optimal


# ─────────────────────────────────────────────
# Artifact I/O
# ─────────────────────────────────────────────

def save_artifacts(
    model: Any,
    threshold: float,
    feature_cols: List[str],
    model_dir: str | Path,
    model_name: str = "fraud_model",
) -> None:
    """
    Persist model, optimal threshold, and feature column list to disk.

    Saves:
    - <model_name>.joblib  (model object)
    - <model_name>_metadata.json  (threshold + feature list + eval metrics if provided)

    Args:
        model: Trained model object (XGB, LGB, or ensemble dict).
        threshold: Optimal classification threshold.
        feature_cols: Ordered list of feature column names.
        model_dir: Directory to save artifacts.
        model_name: Base filename prefix.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}.joblib"
    metadata_path = model_dir / f"{model_name}_metadata.json"

    joblib.dump(model, model_path, compress=3)
    logger.info(f"Model saved to {model_path}")

    metadata: Dict[str, Any] = {
        "threshold": threshold,
        "feature_cols": feature_cols,
        "model_name": model_name,
        "n_features": len(feature_cols),
    }
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")


def load_artifacts(
    model_dir: str | Path,
    model_name: str = "fraud_model",
) -> Tuple[Any, float, List[str]]:
    """
    Load model artifacts from disk.

    Args:
        model_dir: Directory containing saved artifacts.
        model_name: Base filename prefix used in save_artifacts.

    Returns:
        Tuple of (model, threshold, feature_cols).

    Raises:
        FileNotFoundError: If model files are not found.
    """
    model_dir = Path(model_dir)
    model_path = model_dir / f"{model_name}.joblib"
    metadata_path = model_dir / f"{model_name}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    with open(metadata_path, "r") as fh:
        metadata = json.load(fh)

    threshold: float = float(metadata["threshold"])
    feature_cols: List[str] = metadata["feature_cols"]
    logger.info(
        f"Metadata loaded | threshold={threshold:.4f} | "
        f"n_features={len(feature_cols)}"
    )
    return model, threshold, feature_cols
