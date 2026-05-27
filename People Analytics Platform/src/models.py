"""
Model Training and Evaluation for Employee Attrition Prediction
Implements XGBoost, LightGBM, RandomForest, and Logistic Regression.
"""

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def _get_class_weight(y: pd.Series) -> float:
    """Returns scale_pos_weight for imbalanced attrition data."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return float(n_neg / max(n_pos, 1))


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Any]:
    """
    Trains XGBoost, LightGBM, RandomForest, and Logistic Regression classifiers.
    All with class_weight='balanced' for attrition imbalance (~84%/16%).

    Args:
        X_train: Training feature matrix
        y_train: Binary attrition target

    Returns:
        Dict of {model_name: fitted_model}
    """
    models: Dict[str, Any] = {}
    spw = _get_class_weight(y_train)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── LightGBM ──────────────────────────────────────────────────────────────
    if _HAS_LGB:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=400,
            num_leaves=63,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            scale_pos_weight=spw,
            objective="binary",
            metric="auc",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        cv = cross_val_score(lgb_model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
        lgb_model.fit(X_train, y_train)
        models["LightGBM"] = lgb_model
        logger.info(f"LightGBM CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if _HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )
        cv = cross_val_score(xgb_model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        models["XGBoost"] = xgb_model
        logger.info(f"XGBoost CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cv = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    rf_model.fit(X_train, y_train)
    models["RandomForest"] = rf_model
    logger.info(f"RandomForest CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.5,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    cv = cross_val_score(lr_model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    lr_model.fit(X_train, y_train)
    models["LogisticRegression"] = lr_model
    logger.info(f"LogisticRegression CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")

    logger.info(f"Trained {len(models)} models")
    return models


def select_best_model(
    models_dict: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[str, Any]:
    """
    Selects the best model by weighted F1 score on validation set.

    Args:
        models_dict: Dict of {name: fitted_model}
        X_val: Validation features
        y_val: Validation target

    Returns:
        Tuple of (best_model_name, best_model)
    """
    best_name = None
    best_score = -1.0

    for name, model in models_dict.items():
        y_pred = model.predict(X_val)
        score = f1_score(y_val, y_pred, average="weighted")
        logger.info(f"{name} validation F1 (weighted): {score:.4f}")
        if score > best_score:
            best_score = score
            best_name = name

    logger.info(f"Best model: {best_name} (F1={best_score:.4f})")
    return best_name, models_dict[best_name]


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict:
    """
    Full model evaluation: metrics + classification report.

    Args:
        model: Fitted model
        X_test: Test features
        y_test: Test target
        threshold: Decision threshold

    Returns:
        Dict with all metrics
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision": round(average_precision_score(y_test, y_proba), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "f1_binary": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "attrition_rate_predicted": round(y_pred.mean(), 4),
        "attrition_rate_actual": round(float(np.asarray(y_test).mean()), 4),
        "n_test": len(y_test),
        "classification_report": classification_report(y_test, y_pred, target_names=["Retained", "Attrited"]),
    }

    logger.info(
        f"Evaluation: AUC={metrics['roc_auc']}, F1_weighted={metrics['f1_weighted']}, "
        f"Recall={metrics['recall']}"
    )
    return metrics


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> Dict:
    """
    Stratified cross-validation with multiple scoring metrics.

    Args:
        model: Model to validate
        X: Feature matrix
        y: Target series
        cv: Number of folds

    Returns:
        Dict with mean and std for each metric
    """
    from sklearn.model_selection import cross_validate as sklearn_cv

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ["roc_auc", "f1_weighted", "precision_weighted", "recall_weighted"]

    results = sklearn_cv(model, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=True)

    summary = {}
    for metric in scoring:
        key = f"test_{metric}"
        if key in results:
            summary[metric] = {
                "mean": round(float(results[key].mean()), 4),
                "std": round(float(results[key].std()), 4),
                "fold_scores": [round(float(s), 4) for s in results[key]],
            }

    logger.info(f"CV ROC AUC: {summary.get('roc_auc', {}).get('mean', 'N/A')}")
    return summary


def save_artifacts(
    model: Any,
    feature_cols: List[str],
    model_dir: str,
    model_name: str = "best_model",
) -> None:
    """
    Saves model pickle + feature list to disk.

    Args:
        model: Fitted model
        feature_cols: Feature column names
        model_dir: Output directory
        model_name: Base name for artifacts
    """
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)

    model_path = path / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {"feature_cols": feature_cols, "model_name": model_name}
    meta_path = path / f"{model_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Artifacts saved: {model_path}")


def load_artifacts(
    model_dir: str,
    model_name: str = "best_model",
) -> Tuple[Any, List[str]]:
    """
    Loads model and metadata from disk.

    Args:
        model_dir: Directory with artifact files
        model_name: Base name for artifacts

    Returns:
        Tuple of (model, feature_cols)
    """
    path = Path(model_dir)
    model_path = path / f"{model_name}.pkl"
    meta_path = path / f"{model_name}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    logger.info(f"Artifacts loaded from {path}")
    return model, meta["feature_cols"]
