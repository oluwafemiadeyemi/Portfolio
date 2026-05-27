"""
Model Training and Evaluation for Mortgage Underwriting
Implements LightGBM, XGBoost, and Logistic Regression with hyperparameter tuning.
"""

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Optional heavy imports
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    logger.warning("LightGBM not installed. LightGBM training will be skipped.")

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("XGBoost not installed. XGBoost training will be skipped.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False
    logger.warning("Optuna not installed. Using manual grid search for hyperparameter tuning.")


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None,
) -> Any:
    """
    Trains a LightGBM classifier with 5-fold cross-validation.

    Args:
        X_train: Feature matrix
        y_train: Binary target
        params: Optional dict of LightGBM hyperparameters

    Returns:
        Fitted LightGBM model
    """
    if not _HAS_LGB:
        raise ImportError("LightGBM is required for train_lightgbm()")

    default_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    logger.info(f"LightGBM CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(
        X_train,
        y_train,
        eval_set=None,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        if hasattr(lgb, "early_stopping")
        else None,
    )
    logger.info("LightGBM training complete")
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None,
) -> Any:
    """
    Trains an XGBoost classifier as a baseline model.

    Args:
        X_train: Feature matrix
        y_train: Binary target
        params: Optional dict of XGBoost hyperparameters

    Returns:
        Fitted XGBoost model
    """
    if not _HAS_XGB:
        raise ImportError("XGBoost is required for train_xgboost()")

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    default_params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": pos_weight,
        "use_label_encoder": False,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    logger.info(f"XGBoost CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    logger.info("XGBoost training complete")
    return model


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    Trains an L2-regularized Logistic Regression as an interpretable baseline.

    Args:
        X_train: Feature matrix (should be normalized upstream)
        y_train: Binary target

    Returns:
        Fitted LogisticRegression model
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=0.1,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    logger.info(f"Logistic Regression CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete")
    return model


def evaluate_all_models(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluates all trained models and returns a comparison table.

    Args:
        models_dict: Dict of {model_name: fitted_model}
        X_test: Test feature matrix
        y_test: Test target
        threshold: Decision threshold for binary classification

    Returns:
        DataFrame with performance metrics per model
    """
    records = []
    for name, model in models_dict.items():
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            records.append(
                {
                    "model": name,
                    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
                    "avg_precision": round(average_precision_score(y_test, y_proba), 4),
                    "accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                    "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                    "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                    "brier_score": round(brier_score_loss(y_test, y_proba), 4),
                    "approval_rate": round(y_pred.mean(), 4),
                    "n_test": len(y_test),
                }
            )
            logger.info(f"{name}: AUC={records[-1]['roc_auc']:.4f}, F1={records[-1]['f1']:.4f}")
        except Exception as exc:
            logger.error(f"Evaluation failed for {name}: {exc}")

    return pd.DataFrame(records).sort_values("roc_auc", ascending=False)


def hyperparameter_tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
) -> Dict:
    """
    Hyperparameter tuning for LightGBM using Optuna (or manual grid if unavailable).

    Args:
        X_train: Training features
        y_train: Binary target
        n_trials: Number of Optuna trials

    Returns:
        Dict of best hyperparameters
    """
    if not _HAS_LGB:
        logger.warning("LightGBM not available. Returning default params.")
        return {}

    if _HAS_OPTUNA:
        return _tune_with_optuna(X_train, y_train, n_trials)
    else:
        return _tune_with_grid(X_train, y_train)


def _tune_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
) -> Dict:
    """Optuna-based tuning for LightGBM."""
    import optuna
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "objective": "binary",
            "metric": "auc",
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        model = LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    logger.info(f"Optuna best AUC: {study.best_value:.4f}, params: {best}")
    return best


def _tune_with_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict:
    """Manual grid search for LightGBM when Optuna is not available."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold, GridSearchCV

    param_grid = {
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 400],
        "min_child_samples": [20, 50],
    }
    base = LGBMClassifier(objective="binary", metric="auc", random_state=42, verbose=-1)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(base, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    logger.info(f"Grid search best AUC: {gs.best_score_:.4f}, params: {gs.best_params_}")
    return gs.best_params_


def save_artifacts(
    model: Any,
    feature_cols: List[str],
    threshold: float,
    model_dir: str,
    model_name: str = "lightgbm",
) -> None:
    """
    Saves model pickle + feature list + threshold to disk.

    Args:
        model: Fitted model object
        feature_cols: List of feature column names
        threshold: Decision threshold
        model_dir: Directory to save artifacts
        model_name: Base name for artifact files
    """
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)

    model_path = path / f"{model_name}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {"feature_cols": feature_cols, "threshold": threshold, "model_name": model_name}
    meta_path = path / f"{model_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Artifacts saved: {model_path}, {meta_path}")


def load_artifacts(
    model_dir: str,
    model_name: str = "lightgbm",
) -> Tuple[Any, List[str], float]:
    """
    Loads model pickle + metadata from disk.

    Args:
        model_dir: Directory containing artifact files
        model_name: Base name for artifact files

    Returns:
        Tuple of (model, feature_cols, threshold)
    """
    path = Path(model_dir)
    model_path = path / f"{model_name}_model.pkl"
    meta_path = path / f"{model_name}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    logger.info(f"Artifacts loaded from {path}")
    return model, meta["feature_cols"], meta["threshold"]
