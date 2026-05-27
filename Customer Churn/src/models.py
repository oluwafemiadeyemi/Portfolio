"""
Churn Classification Models.
XGBoost + LightGBM + Logistic Regression ensemble with full evaluation suite.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("lightgbm not installed.")


def _build_candidates(X_train: pd.DataFrame, y_train: pd.Series) -> list[tuple[str, Any]]:
    """Build list of (name, fitted_model) candidates."""
    candidates: list[tuple[str, Any]] = []
    scale_pos = max(1, int((y_train == 0).sum() / max(1, (y_train == 1).sum())))

    # XGBoost
    if XGB_AVAILABLE:
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train, verbose=False)
        candidates.append(("XGBoost", xgb_model))
        logger.info("XGBoost fitted.")

    # LightGBM
    if LGB_AVAILABLE:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train)
        candidates.append(("LightGBM", lgb_model))
        logger.info("LightGBM fitted.")

    # Logistic Regression (always available as fallback)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
    lr_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=500,
        random_state=42,
        n_jobs=-1,
    )
    lr_model.fit(X_train_scaled, y_train)

    # Wrap LR with scaler
    class ScaledLR:
        def __init__(self, lr, sc):
            self.lr_ = lr
            self.scaler_ = sc
            self.name = "LogisticRegression"

        def predict_proba(self, X):
            return self.lr_.predict_proba(self.scaler_.transform(pd.DataFrame(X).fillna(0)))

        def predict(self, X):
            return self.lr_.predict(self.scaler_.transform(pd.DataFrame(X).fillna(0)))

        def get_params(self, deep=True):
            return self.lr_.get_params(deep)

    candidates.append(("LogisticRegression", ScaledLR(lr_model, scaler)))
    logger.info("LogisticRegression fitted.")

    return candidates


def train_churn_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    eval_set: Optional[tuple] = None,
) -> tuple[Any, dict[str, float]]:
    """
    Train XGBoost + LightGBM + LogisticRegression.
    Returns best model by AUC-PR and comparison metrics.
    """
    logger.info(f"Training churn classifiers on {len(X_train):,} samples, {X_train.shape[1]} features...")
    X_train_clean = X_train.fillna(0).replace([np.inf, -np.inf], 0)

    candidates = _build_candidates(X_train_clean, y_train)

    if eval_set is not None:
        X_eval, y_eval = eval_set
        X_eval_clean = pd.DataFrame(X_eval).fillna(0).replace([np.inf, -np.inf], 0)
        best_model = None
        best_auc_pr = -1.0
        comparison: dict[str, dict] = {}

        for name, model in candidates:
            try:
                proba = model.predict_proba(X_eval_clean)[:, 1]
                ap = average_precision_score(y_eval, proba)
                comparison[name] = {"auc_pr": round(ap, 4)}
                if ap > best_auc_pr:
                    best_auc_pr = ap
                    best_model = model
                logger.info(f"  {name}: AUC-PR={ap:.4f}")
            except Exception as exc:
                logger.warning(f"  {name} eval failed: {exc}")

        logger.info(f"Best model: {best_model.__class__.__name__} (AUC-PR={best_auc_pr:.4f})")
        return best_model, comparison

    else:
        # Return first available (prefer XGBoost > LightGBM > LR)
        best_model = candidates[0][1] if candidates else None
        comparison = {name: {} for name, _ in candidates}
        return best_model, comparison


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute full evaluation metrics: AUC-ROC, AUC-PR, F1, Precision, Recall.
    """
    X_clean = X_test.fillna(0).replace([np.inf, -np.inf], 0)
    proba = model.predict_proba(X_clean)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics: dict[str, float] = {}

    try:
        metrics["auc_roc"] = round(roc_auc_score(y_test, proba), 4)
    except Exception:
        metrics["auc_roc"] = 0.0

    try:
        metrics["auc_pr"] = round(average_precision_score(y_test, proba), 4)
    except Exception:
        metrics["auc_pr"] = 0.0

    try:
        metrics["f1"] = round(f1_score(y_test, pred, zero_division=0), 4)
    except Exception:
        metrics["f1"] = 0.0

    try:
        metrics["precision"] = round(precision_score(y_test, pred, zero_division=0), 4)
    except Exception:
        metrics["precision"] = 0.0

    try:
        metrics["recall"] = round(recall_score(y_test, pred, zero_division=0), 4)
    except Exception:
        metrics["recall"] = 0.0

    metrics["threshold"] = threshold
    metrics["n_test"] = len(y_test)
    metrics["positive_rate"] = round(float(y_test.mean()), 4)
    metrics["predicted_positive_rate"] = round(float(pred.mean()), 4)

    logger.info(
        f"Evaluation: AUC-ROC={metrics['auc_roc']}, AUC-PR={metrics['auc_pr']}, "
        f"F1={metrics['f1']}, P={metrics['precision']}, R={metrics['recall']}"
    )
    return metrics


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> dict[str, Any]:
    """Stratified K-Fold cross-validation returning AUC-ROC and AUC-PR."""
    X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    results: dict[str, Any] = {"cv_folds": cv}

    # AUC-ROC
    try:
        scores_roc = cross_val_score(
            model, X_clean, y, cv=skf, scoring="roc_auc", n_jobs=-1
        )
        results["cv_auc_roc_mean"] = round(float(scores_roc.mean()), 4)
        results["cv_auc_roc_std"] = round(float(scores_roc.std()), 4)
        results["cv_auc_roc_scores"] = [round(s, 4) for s in scores_roc.tolist()]
    except Exception as exc:
        logger.warning(f"CV AUC-ROC failed: {exc}")
        results["cv_auc_roc_mean"] = 0.0
        results["cv_auc_roc_std"] = 0.0

    # AUC-PR
    try:
        scores_pr = cross_val_score(
            model, X_clean, y, cv=skf, scoring="average_precision", n_jobs=-1
        )
        results["cv_auc_pr_mean"] = round(float(scores_pr.mean()), 4)
        results["cv_auc_pr_std"] = round(float(scores_pr.std()), 4)
        results["cv_auc_pr_scores"] = [round(s, 4) for s in scores_pr.tolist()]
    except Exception as exc:
        logger.warning(f"CV AUC-PR failed: {exc}")
        results["cv_auc_pr_mean"] = 0.0
        results["cv_auc_pr_std"] = 0.0

    logger.info(
        f"CV Results ({cv}-fold): AUC-ROC={results.get('cv_auc_roc_mean', 0):.4f} "
        f"± {results.get('cv_auc_roc_std', 0):.4f}"
    )
    return results


def save_artifacts(
    churn_model: Any,
    clv_artifacts: dict[str, Any],
    survival_model: Any,
    model_dir: str | Path,
    extra: Optional[dict] = None,
) -> None:
    """Save all model artifacts to disk using joblib."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(churn_model, model_dir / "churn_model.pkl")
    joblib.dump(clv_artifacts, model_dir / "clv_artifacts.pkl")
    joblib.dump(survival_model, model_dir / "survival_model.pkl")

    if extra:
        joblib.dump(extra, model_dir / "extra_artifacts.pkl")

    logger.info(f"Artifacts saved to {model_dir}")


def load_artifacts(model_dir: str | Path) -> dict[str, Any]:
    """Load all saved model artifacts from disk."""
    model_dir = Path(model_dir)
    artifacts: dict[str, Any] = {}

    for fname, key in [
        ("churn_model.pkl", "churn_model"),
        ("clv_artifacts.pkl", "clv_artifacts"),
        ("survival_model.pkl", "survival_model"),
        ("extra_artifacts.pkl", "extra"),
    ]:
        p = model_dir / fname
        if p.exists():
            artifacts[key] = joblib.load(p)
            logger.info(f"Loaded {fname}")
        else:
            logger.warning(f"{fname} not found in {model_dir}")

    return artifacts
