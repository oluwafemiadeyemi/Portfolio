"""
Model training, evaluation, and anomaly detection for the Brand Intelligence Platform.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentiment Classifier Training
# ---------------------------------------------------------------------------

def train_sentiment_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """Train XGBoost, LightGBM, and LogisticRegression; return best model by F1 macro."""
    logger.info("Training sentiment classifiers on %d samples, %d features", *X_train.shape)

    # Encode labels if strings
    if y_train.dtype == object or isinstance(y_train[0], str):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train)
    else:
        y_encoded = np.array(y_train)
        le = None

    models_to_try = []

    # Logistic Regression (always available)
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",

        random_state=42,
    )
    models_to_try.append(("LogisticRegression", lr))

    # XGBoost
    try:
        from xgboost import XGBClassifier
        unique_classes = len(np.unique(y_encoded))
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        models_to_try.append(("XGBoost", xgb))
    except ImportError:
        logger.warning("XGBoost not installed — skipping")

    # LightGBM
    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        models_to_try.append(("LightGBM", lgbm))
    except ImportError:
        logger.warning("LightGBM not installed — skipping")

    best_model = None
    best_score = -1.0
    best_name = ""

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in models_to_try:
        try:
            logger.info("Cross-validating %s", name)
            scores = cross_val_score(model, X_train, y_encoded, cv=cv, scoring="f1_macro", n_jobs=-1)
            mean_score = float(np.mean(scores))
            logger.info("%s CV F1-macro: %.4f ± %.4f", name, mean_score, np.std(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name
        except Exception as exc:
            logger.warning("Failed cross-validating %s: %s", name, exc)

    if best_model is None:
        best_model = lr
        best_name = "LogisticRegression"

    logger.info("Best model: %s (CV F1-macro=%.4f) — fitting on full training set", best_name, best_score)
    best_model.fit(X_train, y_encoded)

    # Attach label encoder as attribute for inverse-transform at inference
    best_model._label_encoder = le
    best_model._model_name = best_name
    return best_model


# ---------------------------------------------------------------------------
# Per-Aspect Model
# ---------------------------------------------------------------------------

def train_aspect_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    aspect_name: str,
) -> Any:
    """Train a per-aspect sentiment classifier."""
    logger.info("Training aspect model for '%s' on %d samples", aspect_name, len(y_train))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",

        random_state=42,
    )
    model.fit(X_train, y_encoded)
    model._label_encoder = le
    model._aspect_name = aspect_name
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Return accuracy, precision, recall, f1_macro, f1_weighted, report, confusion_matrix."""
    logger.info("Evaluating model on %d test samples", len(y_test))

    le = getattr(model, "_label_encoder", None)
    if le is not None and (y_test.dtype == object or (len(y_test) > 0 and isinstance(y_test[0], str))):
        y_encoded = le.transform(y_test)
    else:
        y_encoded = np.array(y_test)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_encoded, y_pred)
    prec = precision_score(y_encoded, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_encoded, y_pred, average="macro", zero_division=0)
    f1_mac = f1_score(y_encoded, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y_encoded, y_pred, average="weighted", zero_division=0)

    target_names = le.classes_.tolist() if le is not None else None
    report = classification_report(y_encoded, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_encoded, y_pred).tolist()

    result = {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "n_test_samples": len(y_encoded),
    }
    logger.info(
        "Evaluation: accuracy=%.4f, f1_macro=%.4f, f1_weighted=%.4f",
        acc, f1_mac, f1_wt,
    )
    return result


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> Dict[str, Any]:
    """Return CV scores dict with mean and std for multiple metrics."""
    logger.info("Running %d-fold cross-validation", cv)

    le = getattr(model, "_label_encoder", None)
    if le is not None and (y.dtype == object or (len(y) > 0 and isinstance(y[0], str))):
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = np.array(y)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for metric in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]:
        scores = cross_val_score(model, X, y_encoded, cv=skf, scoring=metric, n_jobs=-1)
        results[f"{metric}_mean"] = round(float(np.mean(scores)), 4)
        results[f"{metric}_std"] = round(float(np.std(scores)), 4)
        results[f"{metric}_scores"] = [round(float(s), 4) for s in scores]

    logger.info("CV results: accuracy=%.4f±%.4f, f1_macro=%.4f±%.4f",
                results["accuracy_mean"], results["accuracy_std"],
                results["f1_macro_mean"], results["f1_macro_std"])
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, path: Union[str, Path]) -> None:
    """Save model to disk using joblib."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info("Model saved to %s", save_path)


def load_model(path: Union[str, Path]) -> Any:
    """Load model from disk using joblib."""
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")
    model = joblib.load(load_path)
    logger.info("Model loaded from %s", load_path)
    return model


# ---------------------------------------------------------------------------
# Anomaly / Crisis Detection
# ---------------------------------------------------------------------------

def detect_anomaly(
    sentiment_scores: List[float],
    window: int = 24,
    threshold: float = 0.15,
) -> bool:
    """Return True if latest score drops >threshold below rolling window average."""
    if len(sentiment_scores) < 2:
        return False

    series = np.array(sentiment_scores, dtype=float)
    # Use last `window` elements (excluding the current point)
    lookback = series[-(window + 1):-1]
    if len(lookback) == 0:
        return False

    window_avg = float(np.mean(lookback))
    current = float(series[-1])
    drop = window_avg - current

    is_anomaly = drop > threshold
    if is_anomaly:
        logger.warning(
            "Anomaly detected: current=%.4f, window_avg=%.4f, drop=%.4f (threshold=%.4f)",
            current, window_avg, drop, threshold,
        )
    return is_anomaly


def detect_anomaly_series(
    df: pd.DataFrame,
    score_col: str = "compound_sentiment",
    date_col: str = "date",
    window_hours: int = 24,
    threshold: float = 0.15,
    freq: str = "D",
) -> pd.DataFrame:
    """Compute rolling anomaly flags on a time-indexed sentiment DataFrame."""
    logger.info("Running anomaly detection on time series (window=%d, threshold=%.3f)", window_hours, threshold)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    ts = (
        df.dropna(subset=[date_col])
        .set_index(date_col)[score_col]
        .resample(freq)
        .mean()
        .ffill()
    )

    rolling_mean = ts.shift(1).rolling(window=window_hours, min_periods=1).mean()
    drop = rolling_mean - ts
    ts_df = pd.DataFrame({
        "date": ts.index,
        "avg_sentiment": ts.values,
        "rolling_baseline": rolling_mean.values,
        "drop": drop.values,
        "is_anomaly": drop.values > threshold,
    })
    anomaly_count = int(ts_df["is_anomaly"].sum())
    logger.info("Anomaly detection complete: %d anomalies found", anomaly_count)
    return ts_df
