"""Predictive modeling.

Two supervised models plus utilities:

  * Regression  — predict ``future_value_eur`` (log-target) from player
    features. Reported with MAE / RMSE / R².
  * Classification — predict the ``transfer_success`` proxy label.
    Reported with accuracy / precision / recall / F1 / ROC-AUC.

Models are returned in a single ``ModelBundle`` together with their evaluation
metrics, permutation-importance tables, and a copy of the input frame augmented
with model predictions. The bundle can be persisted with joblib.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engineering import CLASSIFICATION_FEATURES, REGRESSION_FEATURES
from .utils import MODELS_DIR

RANDOM_STATE = 42


@dataclass
class ModelBundle:
    """Container for trained models, metrics and augmented predictions."""

    value_model: object
    success_model: object
    regression_metrics: dict = field(default_factory=dict)
    classification_metrics: dict = field(default_factory=dict)
    regression_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    classification_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def train_value_regressor(df: pd.DataFrame) -> tuple[object, dict, pd.DataFrame]:
    """Train the future-value regressor on a log-transformed target."""
    X = df[REGRESSION_FEATURES]
    y = df["future_value_eur"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # log1p/expm1 target transform stabilizes the wide value range.
    model = TransformedTargetRegressor(
        regressor=RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": _rmse(y_test, preds),
        "r2": float(r2_score(y_test, preds)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    perm = permutation_importance(
        model, X_test, y_test, n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = (
        pd.DataFrame({"feature": REGRESSION_FEATURES,
                      "importance": perm.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return model, metrics, importance


def train_success_classifier(df: pd.DataFrame) -> tuple[object, dict, pd.DataFrame]:
    """Train the transfer-success proxy classifier."""
    X = df[CLASSIFICATION_FEATURES]
    y = df["transfer_success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=4,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate": float(y.mean()),
    }

    perm = permutation_importance(
        model, X_test, y_test, n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = (
        pd.DataFrame({"feature": CLASSIFICATION_FEATURES,
                      "importance": perm.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return model, metrics, importance


def train_all_models(df: pd.DataFrame) -> ModelBundle:
    """Train both models and attach predictions to a copy of ``df``."""
    value_model, reg_metrics, reg_imp = train_value_regressor(df)
    success_model, clf_metrics, clf_imp = train_success_classifier(df)

    augmented = df.copy()
    augmented["predicted_future_value_eur"] = (
        value_model.predict(df[REGRESSION_FEATURES]).clip(min=10_000).round(-3)
    )
    augmented["predicted_growth_pct"] = (
        (augmented["predicted_future_value_eur"] - augmented["market_value_eur"])
        / augmented["market_value_eur"].clip(lower=1) * 100
    ).round(1)
    augmented["transfer_success_proba"] = (
        success_model.predict_proba(df[CLASSIFICATION_FEATURES])[:, 1].round(4)
    )

    return ModelBundle(
        value_model=value_model,
        success_model=success_model,
        regression_metrics=reg_metrics,
        classification_metrics=clf_metrics,
        regression_importance=reg_imp,
        classification_importance=clf_imp,
        predictions=augmented,
    )


def save_bundle(bundle: ModelBundle, path=None) -> str:
    """Persist the model bundle to disk with joblib."""
    path = path or (MODELS_DIR / "model_bundle.joblib")
    joblib.dump(bundle, path)
    return str(path)


def load_bundle(path=None) -> ModelBundle | None:
    """Load a previously saved bundle, or None if it does not exist."""
    path = path or (MODELS_DIR / "model_bundle.joblib")
    try:
        return joblib.load(path)
    except (FileNotFoundError, OSError):
        return None
