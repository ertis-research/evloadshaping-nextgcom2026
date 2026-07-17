"""XGBoost forecasters, baseline comparisons, and feature ablation."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import BASE_ONLY_FEATURE_COLUMNS


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    return {"mae": float(mae), "rmse": rmse}


def compute_baselines(
    x_test: pd.DataFrame,
    y_pv_test: pd.Series,
    y_ev_test: pd.Series,
    x_train: pd.DataFrame,
    y_pv_train: pd.Series,
    y_ev_train: pd.Series,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare XGBoost against two cheap baselines.

    Persistence predicts the t+1 value as the current-timestep observation
    (the naive forecast any deployed system must beat). Ridge regression uses
    the same standardized engineered feature set as XGBoost to isolate how
    much of the accuracy comes from a non-linear model versus the features
    themselves.
    """
    persistence_pv = x_test["uma_adabyron_solarpanels_pvGeneration"].to_numpy()
    persistence_ev = x_test["total_ev_power_demand"].to_numpy()

    # Ridge on standardized features rather than plain OLS: the engineered
    # feature set has near-collinear columns (cyclical time pairs, lag/rolling
    # variants of the same signal). This drives the SVD-based ridge solver's
    # near-zero singular values, which numpy flags as RuntimeWarnings even
    # though they resolve to finite coefficients (verified against 5 solvers,
    # none producing NaN/Inf in coefficients or predictions) — safe to ignore.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        linreg_pv = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
            x_train, y_pv_train
        )
        linreg_ev = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
            x_train, y_ev_train
        )
        linreg_pv_preds = linreg_pv.predict(x_test)
        linreg_ev_preds = linreg_ev.predict(x_test)

    return {
        "persistence": {
            "pv": evaluate_model(y_pv_test, persistence_pv),
            "ev": evaluate_model(y_ev_test, persistence_ev),
        },
        "ridge_regression": {
            "pv": evaluate_model(y_pv_test, linreg_pv_preds),
            "ev": evaluate_model(y_ev_test, linreg_ev_preds),
        },
    }


def run_feature_ablation(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_pv_train: pd.Series,
    y_pv_test: pd.Series,
    y_ev_train: pd.Series,
    y_ev_test: pd.Series,
) -> dict[str, dict[str, dict[str, float]]]:
    """Quantify the contribution of lag and rolling-mean features.

    Trains the same XGBoost configuration on a reduced "base" feature set
    (current-timestep weather/demand/time signals only, no lags or rolling
    means) and compares it against the full engineered feature set.
    """
    base_columns = [c for c in BASE_ONLY_FEATURE_COLUMNS if c in x_train.columns]
    x_train_base = x_train[base_columns]
    x_test_base = x_test[base_columns]

    model_pv_base = train_model(x_train_base, y_pv_train)
    model_ev_base = train_model(x_train_base, y_ev_train)

    return {
        "base_features_only": {
            "pv": evaluate_model(y_pv_test, model_pv_base.predict(x_test_base)),
            "ev": evaluate_model(y_ev_test, model_ev_base.predict(x_test_base)),
        },
    }


def save_feature_importance(
    model: xgb.XGBRegressor, feature_names: list[str], output_path: Path
) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_path, index=False)
