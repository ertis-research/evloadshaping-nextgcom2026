"""XGBoost forecasters, baseline comparisons, and feature ablation."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import BASE_ONLY_FEATURE_COLUMNS, EV_DEMAND_COLUMN, PV_COLUMN, SEEDS


def train_model(
    x_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    return {"mae": float(mae), "rmse": rmse}


def aggregate_seeds(per_seed_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Collapse a list of per-seed evaluate_model outputs into mean +/- std.

    Reports mean under the original "mae"/"rmse" keys (so existing CSV/print
    consumers keep working unchanged) plus "*_std" across SEEDS, since a
    single-seed number is otherwise indistinguishable from run-to-run
    training variance rather than a real effect.
    """
    maes = np.array([m["mae"] for m in per_seed_metrics])
    rmses = np.array([m["rmse"] for m in per_seed_metrics])
    return {
        "mae": float(maes.mean()),
        "mae_std": float(maes.std(ddof=1)),
        "rmse": float(rmses.mean()),
        "rmse_std": float(rmses.std(ddof=1)),
    }


def evaluate_xgboost_seeds(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, float]:
    """Retrain XGBoost once per seed and report mean +/- std MAE/RMSE.

    subsample=0.9 and colsample_bytree=0.9 make XGBoost's random_state
    control real stochastic row/column sampling during training, not just an
    arbitrary tie-break -- so, like the PyTorch comparison, a single seed's
    result can't be told apart from ordinary training variance without this.
    """
    per_seed = []
    for seed in seeds:
        model = train_model(x_train, y_train, random_state=seed)
        per_seed.append(evaluate_model(y_test, model.predict(x_test)))
    return aggregate_seeds(per_seed)


def persistence_predictions(x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The naive t+1 forecast: carry forward the current-timestep observation.

    Shared by the baseline comparison and by the orchestrator counterfactual
    (Section VI-C), which reruns the edge decision on persistence forecasts
    to quantify what XGBoost actually buys the control loop.
    """
    return (
        x_test[PV_COLUMN].to_numpy(),
        x_test[EV_DEMAND_COLUMN].to_numpy(),
    )


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
    persistence_pv, persistence_ev = persistence_predictions(x_test)

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


def benchmark_inference_latency(
    model_pv: xgb.XGBRegressor,
    model_ev: xgb.XGBRegressor,
    x_test: pd.DataFrame,
    n_samples: int = 500,
) -> dict[str, float]:
    """Time single-sample inference for both targets, the recurring edge workload.

    Training happens once; the deployed edge service only ever calls
    predict() on one fresh 15-minute sample at a time. Times n_samples
    individual (pv, ev) prediction pairs -- not a batch predict, which would
    understate the per-interval latency a real deployment actually sees --
    and reports the mean. This is the only place this number is measured:
    the paper's implementation-details latency claim previously existed only
    as prose, recomputed by hand and never checked into the repository it
    points readers to.
    """
    n_samples = min(n_samples, len(x_test))
    rows = x_test.iloc[:n_samples]

    latencies_ms = []
    for i in range(n_samples):
        row = rows.iloc[[i]]
        t0 = time.perf_counter()
        model_pv.predict(row)
        model_ev.predict(row)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms = np.array(latencies_ms)
    return {
        "mean_ms": float(latencies_ms.mean()),
        "std_ms": float(latencies_ms.std(ddof=1)),
        "n_samples": n_samples,
    }
