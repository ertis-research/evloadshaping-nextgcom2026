"""XGBoost forecasters, baseline comparisons, and feature ablation."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
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
    """Collapse a list of per-seed metric dicts into mean +/- std per key.

    Generic over whatever keys each per-seed dict carries (mae/rmse, and
    optionally n_parameters for architectures like XGBoost whose tree
    structure -- and therefore node count -- varies slightly per seed under
    subsample/colsample_bytree). Reports mean under the original key (so
    existing CSV/print consumers keep working unchanged) plus "*_std" across
    SEEDS, since a single-seed number is otherwise indistinguishable from
    run-to-run training variance rather than a real effect.
    """
    aggregated: dict[str, float] = {}
    for key in per_seed_metrics[0]:
        values = np.array([m[key] for m in per_seed_metrics], dtype=float)
        aggregated[key] = float(values.mean())
        if len(values) > 1:
            aggregated[f"{key}_std"] = float(values.std(ddof=1))
    return aggregated


def count_xgboost_parameters(model: xgb.XGBRegressor) -> int:
    """Count total nodes (splits + leaves) across all trees in a fitted model.

    The tree-ensemble analogue of a neural net's parameter count: each split
    node stores a feature index and threshold, each leaf stores a value, so
    total node count is the natural, serialization-format-independent measure
    of model capacity to compare against nn.Module parameter counts (see
    torch_models.count_torch_parameters). File size on disk is not a
    substitute for this -- XGBoost's save_model formats also embed per-node
    training bookkeeping (loss_changes, sum_hessian) that inflates KB well
    beyond the actual parameter count.
    """
    total = 0
    for tree_json in model.get_booster().get_dump(dump_format="json"):
        stack = [json.loads(tree_json)]
        while stack:
            node = stack.pop()
            total += 1
            stack.extend(node.get("children", []))
    return total


def count_ridge_parameters(pipeline: Pipeline) -> int:
    """Count coefficients + intercept of the Ridge step in a fitted pipeline."""
    ridge = pipeline.named_steps["ridge"]
    return int(np.asarray(ridge.coef_).size + np.asarray(ridge.intercept_).size)


def fit_xgboost_seeds(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    seeds: tuple[int, ...] = SEEDS,
) -> list[dict[str, object]]:
    """Train one model per seed and keep both the model and its test predictions.

    Every seed-averaged quantity the paper reports (accuracy, the hourly error
    breakdown, gain importances, the Fig. 1 window errors) needs the same set
    of fitted models, so they are trained once here and shared rather than
    retrained per analysis.
    """
    fitted: list[dict[str, object]] = []
    for seed in seeds:
        model = train_model(x_train, y_train, random_state=seed)
        fitted.append(
            {"seed": seed, "model": model, "predictions": model.predict(x_test)}
        )
    return fitted


def evaluate_xgboost_seeds(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    seeds: tuple[int, ...] = SEEDS,
    fitted: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    """Retrain XGBoost once per seed and report mean +/- std MAE/RMSE.

    subsample=0.9 and colsample_bytree=0.9 make XGBoost's random_state
    control real stochastic row/column sampling during training, not just an
    arbitrary tie-break -- so, like the PyTorch comparison, a single seed's
    result can't be told apart from ordinary training variance without this.

    Pass ``fitted`` (from fit_xgboost_seeds) to score models that have already
    been trained for another analysis instead of retraining them.
    """
    if fitted is None:
        fitted = fit_xgboost_seeds(x_train, y_train, x_test, seeds=seeds)
    per_seed = []
    for entry in fitted:
        metrics = evaluate_model(y_test, entry["predictions"])
        metrics["n_parameters"] = count_xgboost_parameters(entry["model"])
        per_seed.append(metrics)
    return aggregate_seeds(per_seed)


def persistence_predictions(x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The naive t+1 forecast: carry forward the current-timestep observation.

    Shared by the baseline comparison and by the orchestrator counterfactual
    (Section IV-F), which reruns the edge decision on persistence forecasts
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

    # Ridge on standardized features rather than plain OLS, because the
    # engineered feature set has near-collinear columns (cyclical time pairs,
    # lag/rolling variants of the same signal). This drives the SVD-based ridge
    # solver's near-zero singular values, which numpy flags as RuntimeWarnings
    # even though they resolve to finite coefficients (verified against 5
    # solvers, none producing NaN/Inf in coefficients or predictions), so the
    # warnings are safe to ignore.
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
            # No learned parameters: predictions are just the current-timestep
            # observation carried forward.
            "pv": {**evaluate_model(y_pv_test, persistence_pv), "n_parameters": 0},
            "ev": {**evaluate_model(y_ev_test, persistence_ev), "n_parameters": 0},
        },
        "ridge_regression": {
            "pv": {
                **evaluate_model(y_pv_test, linreg_pv_preds),
                "n_parameters": count_ridge_parameters(linreg_pv),
            },
            "ev": {
                **evaluate_model(y_ev_test, linreg_ev_preds),
                "n_parameters": count_ridge_parameters(linreg_ev),
            },
        },
    }


def run_feature_ablation(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_pv_train: pd.Series,
    y_pv_test: pd.Series,
    y_ev_train: pd.Series,
    y_ev_test: pd.Series,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, dict[str, dict[str, float]]]:
    """Quantify the contribution of lag and rolling-mean features.

    Trains the same XGBoost configuration on a reduced "base" feature set
    (current-timestep weather/demand/time signals only, no lags or rolling
    means) and compares it against the full engineered feature set. Averaged
    over SEEDS, because the full-feature side of the comparison is a 5-seed
    mean and an ablation delta between a mean and a single run would fold
    training variance into what it attributes to the features.
    """
    base_columns = [c for c in BASE_ONLY_FEATURE_COLUMNS if c in x_train.columns]
    x_train_base = x_train[base_columns]
    x_test_base = x_test[base_columns]

    base_metrics = {}
    for target, y_train, y_test in (
        ("pv", y_pv_train, y_pv_test),
        ("ev", y_ev_train, y_ev_test),
    ):
        fitted = fit_xgboost_seeds(x_train_base, y_train, x_test_base, seeds=seeds)
        base_metrics[target] = aggregate_seeds(
            [evaluate_model(y_test, entry["predictions"]) for entry in fitted]
        )

    return {"base_features_only": base_metrics}


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


def seed_averaged_feature_importance(
    fitted: list[dict[str, object]], feature_names: list[str]
) -> pd.DataFrame:
    """Mean +/- std gain importance per feature across the per-seed models.

    Gain importance is a property of the fitted tree structure, so it shifts
    with the seed exactly as the accuracy metrics do; the published table
    reports the mean so no row rests on a single run.
    """
    per_seed = np.vstack(
        [np.asarray(entry["model"].feature_importances_, dtype=float) for entry in fitted]
    )
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": per_seed.mean(axis=0),
            "importance_std": per_seed.std(axis=0, ddof=1)
            if len(fitted) > 1
            else np.zeros(per_seed.shape[1]),
        }
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


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
