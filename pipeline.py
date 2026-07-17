from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TIME_COLUMN = "_time"
BASE_WEATHER_COLUMNS = [
    "uma_adabyron_weatherStation_solarRadiation",
    "uma_adabyron_weatherStation_temperature",
    "uma_adabyron_weatherStation_relativeHumidity",
    "uma_adabyron_weatherStation_windSpeed",
    "uma_adabyron_solarpanels_pvGeneration",
]
CHARGER_POWER_PATTERN = re.compile(r"^uma_adabyron_EVCharger-(\d+)_real_power_sum$")
CHARGER_CONNECTED_PATTERN = re.compile(
    r"^uma_adabyron_EVCharger-(\d+)_vehicle_connected$"
)

# Non-cyclical, non-lagged features: the subset used by the ablation "base"
# model to quantify how much the lag/rolling engineering (build_features)
# actually contributes over the raw current-timestep signals.
BASE_ONLY_FEATURE_COLUMNS = [
    "uma_adabyron_weatherStation_solarRadiation",
    "uma_adabyron_weatherStation_temperature",
    "uma_adabyron_weatherStation_relativeHumidity",
    "uma_adabyron_weatherStation_windSpeed",
    "uma_adabyron_solarpanels_pvGeneration",
    "total_ev_power_demand",
    "total_evs_connected",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train edge load-shaping models for EV charging."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/20260514_uma_adabyron_data.csv"),
        help="Path to the raw UMA Adabyron CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for metrics, plots, and model artifacts.",
    )
    parser.add_argument(
        "--grid-limit",
        type=float,
        default=5000.0,
        help="Safe import capacity available from the grid in watts.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for faster local experiments.",
    )
    return parser.parse_args()


def discover_chargers(columns: Iterable[str]) -> list[int]:
    charger_ids = set()
    for column in columns:
        power_match = CHARGER_POWER_PATTERN.match(column)
        connected_match = CHARGER_CONNECTED_PATTERN.match(column)
        if power_match:
            charger_ids.add(int(power_match.group(1)))
        if connected_match:
            charger_ids.add(int(connected_match.group(1)))
    return sorted(charger_ids)


def select_columns(all_columns: Iterable[str]) -> tuple[list[int], list[str]]:
    charger_ids = discover_chargers(all_columns)
    if not charger_ids:
        raise ValueError("No EV charger columns were found in the dataset header.")

    charger_power_columns = [
        f"uma_adabyron_EVCharger-{idx}_real_power_sum" for idx in charger_ids
    ]
    charger_connected_columns = [
        f"uma_adabyron_EVCharger-{idx}_vehicle_connected" for idx in charger_ids
    ]

    required_columns = [
        TIME_COLUMN,
        *BASE_WEATHER_COLUMNS,
        *charger_power_columns,
        *charger_connected_columns,
    ]
    return charger_ids, [
        column for column in required_columns if column in set(all_columns)
    ]


def load_raw_data(
    data_path: Path, max_rows: int | None = None
) -> tuple[pd.DataFrame, list[int]]:
    header = pd.read_csv(data_path, nrows=0)
    charger_ids, usecols = select_columns(header.columns)

    if TIME_COLUMN not in usecols:
        raise ValueError(f"Missing required time column: {TIME_COLUMN}")

    frame = pd.read_csv(
        data_path,
        usecols=usecols,
        nrows=max_rows,
        low_memory=False,
    )
    # The raw timestamps mix formats (some rows carry fractional seconds) and
    # include UTC offsets, so read_csv's parse_dates leaves the column as object
    # dtype. Convert explicitly and normalise to a tz-naive UTC DatetimeIndex.
    frame[TIME_COLUMN] = pd.to_datetime(
        frame[TIME_COLUMN], utc=True, format="mixed"
    ).dt.tz_localize(None)
    frame = frame.sort_values(TIME_COLUMN).set_index(TIME_COLUMN)
    return frame, charger_ids


def build_features(frame: pd.DataFrame, charger_ids: list[int]) -> pd.DataFrame:
    power_columns = [
        f"uma_adabyron_EVCharger-{idx}_real_power_sum" for idx in charger_ids
    ]
    connected_columns = [
        f"uma_adabyron_EVCharger-{idx}_vehicle_connected" for idx in charger_ids
    ]

    for column in power_columns + connected_columns:
        if column not in frame.columns:
            frame[column] = 0.0

    frame["total_ev_power_demand"] = frame[power_columns].fillna(0).sum(axis=1)
    frame["total_evs_connected"] = frame[connected_columns].fillna(0).sum(axis=1)

    resample_map: dict[str, str] = {}
    for column in BASE_WEATHER_COLUMNS + ["total_ev_power_demand"]:
        if column in frame.columns:
            resample_map[column] = "mean"
    resample_map["total_evs_connected"] = "max"

    resampled = frame.resample("15min").agg(resample_map).ffill()

    resampled["hour_sin"] = np.sin(2 * np.pi * resampled.index.hour / 24.0)
    resampled["hour_cos"] = np.cos(2 * np.pi * resampled.index.hour / 24.0)
    resampled["day_of_week_sin"] = np.sin(2 * np.pi * resampled.index.dayofweek / 7.0)
    resampled["day_of_week_cos"] = np.cos(2 * np.pi * resampled.index.dayofweek / 7.0)

    for lag in (1, 2, 4):
        resampled[f"pv_lag_{lag}"] = resampled[
            "uma_adabyron_solarpanels_pvGeneration"
        ].shift(lag)
        resampled[f"ev_demand_lag_{lag}"] = resampled["total_ev_power_demand"].shift(
            lag
        )
        resampled[f"connected_lag_{lag}"] = resampled["total_evs_connected"].shift(lag)

    resampled["pv_roll_mean_1h"] = (
        resampled["uma_adabyron_solarpanels_pvGeneration"].rolling(4).mean()
    )
    resampled["ev_roll_mean_1h"] = resampled["total_ev_power_demand"].rolling(4).mean()

    resampled["target_pv_next_15m"] = resampled[
        "uma_adabyron_solarpanels_pvGeneration"
    ].shift(-1)
    resampled["target_ev_next_15m"] = resampled["total_ev_power_demand"].shift(-1)

    return resampled.dropna()


def split_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    feature_columns = [
        column
        for column in frame.columns
        if column not in {"target_pv_next_15m", "target_ev_next_15m"}
    ]
    features = frame[feature_columns]
    target_pv = frame["target_pv_next_15m"]
    target_ev = frame["target_ev_next_15m"]

    split_index = int(len(frame) * 0.8)
    x_train, x_test = features.iloc[:split_index], features.iloc[split_index:]
    y_pv_train, y_pv_test = target_pv.iloc[:split_index], target_pv.iloc[split_index:]
    y_ev_train, y_ev_test = target_ev.iloc[:split_index], target_ev.iloc[split_index:]

    return x_train, x_test, y_pv_train, y_pv_test, y_ev_train, y_ev_test


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
    (the naive forecast any deployed system must beat). Linear regression uses
    the same engineered feature set as XGBoost to isolate how much of the
    accuracy comes from a non-linear model versus the features themselves.
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


def grid_limit_sensitivity(
    predicted_pv: np.ndarray,
    predicted_ev: np.ndarray,
    evs_connected: np.ndarray,
    grid_limits: Iterable[float],
) -> pd.DataFrame:
    """Sweep the safe grid import limit and record the resulting throttle rate."""
    rows = []
    for grid_limit in grid_limits:
        stats = summarize_orchestration(
            predicted_pv, predicted_ev, evs_connected, grid_limit=grid_limit
        )
        rows.append({"grid_limit_w": grid_limit, **stats})
    return pd.DataFrame(rows)


def hourly_error_breakdown(
    test_index: pd.DatetimeIndex,
    y_pv_test: pd.Series,
    pv_predictions: np.ndarray,
    y_ev_test: pd.Series,
    ev_predictions: np.ndarray,
    bucket_hours: int = 3,
) -> pd.DataFrame:
    """Bucket absolute forecast error by time of day.

    Short-horizon PV error is expected to concentrate around sunrise/sunset
    transitions; this table lets the discussion make that claim with numbers
    instead of asserting it from the trace plot alone.
    """
    hours = test_index.hour
    bucket = (hours // bucket_hours) * bucket_hours
    frame = pd.DataFrame(
        {
            "hour_bucket": bucket,
            "pv_abs_error": np.abs(y_pv_test.to_numpy() - pv_predictions),
            "ev_abs_error": np.abs(y_ev_test.to_numpy() - ev_predictions),
        }
    )
    grouped = (
        frame.groupby("hour_bucket")
        .agg(
            pv_mae=("pv_abs_error", "mean"),
            ev_mae=("ev_abs_error", "mean"),
            n=("pv_abs_error", "size"),
        )
        .reset_index()
    )
    grouped["hour_range"] = grouped["hour_bucket"].apply(
        lambda h: f"{h:02d}-{(h + bucket_hours) % 24:02d}"
    )
    return grouped


def load_extended_raw(data_path: Path, charger_ids: list[int]) -> pd.DataFrame:
    """Load per-charger session columns not used by the ML pipeline.

    These are read separately from :func:`load_raw_data` because they support
    the charger utilization analysis rather than the 15-minute forecasting
    feature set, and are far sparser than the columns the forecasters rely on.
    """
    header = pd.read_csv(data_path, nrows=0)
    available = set(header.columns)

    energy_columns = [
        f"uma_adabyron_EVCharger-{idx}_real_energy_delivered_sum"
        for idx in charger_ids
    ]
    charging_columns = [
        f"uma_adabyron_EVCharger-{idx}_charging" for idx in charger_ids
    ]
    columns = [TIME_COLUMN, *energy_columns, *charging_columns]
    usecols = [c for c in columns if c in available]

    frame = pd.read_csv(data_path, usecols=usecols, low_memory=False)
    frame[TIME_COLUMN] = pd.to_datetime(
        frame[TIME_COLUMN], utc=True, format="mixed"
    ).dt.tz_localize(None)
    return frame.sort_values(TIME_COLUMN).set_index(TIME_COLUMN)


def charger_utilization_stats(
    extended_frame: pd.DataFrame, charger_ids: list[int]
) -> pd.DataFrame:
    """Per-charger energy delivered and active-charging share over the full study span.

    Cumulative energy counters can reset (e.g. on firmware updates), so total
    energy is the sum of positive deltas rather than a naive max-min.
    """
    rows = []
    for idx in charger_ids:
        energy_col = f"uma_adabyron_EVCharger-{idx}_real_energy_delivered_sum"
        charging_col = f"uma_adabyron_EVCharger-{idx}_charging"

        energy_series = extended_frame.get(energy_col)
        charging_series = extended_frame.get(charging_col)

        total_energy_wh = 0.0
        if energy_series is not None:
            deltas = energy_series.dropna().diff().dropna()
            total_energy_wh = float(deltas[deltas > 0].sum())

        charging_share = (
            float(charging_series.dropna().mean() * 100)
            if charging_series is not None and charging_series.notna().any()
            else 0.0
        )

        rows.append(
            {
                "charger_id": idx,
                "total_energy_delivered_kwh": total_energy_wh / 1000.0,
                "pct_time_charging": charging_share,
            }
        )
    return pd.DataFrame(rows)


def edge_orchestrator(
    predicted_pv: float,
    predicted_ev: float,
    current_evs_connected: int,
    grid_limit: float,
) -> dict[str, float | int | str]:
    available_capacity = predicted_pv + grid_limit
    if current_evs_connected <= 0:
        return {
            "status": "inspect",
            "message": "High demand forecasted but no EVs are connected. Check sensor integrity.",
            "predicted_pv": predicted_pv,
            "predicted_ev": predicted_ev,
            "current_evs_connected": current_evs_connected,
            "grid_limit": grid_limit,
        }

    if predicted_ev <= available_capacity:
        return {
            "status": "stable",
            "message": "Grid stable. Normal charging operations permitted.",
            "predicted_pv": predicted_pv,
            "predicted_ev": predicted_ev,
            "current_evs_connected": current_evs_connected,
            "grid_limit": grid_limit,
        }

    deficit = predicted_ev - available_capacity
    reduction_per_ev = deficit / current_evs_connected
    return {
        "status": "throttle",
        "message": f"Grid peak anticipated. Throttle {current_evs_connected} EVs by {reduction_per_ev:.2f} W each.",
        "predicted_pv": predicted_pv,
        "predicted_ev": predicted_ev,
        "current_evs_connected": current_evs_connected,
        "grid_limit": grid_limit,
        "deficit_w": float(deficit),
        "reduction_per_ev_w": float(reduction_per_ev),
    }


def summarize_orchestration(
    predicted_pv: np.ndarray,
    predicted_ev: np.ndarray,
    evs_connected: np.ndarray,
    grid_limit: float,
) -> dict[str, float | int]:
    """Aggregate the deterministic edge decisions over the full test horizon.

    Runs the same rule as :func:`edge_orchestrator` across every test interval
    so the paper can report how often a throttle command would fire instead of
    relying on a single, possibly unrepresentative, snapshot.
    """
    available_capacity = predicted_pv + grid_limit
    deficit = np.maximum(0.0, predicted_ev - available_capacity)

    connected = np.rint(evs_connected).astype(int)
    throttle_mask = (deficit > 0) & (connected > 0)
    inspect_mask = (deficit > 0) & (connected <= 0)

    n_total = int(len(predicted_pv))
    n_throttle = int(throttle_mask.sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        reduction_per_ev = np.where(
            throttle_mask, deficit / np.maximum(connected, 1), 0.0
        )

    mean_reduction = (
        float(reduction_per_ev[throttle_mask].mean()) if n_throttle else 0.0
    )
    max_reduction = (
        float(reduction_per_ev[throttle_mask].max()) if n_throttle else 0.0
    )

    return {
        "n_intervals": n_total,
        "n_stable": int(n_total - throttle_mask.sum() - inspect_mask.sum()),
        "n_throttle": n_throttle,
        "n_inspect": int(inspect_mask.sum()),
        "throttle_rate": float(n_throttle / n_total) if n_total else 0.0,
        "mean_reduction_per_ev_w": mean_reduction,
        "max_reduction_per_ev_w": max_reduction,
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


def plot_forecasts(
    test_index: pd.Index,
    actual_pv: pd.Series,
    predicted_pv: np.ndarray,
    actual_ev: pd.Series,
    predicted_ev: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(test_index, actual_pv.values, label="Actual PV", linewidth=1.5)
    axes[0].plot(test_index, predicted_pv, label="Predicted PV", linewidth=1.2)
    axes[0].set_ylabel("W")
    axes[0].set_title("PV generation forecast")
    axes[0].legend(loc="best")

    axes[1].plot(test_index, actual_ev.values, label="Actual EV demand", linewidth=1.5)
    axes[1].plot(test_index, predicted_ev, label="Predicted EV demand", linewidth=1.2)
    axes[1].set_ylabel("W")
    axes[1].set_title("EV charging demand forecast")
    axes[1].legend(loc="best")

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_grid_sensitivity(sensitivity: pd.DataFrame, output_path: Path) -> None:
    figure, ax1 = plt.subplots(figsize=(7, 4.5))

    ax1.plot(
        sensitivity["grid_limit_w"],
        sensitivity["throttle_rate"] * 100,
        marker="o",
        color="tab:blue",
        label="Throttle rate",
    )
    ax1.set_xlabel("Safe grid import limit (W)")
    ax1.set_ylabel("Throttle rate (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(
        sensitivity["grid_limit_w"],
        sensitivity["mean_reduction_per_ev_w"],
        marker="s",
        color="tab:orange",
        label="Mean reduction per EV",
    )
    ax2.set_ylabel("Mean reduction per EV (W)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    figure.suptitle("Orchestrator sensitivity to the safe grid import limit")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_path}")
    raw_frame, charger_ids = load_raw_data(args.data_path, max_rows=args.max_rows)
    print(f"Detected EV chargers: {charger_ids}")

    feature_frame = build_features(raw_frame, charger_ids)
    print(f"Prepared {len(feature_frame):,} 15-minute samples")

    x_train, x_test, y_pv_train, y_pv_test, y_ev_train, y_ev_test = split_data(
        feature_frame
    )
    print(f"Train rows: {len(x_train):,} | Test rows: {len(x_test):,}")

    print("Training XGBoost models...")
    model_pv = train_model(x_train, y_pv_train)
    model_ev = train_model(x_train, y_ev_train)

    pv_predictions = model_pv.predict(x_test)
    ev_predictions = model_ev.predict(x_test)

    pv_metrics = evaluate_model(y_pv_test, pv_predictions)
    ev_metrics = evaluate_model(y_ev_test, ev_predictions)

    print("Evaluation metrics")
    print(f"PV   MAE: {pv_metrics['mae']:.2f} W | RMSE: {pv_metrics['rmse']:.2f} W")
    print(f"EV   MAE: {ev_metrics['mae']:.2f} W | RMSE: {ev_metrics['rmse']:.2f} W")

    feature_names = [column for column in x_train.columns]
    model_pv.save_model(str(args.output_dir / "model_pv.json"))
    model_ev.save_model(str(args.output_dir / "model_ev.json"))
    save_feature_importance(
        model_pv, feature_names, args.output_dir / "feature_importance_pv.csv"
    )
    save_feature_importance(
        model_ev, feature_names, args.output_dir / "feature_importance_ev.csv"
    )

    predictions = pd.DataFrame(
        {
            "actual_pv": y_pv_test.values,
            "predicted_pv": pv_predictions,
            "actual_ev": y_ev_test.values,
            "predicted_ev": ev_predictions,
        },
        index=x_test.index,
    )
    predictions.to_csv(args.output_dir / "test_predictions.csv")

    plot_forecasts(
        test_index=x_test.index,
        actual_pv=y_pv_test,
        predicted_pv=pv_predictions,
        actual_ev=y_ev_test,
        predicted_ev=ev_predictions,
        output_path=args.output_dir / "forecast_plot.png",
    )

    test_state = x_test.iloc[0]
    decision = edge_orchestrator(
        predicted_pv=float(pv_predictions[0]),
        predicted_ev=float(ev_predictions[0]),
        current_evs_connected=int(round(float(test_state["total_evs_connected"]))),
        grid_limit=args.grid_limit,
    )

    orchestration_stats = summarize_orchestration(
        pv_predictions,
        ev_predictions,
        x_test["total_evs_connected"].to_numpy(),
        grid_limit=args.grid_limit,
    )

    print("Computing baselines (persistence, linear regression)...")
    baselines = compute_baselines(
        x_test, y_pv_test, y_ev_test, x_train, y_pv_train, y_ev_train
    )
    pd.DataFrame(
        [
            {"model": "persistence", "target": "pv", **baselines["persistence"]["pv"]},
            {"model": "persistence", "target": "ev", **baselines["persistence"]["ev"]},
            {
                "model": "ridge_regression",
                "target": "pv",
                **baselines["ridge_regression"]["pv"],
            },
            {
                "model": "ridge_regression",
                "target": "ev",
                **baselines["ridge_regression"]["ev"],
            },
            {"model": "xgboost", "target": "pv", **pv_metrics},
            {"model": "xgboost", "target": "ev", **ev_metrics},
        ]
    ).to_csv(args.output_dir / "baseline_comparison.csv", index=False)

    print("Running feature ablation (base features vs. full feature set)...")
    ablation = run_feature_ablation(
        x_train, x_test, y_pv_train, y_pv_test, y_ev_train, y_ev_test
    )
    pd.DataFrame(
        [
            {
                "feature_set": "base_only",
                "target": "pv",
                **ablation["base_features_only"]["pv"],
            },
            {
                "feature_set": "base_only",
                "target": "ev",
                **ablation["base_features_only"]["ev"],
            },
            {"feature_set": "full", "target": "pv", **pv_metrics},
            {"feature_set": "full", "target": "ev", **ev_metrics},
        ]
    ).to_csv(args.output_dir / "ablation_study.csv", index=False)

    print("Sweeping grid-limit sensitivity...")
    sensitivity = grid_limit_sensitivity(
        pv_predictions,
        ev_predictions,
        x_test["total_evs_connected"].to_numpy(),
        grid_limits=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000],
    )
    sensitivity.to_csv(args.output_dir / "grid_sensitivity.csv", index=False)
    plot_grid_sensitivity(sensitivity, args.output_dir / "grid_sensitivity_plot.png")

    print("Computing hourly error breakdown...")
    hourly_errors = hourly_error_breakdown(
        x_test.index, y_pv_test, pv_predictions, y_ev_test, ev_predictions
    )
    hourly_errors.to_csv(args.output_dir / "hourly_error_breakdown.csv", index=False)

    print("Loading per-charger telemetry for utilization analysis...")
    extended_frame = load_extended_raw(args.data_path, charger_ids)
    charger_stats = charger_utilization_stats(extended_frame, charger_ids)
    charger_stats.to_csv(args.output_dir / "charger_utilization.csv", index=False)

    summary = {
        "data_path": str(args.data_path),
        "charger_ids": charger_ids,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "pv_metrics": pv_metrics,
        "ev_metrics": ev_metrics,
        "edge_decision": decision,
        "orchestration_stats": orchestration_stats,
        "baselines": baselines,
        "ablation": ablation,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Edge node simulation")
    print(
        f"Forecasted PV: {pv_predictions[0]:.2f} W | Forecasted EV Demand: {ev_predictions[0]:.2f} W"
    )
    print(f"Orchestrator action: {decision['message']}")
    print(
        "Throttle triggered in "
        f"{orchestration_stats['n_throttle']}/{orchestration_stats['n_intervals']} "
        f"test intervals ({orchestration_stats['throttle_rate'] * 100:.2f}%), "
        f"mean reduction {orchestration_stats['mean_reduction_per_ev_w']:.2f} W/EV"
    )


if __name__ == "__main__":
    main()
