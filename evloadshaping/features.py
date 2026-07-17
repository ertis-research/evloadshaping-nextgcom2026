"""Resampling, feature engineering, and the chronological train/test split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASE_WEATHER_COLUMNS


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
