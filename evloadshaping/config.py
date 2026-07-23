"""Shared constants: column names and regex patterns for the raw telemetry schema."""

from __future__ import annotations

import re

TIME_COLUMN = "_time"
PV_COLUMN = "uma_adabyron_solarpanels_pvGeneration"
EV_DEMAND_COLUMN = "total_ev_power_demand"

# Shared across XGBoost and PyTorch stability checks: each model type is
# retrained once per seed and reported as mean +/- std, since a single-seed
# number can't be told apart from ordinary training-variance noise.
SEEDS = (42, 43, 44, 45, 46)
BASE_WEATHER_COLUMNS = [
    "uma_adabyron_weatherStation_solarRadiation",
    "uma_adabyron_weatherStation_temperature",
    "uma_adabyron_weatherStation_relativeHumidity",
    "uma_adabyron_weatherStation_windSpeed",
    PV_COLUMN,
]
CHARGER_POWER_PATTERN = re.compile(r"^uma_adabyron_EVCharger-(\d+)_real_power_sum$")
CHARGER_CONNECTED_PATTERN = re.compile(
    r"^uma_adabyron_EVCharger-(\d+)_vehicle_connected$"
)

# Non-cyclical, non-lagged features: the subset used by the ablation "base"
# model to quantify how much the lag/rolling engineering (features.build_features)
# actually contributes over the raw current-timestep signals.
BASE_ONLY_FEATURE_COLUMNS = [
    "uma_adabyron_weatherStation_solarRadiation",
    "uma_adabyron_weatherStation_temperature",
    "uma_adabyron_weatherStation_relativeHumidity",
    "uma_adabyron_weatherStation_windSpeed",
    PV_COLUMN,
    EV_DEMAND_COLUMN,
    "total_evs_connected",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]
