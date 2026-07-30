"""Column-scoped loading of the raw UMA Adabyron telemetry CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import (
    CHARGER_CONNECTED_PATTERN,
    CHARGER_POWER_PATTERN,
    BASE_WEATHER_COLUMNS,
    TIME_COLUMN,
)


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
        low_memory=False,
    )
    # The raw timestamps mix formats (some rows carry fractional seconds) and
    # include UTC offsets, so read_csv's parse_dates leaves the column as object
    # dtype. Convert explicitly and normalise to a tz-naive UTC DatetimeIndex.
    frame[TIME_COLUMN] = pd.to_datetime(
        frame[TIME_COLUMN], utc=True, format="mixed"
    ).dt.tz_localize(None)
    frame = frame.sort_values(TIME_COLUMN).set_index(TIME_COLUMN)
    if max_rows is not None:
        # Take the most recent max_rows samples, not the first max_rows: on
        # this site's telemetry, the earliest rows predate PV/charger
        # installation (100% NaN on those columns), so a naive head slice
        # gets entirely dropped by build_features's dropna() and crashes
        # downstream with zero training rows.
        frame = frame.tail(max_rows)
    return frame, charger_ids


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
