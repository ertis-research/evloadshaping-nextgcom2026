"""Descriptive analyses: temporal error breakdown and per-charger utilization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hourly_error_breakdown(
    test_index: pd.DatetimeIndex,
    y_pv_test: pd.Series,
    pv_predictions: np.ndarray,
    y_ev_test: pd.Series,
    ev_predictions: np.ndarray,
    bucket_hours: int = 6,
) -> pd.DataFrame:
    """Bucket absolute forecast error by time of day.

    Short-horizon PV error is expected to concentrate around sunrise/sunset
    transitions; this table lets the discussion make that claim with numbers
    instead of asserting it from the trace plot alone. Defaults to 6-hour
    buckets to match the paper's Table VI (00-06/06-12/12-18/18-00); the
    default had drifted to 3h at some point while the paper's own table
    stayed at 6h, so the CLI/notebook output no longer matched what's
    published until this was caught and fixed.
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
