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
    buckets to match the paper's Table II (00-06/06-12/12-18/18-00); the
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


def hourly_error_breakdown_seeds(
    test_index: pd.DatetimeIndex,
    y_pv_test: pd.Series,
    pv_prediction_sets: list[np.ndarray],
    y_ev_test: pd.Series,
    ev_prediction_sets: list[np.ndarray],
    bucket_hours: int = 6,
) -> pd.DataFrame:
    """Seed-averaged version of hourly_error_breakdown.

    Takes one prediction array per seed for each target and reports the mean
    bucket MAE with its across-seed std, so the published table carries no
    single-run number. Bucket boundaries and sample counts are identical
    across seeds, so only the error columns are aggregated.
    """
    per_seed = [
        hourly_error_breakdown(
            test_index, y_pv_test, pv_predictions, y_ev_test, ev_predictions, bucket_hours
        )
        for pv_predictions, ev_predictions in zip(pv_prediction_sets, ev_prediction_sets)
    ]
    combined = per_seed[0][["hour_bucket", "hour_range", "n"]].copy()
    for column in ("pv_mae", "ev_mae"):
        stacked = np.vstack([frame[column].to_numpy() for frame in per_seed])
        combined[column] = stacked.mean(axis=0)
        combined[f"{column}_std"] = (
            stacked.std(axis=0, ddof=1) if len(per_seed) > 1 else 0.0
        )
    return combined[
        ["hour_bucket", "hour_range", "pv_mae", "pv_mae_std", "ev_mae", "ev_mae_std", "n"]
    ]


def select_representative_window(
    test_index: pd.DatetimeIndex, actual_ev: pd.Series, window_days: int = 14
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Pick the contiguous window whose peak EV demand is closest to typical.

    Centering on the single largest demand spike would overstate worst-case
    error and misrepresent day-to-day operation, so this selects the span
    whose peak daily EV demand is nearest the median peak across all such
    spans. Computed from the data rather than hard-coded, so it generalizes to
    any dataset or split. Shared by plotting.plot_forecast_zoom and by
    window_error_metrics, which must describe the very window that is plotted.
    """
    daily_peak = pd.Series(actual_ev.to_numpy(), index=test_index).resample("1D").max()
    # Clamp to the available span so a short test split (e.g. from --max-rows
    # during local development) degrades to "the whole available span"
    # instead of an empty rolling window and a crash in idxmin() below.
    window_days = max(1, min(window_days, len(daily_peak)))
    rolling_peak = daily_peak.rolling(window=window_days, min_periods=window_days).max()
    rolling_peak = rolling_peak.dropna()
    representative_end_day = (rolling_peak - daily_peak.median()).abs().idxmin()

    window_end = min(test_index.max(), representative_end_day + pd.Timedelta(days=1))
    window_start = max(test_index.min(), window_end - pd.Timedelta(days=window_days))
    return window_start, window_end


def window_error_metrics(
    test_index: pd.DatetimeIndex,
    y_pv_test: pd.Series,
    pv_prediction_sets: list[np.ndarray],
    y_ev_test: pd.Series,
    ev_prediction_sets: list[np.ndarray],
    window_days: int = 14,
) -> dict[str, float | str]:
    """Seed-averaged forecast error inside the window Fig. 1 plots.

    The paper contrasts this window's error against the full horizon to show
    that the aggregate EV metric is dominated by rare high-demand events. That
    comparison was previously computed by hand outside the pipeline, so it is
    measured here instead, on the same window the figure selects.
    """
    window_start, window_end = select_representative_window(
        test_index, y_ev_test, window_days=window_days
    )
    mask = (test_index >= window_start) & (test_index < window_end)

    metrics: dict[str, float | str] = {
        "window_start": str(window_start),
        # Inclusive last plotted day, which is what the figure title shows.
        "window_end": str(window_end - pd.Timedelta(days=1)),
        "window_days": int(window_days),
        "n_intervals": int(mask.sum()),
    }
    for target, y_test, prediction_sets in (
        ("pv", y_pv_test, pv_prediction_sets),
        ("ev", y_ev_test, ev_prediction_sets),
    ):
        actual = y_test.to_numpy()[mask]
        per_seed = np.array(
            [
                np.abs(actual - np.asarray(predictions)[mask]).mean()
                for predictions in prediction_sets
            ]
        )
        metrics[f"{target}_mae"] = float(per_seed.mean())
        metrics[f"{target}_mae_std"] = float(
            per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0
        )
        metrics[f"{target}_peak_actual_w"] = float(actual.max())
        metrics[f"{target}_peak_actual_horizon_w"] = float(y_test.to_numpy().max())
    return metrics


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
