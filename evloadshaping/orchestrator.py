"""Deterministic edge orchestrator: forecast-driven load-shaping decisions."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


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


def orchestrator_event_log(
    predicted_pv: np.ndarray,
    predicted_ev: np.ndarray,
    actual_pv: np.ndarray,
    actual_ev: np.ndarray,
    evs_connected: np.ndarray,
    index: pd.DatetimeIndex,
    grid_limit: float,
) -> pd.DataFrame:
    """Per-event detail for every interval where the deterministic rule throttles.

    The aggregate throttle rate (summarize_orchestration) is too sparse at
    realistic grid limits (Section VI-D) to say anything about control
    quality -- whether throttle magnitudes are sensible, or whether the
    controller is reacting to real supply/demand mismatches rather than
    forecast noise. This logs every triggering interval individually,
    including a "genuine_risk" flag computed from the *actual* (not
    forecasted) PV/demand values, so each event can be checked rather than
    only counted.
    """
    available_capacity = predicted_pv + grid_limit
    deficit = np.maximum(0.0, predicted_ev - available_capacity)
    connected = np.rint(evs_connected).astype(int)
    throttle_mask = (deficit > 0) & (connected > 0)

    actual_deficit = np.maximum(0.0, actual_ev - (actual_pv + grid_limit))

    with np.errstate(divide="ignore", invalid="ignore"):
        reduction_per_ev = np.where(
            throttle_mask, deficit / np.maximum(connected, 1), 0.0
        )

    frame = pd.DataFrame(
        {
            "timestamp": index,
            "evs_connected": connected,
            "predicted_pv_w": predicted_pv,
            "predicted_ev_w": predicted_ev,
            "actual_pv_w": actual_pv,
            "actual_ev_w": actual_ev,
            "deficit_w": deficit,
            "reduction_per_ev_w": reduction_per_ev,
            "actual_deficit_w": actual_deficit,
            "genuine_risk": actual_deficit > 0,
        }
    )
    return frame[throttle_mask].reset_index(drop=True)


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
