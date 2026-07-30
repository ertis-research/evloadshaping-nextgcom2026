"""Plotting helpers, shared by the CLI pipeline and the notebooks.

This module does not force a matplotlib backend: the CLI entry point
(``evloadcontrol.cli``) selects the non-interactive ``Agg`` backend for
headless runs, while notebooks keep their own inline backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import select_representative_window


def plot_forecasts(
    test_index: pd.Index,
    actual_pv: pd.Series,
    predicted_pv: np.ndarray,
    actual_ev: pd.Series,
    predicted_ev: np.ndarray,
    output_path: Path | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """Plot actual vs. predicted traces for both targets.

    Predicted is drawn dashed (not just a second color) so the two series stay
    distinguishable if the figure is printed or reviewed in grayscale, and so
    small tracking gaps are visible even where the lines nearly overlap. The
    wide, short aspect ratio is intended for a double-column IEEE figure: it
    reads well at full text width without the height ballooning to match.
    """
    figure, axes = plt.subplots(2, 1, figsize=(16, 6.5), sharex=True)
    axes[0].set_xlim(test_index.min(), test_index.max())

    axes[0].plot(test_index, actual_pv.values, label="Actual PV", linewidth=1.6)
    axes[0].plot(
        test_index, predicted_pv, label="Predicted PV", linewidth=1.3, linestyle="--"
    )
    axes[0].set_ylabel("W")
    axes[0].set_title(f"PV generation forecast{title_suffix}")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(test_index, actual_ev.values, label="Actual EV demand", linewidth=1.6)
    axes[1].plot(
        test_index,
        predicted_ev,
        label="Predicted EV demand",
        linewidth=1.3,
        linestyle="--",
    )
    axes[1].set_ylabel("W")
    axes[1].set_title(f"EV charging demand forecast{title_suffix}")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    figure.autofmt_xdate()
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=200)
    return figure


def plot_forecast_zoom(
    test_index: pd.Index,
    actual_pv: pd.Series,
    predicted_pv: np.ndarray,
    actual_ev: pd.Series,
    predicted_ev: np.ndarray,
    output_path: Path | None = None,
    window_days: int = 14,
) -> plt.Figure:
    """Plot a short, readable window representative of routine operation.

    The full test horizon (months of 15-minute samples) compresses each day to
    a sliver and makes actual/predicted visually indistinguishable, so a short
    window is needed for a readable figure. Centering that window on the
    single largest demand spike would overstate worst-case error and is not
    representative of day-to-day operation, so instead this plots the window
    chosen by analysis.select_representative_window: a typical period rather
    than an outlier event, computed from the data rather than a hard-coded
    date range. The selection lives in analysis so that the window errors the
    paper quotes are measured on exactly the window plotted here.
    """
    predicted_ev = np.asarray(predicted_ev)
    predicted_pv = np.asarray(predicted_pv)

    window_start, window_end = select_representative_window(
        test_index, actual_ev, window_days=window_days
    )
    mask = (test_index >= window_start) & (test_index < window_end)

    return plot_forecasts(
        test_index[mask],
        actual_pv[mask],
        predicted_pv[mask],
        actual_ev[mask],
        predicted_ev[mask],
        output_path=output_path,
        title_suffix=(
            f" ({window_start:%Y-%m-%d} to "
            f"{(window_end - pd.Timedelta(days=1)):%Y-%m-%d})"
        ),
    )


def plot_grid_sensitivity(
    sensitivity: pd.DataFrame, output_path: Path | None = None
) -> plt.Figure:
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
    if output_path is not None:
        figure.savefig(output_path, dpi=200)
    return figure
