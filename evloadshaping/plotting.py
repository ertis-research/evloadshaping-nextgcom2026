"""Plotting helpers, shared by the CLI pipeline and the notebooks.

This module does not force a matplotlib backend: the CLI entry point
(``evloadshaping.cli``) selects the non-interactive ``Agg`` backend for
headless runs, while notebooks keep their own inline backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forecasts(
    test_index: pd.Index,
    actual_pv: pd.Series,
    predicted_pv: np.ndarray,
    actual_ev: pd.Series,
    predicted_ev: np.ndarray,
    output_path: Path | None = None,
) -> plt.Figure:
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
    if output_path is not None:
        figure.savefig(output_path, dpi=200)
    return figure


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
