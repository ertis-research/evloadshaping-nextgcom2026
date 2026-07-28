"""Block bootstrap significance testing for paired forecast comparisons.

An i.i.d. bootstrap overstates confidence on autocorrelated 15-minute
telemetry, since adjacent intervals share forecast error (a cloud transient
or a charging session spans several consecutive samples). This module
resamples contiguous blocks with replacement instead of individual points,
so within-block temporal correlation is preserved in each resample -- the
methodology described in the paper's Section IV-C but, until now, never
checked into the repository the paper points readers to.
"""

from __future__ import annotations

import numpy as np

BLOCK_SIZE_24H = 96  # 24h / 15-minute intervals


def block_bootstrap_mae_diff(
    y_true: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    block_size: int = BLOCK_SIZE_24H,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    random_state: int = 42,
    chunk_size: int = 1_000,
    return_distribution: bool = False,
) -> dict[str, float | bool | int]:
    """Moving block bootstrap CI for MAE(a) - MAE(b) on the same held-out split.

    Positive values mean ``predictions_a`` has the larger error (i.e. ``b``
    is more accurate); the caller picks the a/b order to match whichever
    direction it wants to report. The CI is computed over contiguous blocks
    of ``block_size`` consecutive intervals (default 96 = 24h at 15-minute
    resolution) rather than individual points, and the reported effect is
    "significant" only if the interval excludes zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    error_a = np.abs(y_true - np.asarray(predictions_a, dtype=float))
    error_b = np.abs(y_true - np.asarray(predictions_b, dtype=float))
    diff = error_a - error_b
    n = len(diff)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size + 1
    offsets = np.arange(block_size)

    rng = np.random.default_rng(random_state)
    resample_means = np.empty(n_resamples)
    for chunk_start in range(0, n_resamples, chunk_size):
        chunk = min(chunk_size, n_resamples - chunk_start)
        block_starts = rng.integers(0, max_start, size=(chunk, n_blocks))
        idx = (block_starts[:, :, None] + offsets[None, None, :]).reshape(chunk, -1)[
            :, :n
        ]
        resample_means[chunk_start : chunk_start + chunk] = diff[idx].mean(axis=1)

    alpha = 1 - confidence
    lower, upper = np.percentile(
        resample_means, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    result: dict[str, float | bool | int | np.ndarray] = {
        "point_estimate_w": float(diff.mean()),
        "ci_lower_w": float(lower),
        "ci_upper_w": float(upper),
        "confidence": confidence,
        "block_size": block_size,
        "n_resamples": n_resamples,
        "significant": bool(lower > 0 or upper < 0),
    }
    if return_distribution:
        # Opt-in only: cli.py doesn't pass this, so bootstrap_significance.csv
        # is unaffected. Exists for presentation figures that plot the actual
        # resampled distribution rather than just its summary CI.
        result["resample_means"] = resample_means
    return result
