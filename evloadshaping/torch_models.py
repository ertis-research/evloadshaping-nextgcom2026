"""PyTorch architecture comparison: MLP, LSTM, and 1D-CNN forecasters.

This module is a strictly optional add-on for evaluating whether a nonlinear
neural net, or a learned temporal representation, beats XGBoost and its hand-
engineered lag features on this task. It is never imported by the default CLI
pipeline (see cli.py's --include-torch flag), so a standard edge deployment
does not need PyTorch installed -- consistent with the project's lightweight,
edge-deployable framing.

Three input representations are used, on purpose:
  * MLPRegressor consumes the identical full engineered feature set used by
    XGBoost/ridge, isolating whether a neural net's nonlinearity helps over a
    tree ensemble on the same inputs.
  * LSTMRegressor and CNN1DRegressor consume raw SEQUENCE_WINDOW-step
    sequences of the base (non-lagged) variables, testing whether a learned
    temporal representation can substitute for hand-engineered lag/rolling
    features rather than just re-deriving them.
  * A second LSTMRegressor ("lstm_features") instead consumes
    SEQUENCE_WINDOW-step sequences of the *same* full engineered feature set
    as XGBoost/MLP, closing the remaining fairness gap: whether a learned
    temporal representation still helps once it is not starved of the
    hand-engineered signal either.

Each of the four configurations is retrained from scratch across SEEDS and
reported as mean +/- std, so a single lucky/unlucky initialization can't be
mistaken for a real effect.
"""

from __future__ import annotations

import os

# Must be set before `import torch`: when XGBoost/scikit-learn (already
# imported by models.py/cli.py) and PyTorch each initialize their own OpenMP
# runtime in the same process, they deadlock silently on this machine as soon
# as PyTorch actually spins up its thread pool -- no exception, no traceback,
# just a hung process. KMP_DUPLICATE_LIB_OK=TRUE plus a single-threaded torch
# is the standard workaround. Only set if the caller hasn't already chosen a
# value, so an explicit environment override still wins.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)

from .config import BASE_ONLY_FEATURE_COLUMNS
from .models import evaluate_model

SEQUENCE_WINDOW = 8  # 2 hours of 15-minute steps
SEEDS = (42, 43, 44, 45, 46)  # each DL config is retrained per seed to report mean +/- std


def _device() -> torch.device:
    # MPS (Apple GPU) is deliberately not used here: it hung indefinitely when
    # this module ran as a detached/non-interactive process (no controlling
    # terminal), reproducibly, even though it worked fine interactively. These
    # networks are tiny, so CPU training is fast enough and avoids that
    # environment-specific failure mode entirely -- also a better match for
    # the paper's no-special-hardware framing.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, n_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


class CNN1DRegressor(nn.Module):
    def __init__(self, n_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives as (batch, seq_len, n_channels); Conv1d wants (batch, channels, length).
        features = self.conv(x.transpose(1, 2)).squeeze(-1)
        return self.head(features).squeeze(-1)


def build_sequence_windows(
    frame: pd.DataFrame,
    target: pd.Series,
    channels: list[str],
    window: int = SEQUENCE_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape contiguous rows of `channels` into (window, n_channels) sequences.

    Assumes `frame` is a contiguous 15-minute-indexed slice, true for x_train
    and x_test from features.split_data since build_features only drops
    leading rows before the split. The first `window - 1` rows of the slice
    cannot form a full window and are dropped.
    """
    values = frame[channels].to_numpy(dtype=np.float32)
    targets = target.to_numpy(dtype=np.float32)
    if len(values) < window:
        raise ValueError("Not enough rows to build a single sequence window.")

    sequences = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    # sliding_window_view yields (n_windows, n_channels, window); reorder to
    # (n_windows, window, n_channels) to match (batch, seq_len, channels).
    sequences = np.ascontiguousarray(np.moveaxis(sequences, -1, 1))
    aligned_targets = np.ascontiguousarray(targets[window - 1 :])
    return sequences, aligned_targets


def _train_regressor(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 8,
) -> nn.Module:
    """Train with early stopping on a validation slice carved from the training split.

    The held-out test split is never touched here -- it is only used for the
    final evaluate_model call in run_torch_comparison, after training and
    model selection are both finished.

    Deliberately does not reseed here: callers already call torch.manual_seed
    immediately before constructing the model, and nothing consumes RNG state
    between construction and this call, so reseeding here would silently pin
    the DataLoader's shuffle order to whatever value last reseeded regardless
    of which seed the caller intended -- defeating multi-seed runs.
    """
    device = _device()
    model = model.to(device)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val_t), y_val_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return [((a - mean) / std).astype(np.float32) for a in (train, *others)]


def _standardize_target(train: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) of a 1-D target array, computed on the training split only.

    PV and EV targets range into the thousands of watts; training directly on
    that scale with MSE loss produced unstable, badly underfit models
    (especially the LSTM). Standardizing targets to zero mean / unit variance
    for training, then de-standardizing predictions before evaluate_model, is
    the standard fix and is applied uniformly to all three architectures.
    """
    mean = float(train.mean())
    std = float(train.std())
    return mean, (std if std > 0 else 1.0)


def _aggregate_seeds(per_seed_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Collapse a list of per-seed evaluate_model outputs into mean +/- std.

    Reports mean under the original "mae"/"rmse" keys (so existing CSV/print
    consumers keep working unchanged) plus "*_std" across SEEDS, since a
    single-seed DL number is otherwise indistinguishable from run-to-run
    training variance rather than a real effect.
    """
    maes = np.array([m["mae"] for m in per_seed_metrics])
    rmses = np.array([m["rmse"] for m in per_seed_metrics])
    return {
        "mae": float(maes.mean()),
        "mae_std": float(maes.std(ddof=1)),
        "rmse": float(rmses.mean()),
        "rmse_std": float(rmses.std(ddof=1)),
    }


def run_torch_comparison(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_pv_train: pd.Series,
    y_pv_test: pd.Series,
    y_ev_train: pd.Series,
    y_ev_test: pd.Series,
) -> dict[str, dict[str, dict[str, float]]]:
    """Train and evaluate MLP, LSTM, and 1D-CNN forecasters for both targets."""
    device = _device()
    results: dict[str, dict[str, dict[str, float]]] = {}

    val_size = max(1, int(len(x_train) * 0.1))

    # --- MLP on the full engineered feature set (same inputs as XGBoost/ridge) ---
    x_tr_df, x_val_df = x_train.iloc[:-val_size], x_train.iloc[-val_size:]
    for target_name, y_train_full, y_test in (
        ("pv", y_pv_train, y_pv_test),
        ("ev", y_ev_train, y_ev_test),
    ):
        y_tr, y_val = y_train_full.iloc[:-val_size], y_train_full.iloc[-val_size:]

        x_tr_s, x_val_s, x_test_s = _standardize(
            x_tr_df.to_numpy(dtype=np.float32),
            x_val_df.to_numpy(dtype=np.float32),
            x_test.to_numpy(dtype=np.float32),
        )
        y_mean, y_std = _standardize_target(y_tr.to_numpy(dtype=np.float32))
        y_tr_s = (y_tr.to_numpy(dtype=np.float32) - y_mean) / y_std
        y_val_s = (y_val.to_numpy(dtype=np.float32) - y_mean) / y_std

        # Retrain across SEEDS and report mean +/- std: a single-seed DL number
        # can't be told apart from ordinary training-variance noise, and
        # nn.Module init draws from the global RNG at construction time, so
        # each seed must be set immediately before its own model constructor.
        per_seed = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = MLPRegressor(input_dim=x_tr_s.shape[1])
            model = _train_regressor(model, x_tr_s, y_tr_s, x_val_s, y_val_s)
            model.eval()
            with torch.no_grad():
                preds_std = model(torch.from_numpy(x_test_s).to(device)).cpu().numpy()
            preds = preds_std * y_std + y_mean
            per_seed.append(evaluate_model(y_test, preds))
        results.setdefault("mlp", {})[target_name] = _aggregate_seeds(per_seed)

    # --- LSTM / CNN on raw sequence windows of the base (non-lagged) channels ---
    channels = [c for c in BASE_ONLY_FEATURE_COLUMNS if c in x_train.columns]

    for target_name, y_train_full, y_test in (
        ("pv", y_pv_train, y_pv_test),
        ("ev", y_ev_train, y_ev_test),
    ):
        seq_train_all, target_train_all = build_sequence_windows(
            x_train, y_train_full, channels
        )
        seq_test, target_test = build_sequence_windows(x_test, y_test, channels)

        val_n = max(1, int(len(seq_train_all) * 0.1))
        seq_tr, seq_val = seq_train_all[:-val_n], seq_train_all[-val_n:]
        tgt_tr, tgt_val = target_train_all[:-val_n], target_train_all[-val_n:]

        seq_tr_s, seq_val_s, seq_test_s = _standardize(seq_tr, seq_val, seq_test)
        y_mean, y_std = _standardize_target(tgt_tr)
        tgt_tr_s = (tgt_tr - y_mean) / y_std
        tgt_val_s = (tgt_val - y_mean) / y_std

        for model_name, model_cls, model_kwargs in (
            ("lstm", LSTMRegressor, {"n_channels": len(channels)}),
            ("cnn", CNN1DRegressor, {"n_channels": len(channels)}),
        ):
            per_seed = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                model = model_cls(**model_kwargs)
                trained = _train_regressor(
                    model, seq_tr_s, tgt_tr_s, seq_val_s, tgt_val_s
                )
                trained.eval()
                with torch.no_grad():
                    preds_std = (
                        trained(torch.from_numpy(seq_test_s).to(device)).cpu().numpy()
                    )
                preds = preds_std * y_std + y_mean
                per_seed.append(evaluate_model(target_test, preds))
            results.setdefault(model_name, {})[target_name] = _aggregate_seeds(
                per_seed
            )

    # --- LSTM on sequence windows of the full engineered feature set, i.e. the
    # same inputs XGBoost/MLP receive, so the raw-sequence LSTM/CNN above are
    # not the only temporal-model data point ---
    feature_channels = list(x_train.columns)

    for target_name, y_train_full, y_test in (
        ("pv", y_pv_train, y_pv_test),
        ("ev", y_ev_train, y_ev_test),
    ):
        seq_train_all, target_train_all = build_sequence_windows(
            x_train, y_train_full, feature_channels
        )
        seq_test, target_test = build_sequence_windows(x_test, y_test, feature_channels)

        val_n = max(1, int(len(seq_train_all) * 0.1))
        seq_tr, seq_val = seq_train_all[:-val_n], seq_train_all[-val_n:]
        tgt_tr, tgt_val = target_train_all[:-val_n], target_train_all[-val_n:]

        seq_tr_s, seq_val_s, seq_test_s = _standardize(seq_tr, seq_val, seq_test)
        y_mean, y_std = _standardize_target(tgt_tr)
        tgt_tr_s = (tgt_tr - y_mean) / y_std
        tgt_val_s = (tgt_val - y_mean) / y_std

        per_seed = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = LSTMRegressor(n_channels=len(feature_channels))
            trained = _train_regressor(model, seq_tr_s, tgt_tr_s, seq_val_s, tgt_val_s)
            trained.eval()
            with torch.no_grad():
                preds_std = (
                    trained(torch.from_numpy(seq_test_s).to(device)).cpu().numpy()
                )
            preds = preds_std * y_std + y_mean
            per_seed.append(evaluate_model(target_test, preds))
        results.setdefault("lstm_features", {})[target_name] = _aggregate_seeds(
            per_seed
        )

    return results
