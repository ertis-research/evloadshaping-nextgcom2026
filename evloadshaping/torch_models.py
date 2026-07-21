"""PyTorch architecture comparison: MLP, LSTM, and 1D-CNN forecasters.

This module is a strictly optional add-on for evaluating whether a nonlinear
neural net, or a learned temporal representation, beats XGBoost and its hand-
engineered lag features on this task. It is never imported by the default CLI
pipeline (see cli.py's --include-torch flag), so a standard edge deployment
does not need PyTorch installed -- consistent with the project's lightweight,
edge-deployable framing.

Two distinct input representations are used, on purpose:
  * MLPRegressor consumes the identical full engineered feature set used by
    XGBoost/ridge, isolating whether a neural net's nonlinearity helps over a
    tree ensemble on the same inputs.
  * LSTMRegressor and CNN1DRegressor instead consume raw SEQUENCE_WINDOW-step
    sequences of the base (non-lagged) variables, testing whether a learned
    temporal representation can substitute for hand-engineered lag/rolling
    features rather than just re-deriving them.
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
RANDOM_SEED = 42


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
    """
    torch.manual_seed(RANDOM_SEED)
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

        model = MLPRegressor(input_dim=x_tr_s.shape[1])
        model = _train_regressor(
            model,
            x_tr_s,
            ((y_tr.to_numpy(dtype=np.float32) - y_mean) / y_std),
            x_val_s,
            ((y_val.to_numpy(dtype=np.float32) - y_mean) / y_std),
        )
        model.eval()
        with torch.no_grad():
            preds_std = model(torch.from_numpy(x_test_s).to(device)).cpu().numpy()
        preds = preds_std * y_std + y_mean
        results.setdefault("mlp", {})[target_name] = evaluate_model(y_test, preds)

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

        for model_name, model in (
            ("lstm", LSTMRegressor(n_channels=len(channels))),
            ("cnn", CNN1DRegressor(n_channels=len(channels))),
        ):
            trained = _train_regressor(model, seq_tr_s, tgt_tr_s, seq_val_s, tgt_val_s)
            trained.eval()
            with torch.no_grad():
                preds_std = (
                    trained(torch.from_numpy(seq_test_s).to(device)).cpu().numpy()
                )
            preds = preds_std * y_std + y_mean
            results.setdefault(model_name, {})[target_name] = evaluate_model(
                target_test, preds
            )

    return results
