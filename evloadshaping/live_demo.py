"""Live edge-node replay: for demos and presentation video, not the research pipeline.

Loads the two already-trained XGBoost models (from a prior `pipeline.py`
run) and replays a window of the held-out test split one interval at a
time, as if running on the edge: single-sample inference (not a batch
predict), timed per interval, fed straight into the orchestrator, printed
as it happens with a short delay so it reads as a live stream on camera.

This intentionally does no training and reports nothing that isn't already
in the paper -- it is a presentation aid, replaying real held-out data
through the real trained models and the real orchestrator rule.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from .data import load_raw_data
from .features import build_features, split_data
from .orchestrator import edge_orchestrator, orchestrator_event_log

RESET = "\033[0m"
DIM = "\033[2m"
STATUS_STYLE = {
    "stable": "\033[32m",  # green
    "throttle": "\033[31;1m",  # bold red
    "inspect": "\033[33m",  # yellow
}


def _load_models(model_dir: Path) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    model_pv = xgb.XGBRegressor()
    model_pv.load_model(str(model_dir / "model_pv.json"))
    model_ev = xgb.XGBRegressor()
    model_ev.load_model(str(model_dir / "model_ev.json"))
    return model_pv, model_ev


def find_demo_window(
    pv_predictions: np.ndarray,
    ev_predictions: np.ndarray,
    y_pv_test: np.ndarray,
    y_ev_test: np.ndarray,
    connected_evs: np.ndarray,
    index,
    grid_limit: float,
    window_intervals: int,
) -> tuple[int, int]:
    """Pick a short, compelling window: centered on the largest genuine throttle event.

    Computed from the data (not a hard-coded date), matching the same
    principle plotting.plot_forecast_zoom uses -- so this generalizes to any
    dataset/test split instead of only this one. Falls back to the middle of
    the test set if nothing genuinely triggers at this grid limit.
    """
    events = orchestrator_event_log(
        pv_predictions, ev_predictions, y_pv_test, y_ev_test, connected_evs,
        index, grid_limit=grid_limit,
    )
    genuine = events[events["genuine_risk"]]
    if genuine.empty:
        center = len(index) // 2
    else:
        biggest = genuine.sort_values("actual_deficit_w", ascending=False).iloc[0]
        center = int(index.get_indexer([biggest["timestamp"]])[0])

    half = window_intervals // 2
    start = max(0, center - half)
    end = min(len(index), start + window_intervals)
    start = max(0, end - window_intervals)
    return start, end


def _format_line(
    timestamp,
    predicted_pv: float,
    actual_pv: float,
    predicted_ev: float,
    actual_ev: float,
    connected: int,
    decision: dict,
    latency_ms: float,
) -> str:
    color = STATUS_STYLE.get(decision["status"], "")
    status_label = decision["status"].upper()
    detail = ""
    if decision["status"] == "throttle":
        detail = f" -> -{decision['reduction_per_ev_w']:.0f} W/EV"

    return (
        f"{DIM}{timestamp:%Y-%m-%d %H:%M}{RESET}  "
        f"PV {predicted_pv:6.0f} W (actual {actual_pv:6.0f} W)  "
        f"EV {predicted_ev:6.0f} W (actual {actual_ev:6.0f} W)  "
        f"EVs connected: {connected}  "
        f"{color}[{status_label}]{RESET}{detail}  "
        f"{DIM}({latency_ms:.2f} ms){RESET}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a window of the held-out test split through the trained "
            "edge models and orchestrator, one interval at a time, for a demo."
        )
    )
    parser.add_argument(
        "--data-path", type=Path, default=Path("data/20260514_uma_adabyron_data.csv")
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing model_pv.json/model_ev.json from a prior pipeline.py run.",
    )
    parser.add_argument("--grid-limit", type=float, default=1000.0)
    parser.add_argument(
        "--window-intervals",
        type=int,
        default=24,
        help="How many 15-minute intervals to replay (default: 24 = 6 hours).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds to pause between intervals, for watchable pacing on camera.",
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="Explicit start index into the test split, overriding auto window selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (args.model_dir / "model_pv.json").exists():
        raise SystemExit(
            f"No trained model found at {args.model_dir}/model_pv.json -- "
            "run `python pipeline.py` first to train and save the models."
        )

    print(f"Loading trained models from {args.model_dir}...")
    model_pv, model_ev = _load_models(args.model_dir)

    print(f"Loading test split from {args.data_path}...")
    raw_frame, charger_ids = load_raw_data(args.data_path)
    feature_frame = build_features(raw_frame, charger_ids)
    x_train, x_test, y_pv_train, y_pv_test, y_ev_train, y_ev_test = split_data(
        feature_frame
    )
    connected_evs = x_test["total_evs_connected"].to_numpy()

    # A single batch predict just to locate a compelling window; the replay
    # loop below re-predicts each row individually to time real single-sample
    # inference latency, matching how the edge service actually runs.
    pv_batch = model_pv.predict(x_test)
    ev_batch = model_ev.predict(x_test)

    if args.start is not None:
        start = max(0, args.start)
        end = min(len(x_test), start + args.window_intervals)
    else:
        start, end = find_demo_window(
            pv_batch, ev_batch, y_pv_test.to_numpy(), y_ev_test.to_numpy(),
            connected_evs, x_test.index, args.grid_limit, args.window_intervals,
        )

    print(
        f"\nReplaying {end - start} intervals "
        f"({x_test.index[start]:%Y-%m-%d %H:%M} to {x_test.index[end - 1]:%Y-%m-%d %H:%M}), "
        f"grid limit {args.grid_limit:.0f} W\n"
    )

    for i in range(start, end):
        row = x_test.iloc[[i]]

        t0 = time.perf_counter()
        predicted_pv = float(model_pv.predict(row)[0])
        predicted_ev = float(model_ev.predict(row)[0])
        latency_ms = (time.perf_counter() - t0) * 1000

        connected = int(round(float(row["total_evs_connected"].iloc[0])))
        decision = edge_orchestrator(
            predicted_pv=predicted_pv,
            predicted_ev=predicted_ev,
            current_evs_connected=connected,
            grid_limit=args.grid_limit,
        )

        print(
            _format_line(
                x_test.index[i],
                predicted_pv,
                float(y_pv_test.iloc[i]),
                predicted_ev,
                float(y_ev_test.iloc[i]),
                connected,
                decision,
                latency_ms,
            )
        )
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
