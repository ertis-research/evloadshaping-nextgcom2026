"""Command-line entry point: runs the full pipeline end to end."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from .analysis import charger_utilization_stats, hourly_error_breakdown
from .data import load_extended_raw, load_raw_data
from .features import build_features, split_data
from .models import (
    compute_baselines,
    evaluate_model,
    run_feature_ablation,
    save_feature_importance,
    train_model,
)
from .orchestrator import (
    edge_orchestrator,
    grid_limit_sensitivity,
    summarize_orchestration,
)
from .plotting import plot_forecast_zoom, plot_forecasts, plot_grid_sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train edge load-shaping models for EV charging."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/20260514_uma_adabyron_data.csv"),
        help="Path to the raw UMA Adabyron CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for metrics, plots, and model artifacts.",
    )
    parser.add_argument(
        "--grid-limit",
        type=float,
        default=5000.0,
        help="Safe import capacity available from the grid in watts.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for faster local experiments.",
    )
    parser.add_argument(
        "--include-torch",
        action="store_true",
        help=(
            "Also train and evaluate the PyTorch MLP/LSTM/CNN comparison "
            "(evloadshaping.torch_models). Requires PyTorch "
            "(pip install -r requirements-torch.txt); not needed for the "
            "default edge pipeline."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_path}")
    raw_frame, charger_ids = load_raw_data(args.data_path, max_rows=args.max_rows)
    print(f"Detected EV chargers: {charger_ids}")

    feature_frame = build_features(raw_frame, charger_ids)
    print(f"Prepared {len(feature_frame):,} 15-minute samples")

    x_train, x_test, y_pv_train, y_pv_test, y_ev_train, y_ev_test = split_data(
        feature_frame
    )
    print(f"Train rows: {len(x_train):,} | Test rows: {len(x_test):,}")

    print("Training XGBoost models...")
    model_pv = train_model(x_train, y_pv_train)
    model_ev = train_model(x_train, y_ev_train)

    pv_predictions = model_pv.predict(x_test)
    ev_predictions = model_ev.predict(x_test)

    pv_metrics = evaluate_model(y_pv_test, pv_predictions)
    ev_metrics = evaluate_model(y_ev_test, ev_predictions)

    print("Evaluation metrics")
    print(f"PV   MAE: {pv_metrics['mae']:.2f} W | RMSE: {pv_metrics['rmse']:.2f} W")
    print(f"EV   MAE: {ev_metrics['mae']:.2f} W | RMSE: {ev_metrics['rmse']:.2f} W")

    feature_names = [column for column in x_train.columns]
    model_pv.save_model(str(args.output_dir / "model_pv.json"))
    model_ev.save_model(str(args.output_dir / "model_ev.json"))
    save_feature_importance(
        model_pv, feature_names, args.output_dir / "feature_importance_pv.csv"
    )
    save_feature_importance(
        model_ev, feature_names, args.output_dir / "feature_importance_ev.csv"
    )

    predictions = pd.DataFrame(
        {
            "actual_pv": y_pv_test.values,
            "predicted_pv": pv_predictions,
            "actual_ev": y_ev_test.values,
            "predicted_ev": ev_predictions,
        },
        index=x_test.index,
    )
    predictions.to_csv(args.output_dir / "test_predictions.csv")

    forecast_figure = plot_forecasts(
        test_index=x_test.index,
        actual_pv=y_pv_test,
        predicted_pv=pv_predictions,
        actual_ev=y_ev_test,
        predicted_ev=ev_predictions,
        output_path=args.output_dir / "forecast_plot.png",
    )
    plt.close(forecast_figure)

    zoom_figure = plot_forecast_zoom(
        test_index=x_test.index,
        actual_pv=y_pv_test,
        predicted_pv=pv_predictions,
        actual_ev=y_ev_test,
        predicted_ev=ev_predictions,
        output_path=args.output_dir / "forecast_plot_zoom.png",
    )
    plt.close(zoom_figure)

    test_state = x_test.iloc[0]
    decision = edge_orchestrator(
        predicted_pv=float(pv_predictions[0]),
        predicted_ev=float(ev_predictions[0]),
        current_evs_connected=int(round(float(test_state["total_evs_connected"]))),
        grid_limit=args.grid_limit,
    )

    orchestration_stats = summarize_orchestration(
        pv_predictions,
        ev_predictions,
        x_test["total_evs_connected"].to_numpy(),
        grid_limit=args.grid_limit,
    )

    print("Computing baselines (persistence, ridge regression)...")
    baselines = compute_baselines(
        x_test, y_pv_test, y_ev_test, x_train, y_pv_train, y_ev_train
    )
    pd.DataFrame(
        [
            {"model": "persistence", "target": "pv", **baselines["persistence"]["pv"]},
            {"model": "persistence", "target": "ev", **baselines["persistence"]["ev"]},
            {
                "model": "ridge_regression",
                "target": "pv",
                **baselines["ridge_regression"]["pv"],
            },
            {
                "model": "ridge_regression",
                "target": "ev",
                **baselines["ridge_regression"]["ev"],
            },
            {"model": "xgboost", "target": "pv", **pv_metrics},
            {"model": "xgboost", "target": "ev", **ev_metrics},
        ]
    ).to_csv(args.output_dir / "baseline_comparison.csv", index=False)

    print("Running feature ablation (base features vs. full feature set)...")
    ablation = run_feature_ablation(
        x_train, x_test, y_pv_train, y_pv_test, y_ev_train, y_ev_test
    )
    pd.DataFrame(
        [
            {
                "feature_set": "base_only",
                "target": "pv",
                **ablation["base_features_only"]["pv"],
            },
            {
                "feature_set": "base_only",
                "target": "ev",
                **ablation["base_features_only"]["ev"],
            },
            {"feature_set": "full", "target": "pv", **pv_metrics},
            {"feature_set": "full", "target": "ev", **ev_metrics},
        ]
    ).to_csv(args.output_dir / "ablation_study.csv", index=False)

    print("Sweeping grid-limit sensitivity...")
    sensitivity = grid_limit_sensitivity(
        pv_predictions,
        ev_predictions,
        x_test["total_evs_connected"].to_numpy(),
        grid_limits=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000],
    )
    sensitivity.to_csv(args.output_dir / "grid_sensitivity.csv", index=False)
    sensitivity_figure = plot_grid_sensitivity(
        sensitivity, args.output_dir / "grid_sensitivity_plot.png"
    )
    plt.close(sensitivity_figure)

    print("Computing hourly error breakdown...")
    hourly_errors = hourly_error_breakdown(
        x_test.index, y_pv_test, pv_predictions, y_ev_test, ev_predictions
    )
    hourly_errors.to_csv(args.output_dir / "hourly_error_breakdown.csv", index=False)

    print("Loading per-charger telemetry for utilization analysis...")
    extended_frame = load_extended_raw(args.data_path, charger_ids)
    charger_stats = charger_utilization_stats(extended_frame, charger_ids)
    charger_stats.to_csv(args.output_dir / "charger_utilization.csv", index=False)

    torch_results = None
    if args.include_torch:
        print("Training PyTorch MLP/LSTM/CNN comparison (this takes a while)...")
        from .torch_models import SEEDS, run_torch_comparison  # lazy: torch is optional

        torch_results = run_torch_comparison(
            x_train, x_test, y_pv_train, y_pv_test, y_ev_train, y_ev_test
        )
        rows = [
            {"model": "persistence", "target": "pv", **baselines["persistence"]["pv"]},
            {"model": "persistence", "target": "ev", **baselines["persistence"]["ev"]},
            {
                "model": "ridge_regression",
                "target": "pv",
                **baselines["ridge_regression"]["pv"],
            },
            {
                "model": "ridge_regression",
                "target": "ev",
                **baselines["ridge_regression"]["ev"],
            },
            {"model": "xgboost", "target": "pv", **pv_metrics},
            {"model": "xgboost", "target": "ev", **ev_metrics},
        ]
        for model_name in ("mlp", "lstm", "cnn", "lstm_features"):
            for target_name in ("pv", "ev"):
                rows.append(
                    {
                        "model": model_name,
                        "target": target_name,
                        **torch_results[model_name][target_name],
                    }
                )
        pd.DataFrame(rows).to_csv(
            args.output_dir / "baseline_comparison_with_torch.csv", index=False
        )
        for model_name in ("mlp", "lstm", "cnn", "lstm_features"):
            pv_m = torch_results[model_name]["pv"]
            ev_m = torch_results[model_name]["ev"]
            print(
                f"{model_name.upper():5s} PV MAE: {pv_m['mae']:.2f} +/- {pv_m['mae_std']:.2f} W | "
                f"EV MAE: {ev_m['mae']:.2f} +/- {ev_m['mae_std']:.2f} W "
                f"(across {len(SEEDS)} seeds)"
            )

    summary = {
        "data_path": str(args.data_path),
        "charger_ids": charger_ids,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "pv_metrics": pv_metrics,
        "ev_metrics": ev_metrics,
        "edge_decision": decision,
        "orchestration_stats": orchestration_stats,
        "baselines": baselines,
        "ablation": ablation,
    }
    if torch_results is not None:
        summary["torch_comparison"] = torch_results
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Edge node simulation")
    print(
        f"Forecasted PV: {pv_predictions[0]:.2f} W | Forecasted EV Demand: {ev_predictions[0]:.2f} W"
    )
    print(f"Orchestrator action: {decision['message']}")
    print(
        "Throttle triggered in "
        f"{orchestration_stats['n_throttle']}/{orchestration_stats['n_intervals']} "
        f"test intervals ({orchestration_stats['throttle_rate'] * 100:.2f}%), "
        f"mean reduction {orchestration_stats['mean_reduction_per_ev_w']:.2f} W/EV"
    )


if __name__ == "__main__":
    main()
