# Edge-Driven Load Shaping for EV Charging

This workspace contains a rapid research prototype for local forecasting and edge orchestration of EV charging demand against on-site photovoltaic generation. It is the code release for a short paper submitted to the NextGCom 2026 EVOLVE special session.

## What is included

- A small package, [evloadshaping/](evloadshaping/), that:
  - discovers the EV charger columns present in the dataset and loads only the columns it needs ([data.py](evloadshaping/data.py)),
  - resamples raw telemetry into 15-minute intervals and engineers the lag/rolling/cyclical features ([features.py](evloadshaping/features.py)),
  - trains the XGBoost forecasters and the persistence/ridge baselines ([models.py](evloadshaping/models.py)),
  - implements the deterministic edge load-shaping decision, its persistence-forecast counterfactual, and the grid-limit sensitivity sweep ([orchestrator.py](evloadshaping/orchestrator.py)),
  - runs a moving block bootstrap to check whether XGBoost's edge over persistence survives temporal autocorrelation ([significance.py](evloadshaping/significance.py)),
  - computes the temporal-error and per-charger-utilization breakdowns ([analysis.py](evloadshaping/analysis.py)),
  - plots the forecast traces and sensitivity curves ([plotting.py](evloadshaping/plotting.py)),
  - and optionally trains a PyTorch MLP/LSTM/1-D CNN comparison — plus a second LSTM given XGBoost's own engineered features, closing the input asymmetry with the raw-sequence models — against the same held-out split ([torch_models.py](evloadshaping/torch_models.py)), retraining each configuration across 5 seeds and reporting mean ± std — a strictly optional add-on, never imported by default.
- Four notebooks in [notebooks/](notebooks/) — the recommended way to explore this project (see below).
- A thin CLI entry point in [pipeline.py](pipeline.py) (`evloadshaping/cli.py` underneath) that runs the full pipeline end to end and writes metrics, plots, and model artifacts to `outputs/` — useful for automation or a real edge deployment, not required for exploring the results.
- Dependencies managed as a [uv](https://docs.astral.sh/uv/) project (`pyproject.toml` + `uv.lock`).

**Reproducibility:** every stochastic step (XGBoost's row/column subsampling, the PyTorch initializations, the block bootstrap) uses the fixed seeds in `evloadshaping/config.py` (`SEEDS = (42, 43, 44, 45, 46)`), so the CLI and notebooks reproduce the paper's tables and figures exactly, not just approximately.

## Dataset note

This repository does not include the raw telemetry CSV. The UMA Adabyron dataset is private and available on request from its maintainers, not redistributed here. To run the pipeline, place your own export (or a compatible CSV with the same column naming convention) at `data/20260514_uma_adabyron_data.csv`, or pass a different path via `--data-path`. The pipeline discovers whichever EV charger columns are present in the header and only loads the columns it needs, so it adapts automatically to datasets with a different charger count.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv` on macOS), then from this directory:

```bash
uv sync --extra notebooks --extra torch
```

That resolves and installs everything — the core pipeline plus the notebook and PyTorch extras — into a local `.venv`, without needing to create or activate one by hand. Leave off `--extra torch` if you don't need the deep-learning comparison (it is never required for the default pipeline).

## Notebooks (recommended)

[notebooks/](notebooks/) is the easiest way to run this project: no CLI flags, each step's output is visible inline, and every notebook is pinned to the same fixed seeds as the paper, so re-running one reproduces the paper's numbers exactly.

```bash
uv run --extra notebooks jupyter lab notebooks/
```

1. [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) — load the raw telemetry, resample it, and look at the PV/EV signals and their correlations before any modeling.
2. [02_forecasting_and_baselines.ipynb](notebooks/02_forecasting_and_baselines.ipynb) — train the XGBoost forecasters (Table II), compare against persistence/ridge and the 5-seed XGBoost mean (Table IV), run the block bootstrap significance test (Section VI-B), and reproduce the ablation study and Fig. 1.
3. [03_orchestrator_and_sensitivity.ipynb](notebooks/03_orchestrator_and_sensitivity.ipynb) — run the deterministic orchestrator, log individual throttle events with genuine-risk flags, rerun the orchestrator on persistence forecasts to quantify what the forecaster buys the control loop (Section VI-C), sweep the safe grid-limit parameter, and break down forecast error by time of day and by charger.
4. [04_deep_learning_comparison.ipynb](notebooks/04_deep_learning_comparison.ipynb) — train the PyTorch MLP/LSTM/CNN comparison and compare it against XGBoost's 5-seed mean on the same basis (requires `--extra torch`).

Each notebook is self-contained (it reloads and rebuilds whatever it needs), so they can be run independently and in any order. They import directly from the `evloadshaping` package rather than duplicating logic.

## CLI (for automation or edge deployment)

The same pipeline as a single script, useful for a real deployment or a CI job rather than interactive exploration:

```bash
uv run python pipeline.py --data-path data/20260514_uma_adabyron_data.csv
```

For faster iteration, limit the rows during development:

```bash
uv run python pipeline.py --data-path data/20260514_uma_adabyron_data.csv --max-rows 50000
```

To also run the PyTorch MLP/LSTM/CNN comparison against the same held-out split (requires the `torch` extra installed via `uv sync --extra torch`):

```bash
uv run python pipeline.py --data-path data/20260514_uma_adabyron_data.csv --include-torch
```

This is never required for the default pipeline: `evloadshaping.torch_models` is only imported when `--include-torch` is passed, so a standard edge deployment does not need PyTorch installed.

## Live demo

For presentations: [live_demo.py](live_demo.py) (`evloadshaping/live_demo.py`
underneath) loads the already-trained models from a prior `pipeline.py` run
and replays a short window of the held-out test split one interval at a
time, single-sample inference (not a batch predict), timed and printed live
with a short pause between intervals so it reads as a real-time stream on
camera. By default it auto-selects a window centered on the largest genuine
throttle event, the same data-driven-window principle `plot_forecast_zoom`
uses, so it doesn't need a hand-picked date:

```bash
uv run python pipeline.py --data-path data/20260514_uma_adabyron_data.csv  # once, to produce outputs/model_*.json
uv run python live_demo.py
```

Useful flags: `--grid-limit` (default 1000 W, the stress case with more
events to show), `--window-intervals` (how many 15-minute steps to replay),
`--delay` (seconds between intervals, for pacing on camera), `--start` (an
explicit index into the test split instead of auto-selection).

## Outputs

The pipeline writes the following files to `outputs/`:

- `model_pv.json`, `model_ev.json` — trained XGBoost models.
- `feature_importance_pv.csv`, `feature_importance_ev.csv` — per-feature gain importances.
- `test_predictions.csv` — actual vs. predicted values for both targets on the held-out split.
- `forecast_plot.png` — forecast trace over the full test horizon (each day is a sliver at this scale; useful as a coverage sanity check, not for reading forecast quality).
- `forecast_plot_zoom.png` — the same traces over a two-week window chosen for typical (not spike) EV demand, readable at daily resolution. This is the paper's Fig. 1.
- `baseline_comparison.csv` — XGBoost vs. persistence vs. ridge regression, MAE/RMSE per target. XGBoost's row is a mean/std over 5 seeds (its `subsample`/`colsample_bytree` < 1 make `random_state` a real variance source); persistence and ridge are deterministic solvers with no seed dependence.
- `bootstrap_significance.csv` — moving block bootstrap (24h blocks, 10,000 resamples) 95% CI on `MAE(XGBoost) - MAE(persistence)` per target, i.e. whether XGBoost's point-estimate edge over persistence survives temporal autocorrelation in the 15-minute series (Section VI-B).
- `baseline_comparison_with_torch.csv` — the above plus the PyTorch MLP/LSTM/CNN comparison and the engineered-feature LSTM variant, each as mean/std MAE and RMSE over the same 5 seeds (only written with `--include-torch`).
- `ablation_study.csv` — base (current-timestep only) vs. full engineered feature set.
- `grid_sensitivity.csv`, `grid_sensitivity_plot.png` — throttle rate and mean per-EV reduction swept across safe grid limits.
- `hourly_error_breakdown.csv` — forecast MAE bucketed by time of day.
- `charger_utilization.csv` — per-charger total energy delivered and active-charging share over the full telemetry span.
- `orchestrator_events_<grid_limit>w.csv`, `orchestrator_events_1000w.csv` — one row per triggered throttle event at the run's `--grid-limit` and at a fixed 1 kW stress case, including a `genuine_risk` flag computed from actual (not forecasted) PV/EV values. The default grid limit triggers too rarely to assess control quality from the aggregate rate alone, so these logs let each event be checked individually instead of only counted.
- `summary.json` — consolidated run summary including all of the above, plus the persistence-forecast orchestrator counterfactual (throttle rate and decision-agreement rate vs. XGBoost, Section VI-C).

## Research framing

The workflow is designed for a short paper on edge-deployed load shaping, with a focus on low-latency local forecasting, deterministic orchestration, and battery-less microgrid constraints.

## License

Released under the [MIT License](LICENSE).
