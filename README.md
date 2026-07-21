# Edge-Driven Load Shaping for EV Charging

This workspace contains a rapid research prototype for local forecasting and edge orchestration of EV charging demand against on-site photovoltaic generation.

## What is included

- A small package, [evloadshaping/](evloadshaping/), that:
  - discovers the EV charger columns present in the dataset and loads only the columns it needs ([data.py](evloadshaping/data.py)),
  - resamples raw telemetry into 15-minute intervals and engineers the lag/rolling/cyclical features ([features.py](evloadshaping/features.py)),
  - trains the XGBoost forecasters and the persistence/ridge baselines ([models.py](evloadshaping/models.py)),
  - implements the deterministic edge load-shaping decision and its grid-limit sensitivity sweep ([orchestrator.py](evloadshaping/orchestrator.py)),
  - computes the temporal-error and per-charger-utilization breakdowns ([analysis.py](evloadshaping/analysis.py)),
  - plots the forecast traces and sensitivity curves ([plotting.py](evloadshaping/plotting.py)),
  - and optionally trains a PyTorch MLP/LSTM/1-D CNN comparison — plus a second LSTM given XGBoost's own engineered features, closing the input asymmetry with the raw-sequence models — against the same held-out split ([torch_models.py](evloadshaping/torch_models.py)) — a strictly optional add-on, never imported by default.
- A thin CLI entry point in [pipeline.py](pipeline.py) (`evloadshaping/cli.py` underneath) that runs the full pipeline end to end and writes metrics, plots, and model artifacts to `outputs/`.
- Four notebooks in [notebooks/](notebooks/) that walk through the same building blocks interactively — see below.
- A minimal dependency list in [requirements.txt](requirements.txt).
- A Docker image for local edge-style execution in [Dockerfile](Dockerfile).

## Dataset note

This repository does not include the raw telemetry CSV. The UMA Adabyron dataset is private and available on request from its maintainers, not redistributed here. To run the pipeline, place your own export (or a compatible CSV with the same column naming convention) at `data/20260514_uma_adabyron_data.csv`, or pass a different path via `--data-path`. The pipeline discovers whichever EV charger columns are present in the header and only loads the columns it needs, so it adapts automatically to datasets with a different charger count.

## Local setup

Create a virtual environment, install the dependencies, and run the pipeline:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py --data-path data/20260514_uma_adabyron_data.csv
```

For faster iteration, limit the rows during development:

```bash
python pipeline.py --data-path data/20260514_uma_adabyron_data.csv --max-rows 50000
```

To also run the PyTorch MLP/LSTM/CNN comparison against the same held-out split, install the extra dependency and pass `--include-torch`:

```bash
pip install -r requirements-torch.txt
python pipeline.py --data-path data/20260514_uma_adabyron_data.csv --include-torch
```

This is never required for the default pipeline: `evloadshaping.torch_models` is only imported when `--include-torch` is passed, so a standard edge deployment does not need PyTorch installed.

## Notebooks

[notebooks/](notebooks/) walks through the same pipeline interactively, split across four notebooks:

1. [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) — load the raw telemetry, resample it, and look at the PV/EV signals and their correlations before any modeling.
2. [02_forecasting_and_baselines.ipynb](notebooks/02_forecasting_and_baselines.ipynb) — train the XGBoost forecasters, compare them against the persistence and ridge baselines, and inspect feature importance.
3. [03_orchestrator_and_sensitivity.ipynb](notebooks/03_orchestrator_and_sensitivity.ipynb) — run the deterministic orchestrator, sweep the safe grid-limit parameter, and break down forecast error by time of day and by charger.
4. [04_deep_learning_comparison.ipynb](notebooks/04_deep_learning_comparison.ipynb) — train the PyTorch MLP/LSTM/CNN comparison and compare it against all of the above (requires `requirements-torch.txt`).

Each notebook is self-contained (it reloads and rebuilds whatever it needs), so they can be run independently and in any order. They import directly from the `evloadshaping` package rather than duplicating logic.

Install the extra notebook dependencies (on top of `requirements.txt`) and register a kernel:

```bash
pip install -r requirements-notebooks.txt
jupyter lab notebooks/
```

Notebook 4 additionally needs PyTorch: `pip install -r requirements-torch.txt`.

## Outputs

The pipeline writes the following files to `outputs/`:

- `model_pv.json`, `model_ev.json` — trained XGBoost models.
- `feature_importance_pv.csv`, `feature_importance_ev.csv` — per-feature gain importances.
- `test_predictions.csv` — actual vs. predicted values for both targets on the held-out split.
- `forecast_plot.png` — forecast trace over the full test horizon (each day is a sliver at this scale; useful as a coverage sanity check, not for reading forecast quality).
- `forecast_plot_zoom.png` — the same traces over a two-week window chosen for typical (not spike) EV demand, readable at daily resolution.
- `baseline_comparison.csv` — XGBoost vs. persistence vs. ridge regression, MAE/RMSE per target.
- `baseline_comparison_with_torch.csv` — the above plus the PyTorch MLP/LSTM/CNN comparison and the engineered-feature LSTM variant (only written with `--include-torch`).
- `ablation_study.csv` — base (current-timestep only) vs. full engineered feature set.
- `grid_sensitivity.csv`, `grid_sensitivity_plot.png` — throttle rate and mean per-EV reduction swept across safe grid limits.
- `hourly_error_breakdown.csv` — forecast MAE bucketed by time of day.
- `charger_utilization.csv` — per-charger total energy delivered and active-charging share over the full telemetry span.
- `summary.json` — consolidated run summary including all of the above.

## Docker

Build and run the container locally:

```bash
docker build -t ev-load-shaping .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs ev-load-shaping
```

The dataset and generated outputs are mounted as volumes rather than baked into the image (see `.dockerignore`), since the raw CSV is not part of the repository.

## Research framing

The workflow is designed for a short paper on edge-deployed load shaping, with a focus on low-latency local forecasting, deterministic orchestration, and battery-less microgrid constraints.

## License

Released under the [MIT License](LICENSE).