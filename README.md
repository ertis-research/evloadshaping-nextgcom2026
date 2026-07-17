# Edge-Driven Load Shaping for EV Charging

This workspace contains a rapid research prototype for local forecasting and edge orchestration of EV charging demand against on-site photovoltaic generation.

## What is included

- A single reproducible Python pipeline in [pipeline.py](pipeline.py) that:
  - discovers the EV charger columns present in the dataset,
  - aggregates raw telemetry into 15-minute intervals,
  - trains two XGBoost regressors for next-step PV and EV-demand forecasting,
  - simulates a deterministic edge load-shaping decision,
  - writes metrics, plots, and model artifacts to `outputs/`.
- A minimal dependency list in [requirements.txt](requirements.txt).
- A Docker image for local edge-style execution in [Dockerfile](Dockerfile).

## Dataset note

This repository does not include the raw telemetry CSV: it is several hundred megabytes and is not distributed with the source. To run the pipeline, place your own UMA Adabyron export (or a compatible CSV with the same column naming convention) at `data/20260514_uma_adabyron_data.csv`, or pass a different path via `--data-path`. The pipeline discovers whichever EV charger columns are present in the header and only loads the columns it needs, so it adapts automatically to datasets with a different charger count.

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

## Outputs

The pipeline writes the following files to `outputs/`:

- `model_pv.json`, `model_ev.json` — trained XGBoost models.
- `feature_importance_pv.csv`, `feature_importance_ev.csv` — per-feature gain importances.
- `test_predictions.csv` — actual vs. predicted values for both targets on the held-out split.
- `forecast_plot.png` — forecast trace plot over the test horizon.
- `baseline_comparison.csv` — XGBoost vs. persistence vs. ridge regression, MAE/RMSE per target.
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