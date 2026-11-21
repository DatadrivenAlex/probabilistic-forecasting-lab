# Probabilistic Forecasting Lab

A compact Python project for leakage-aware hourly time-series forecasting. It compares a fitted gradient-boosting forecaster with a seasonal-naive baseline, then calibrates prediction intervals on a separate chronological block and scores the final holdout.

> All data is synthetic. The benchmark is an engineering demonstration, not evidence that the model will outperform a baseline on real operations data.

## What it demonstrates

- Reproducible synthetic demand with daily and weekly seasonality, trend, event pulses and noise.
- Lag/calendar feature construction using observations strictly before each target timestamp.
- Chronological train, calibration and test blocks; no random shuffling of time-series rows.
- Gradient-boosting point and 10th/90th percentile forecasts, with split-conformal interval expansion.
- MAE, RMSE, quantile pinball loss, holdout interval coverage and width, compared with a daily seasonal-naive baseline.
- A one-step rolling-origin evaluation: each next timestamp uses preceding observed values, while the fitted models stay fixed during the holdout.

## Run locally

```bash
uv sync --extra dev
uv run forecast-lab --hours 2880 --seed 42
uv run pytest
uv run ruff check .
```

The command prints train/calibration/test sizes and JSON metrics. No external data, model API, GPU or credentials are needed.

## Evaluation notes

The point and quantile models fit only the earliest training block. A later calibration block estimates a nonnegative expansion for the raw 10–90% interval; only then is the final block evaluated. The baseline predicts the value observed at the same hour on the previous day. It is deliberately simple so that results can be compared without a hidden deployment dependency.

The finite-sample calibration adjustment does not guarantee future coverage when temporal dependence, distribution drift or changing seasonality violate exchangeability. The synthetic data generator is known, so its results should be read as a reproducible software demo rather than a real forecasting claim.

## Limitations

- One-step forecasting only; it does not provide long-range recursive forecasts or automatic retraining.
- Synthetic univariate demand, one fixed seed by default and a single chronological split; no real-world data ingestion, holiday calendar or operational monitoring.
- Models and feature choices are intentionally small and transparent; no claim of optimality or production readiness.

## License

MIT
