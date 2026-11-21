"""Chronological one-step rolling-origin backtest with calibrated intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor

FloatArray = NDArray[np.float64]
LAGS = (1, 2, 24, 48, 168)


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    pinball_p10: float
    pinball_p50: float
    pinball_p90: float
    interval_coverage: float
    interval_width: float


@dataclass(frozen=True)
class PointMetrics:
    mae: float
    rmse: float


@dataclass(frozen=True)
class BacktestResult:
    train_rows: int
    calibration_rows: int
    test_rows: int
    conformal_adjustment: float
    model: ForecastMetrics
    seasonal_naive: PointMetrics

    def to_dict(self) -> dict:
        return asdict(self)


def make_supervised(series: FloatArray) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """Create one-step examples using only lagged observations and target-time calendar terms."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("series must be a finite one-dimensional array")
    if len(values) <= max(LAGS) + 20:
        raise ValueError(f"series must contain more than {max(LAGS) + 20} observations")
    target_idx = np.arange(max(LAGS), len(values), dtype=np.int64)
    lag_features = np.column_stack([values[target_idx - lag] for lag in LAGS])
    hour = target_idx % 24
    weekday = (target_idx // 24) % 7
    calendar = np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
        ]
    )
    X = np.column_stack([lag_features, calendar]).astype(float)
    return X, values[target_idx], target_idx


def _pinball(actual: FloatArray, predicted: FloatArray, quantile: float) -> float:
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def _metrics(
    actual: FloatArray,
    lower: FloatArray,
    median: FloatArray,
    upper: FloatArray,
) -> ForecastMetrics:
    error = actual - median
    return ForecastMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        pinball_p10=_pinball(actual, lower, 0.1),
        pinball_p50=_pinball(actual, median, 0.5),
        pinball_p90=_pinball(actual, upper, 0.9),
        interval_coverage=float(np.mean((actual >= lower) & (actual <= upper))),
        interval_width=float(np.mean(upper - lower)),
    )


def run_backtest(series: FloatArray, *, alpha: float = 0.2) -> BacktestResult:
    """Fit train-only quantile models, calibrate on the next block, score the final block.

    Inputs are rolling one-step forecasts: at each test timestamp the preceding observed
    series values are available as lags. Models are not retrained during the holdout.
    Calibration uses a finite-sample absolute interval score; temporal dependence and drift
    mean its nominal coverage is not a universal guarantee.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    X, y, target_idx = make_supervised(series)
    n = len(y)
    train_end = int(n * 0.65)
    calibration_end = int(n * 0.80)
    if train_end < 50 or calibration_end - train_end < 10 or n - calibration_end < 10:
        raise ValueError("series is too short for non-empty train/calibration/test blocks")
    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[train_end:calibration_end], y[train_end:calibration_end]
    X_test, y_test = X[calibration_end:], y[calibration_end:]

    def model(loss: str, quantile: float | None = None) -> HistGradientBoostingRegressor:
        params: dict[str, float | int | str] = {
            "loss": loss,
            "max_iter": 120,
            "max_leaf_nodes": 15,
            "l2_regularization": 2.0,
            "random_state": 42,
        }
        if quantile is not None:
            params["quantile"] = quantile
        return HistGradientBoostingRegressor(**params)

    lower_model = model("quantile", 0.1).fit(X_train, y_train)
    median_model = model("squared_error").fit(X_train, y_train)
    upper_model = model("quantile", 0.9).fit(X_train, y_train)
    cal_lower = lower_model.predict(X_cal)
    cal_upper = upper_model.predict(X_cal)
    # Ensure raw bounds are ordered before computing calibration scores.
    cal_lower, cal_upper = np.minimum(cal_lower, cal_upper), np.maximum(cal_lower, cal_upper)
    scores = np.maximum.reduce([cal_lower - y_cal, y_cal - cal_upper, np.zeros(len(y_cal))])
    rank = min(len(scores), ceil((len(scores) + 1) * (1 - alpha)))
    adjustment = float(np.partition(scores, rank - 1)[rank - 1])

    raw_lower = lower_model.predict(X_test)
    raw_median = median_model.predict(X_test)
    raw_upper = upper_model.predict(X_test)
    lower = np.minimum(raw_lower, raw_upper) - adjustment
    upper = np.maximum(raw_lower, raw_upper) + adjustment
    model_metrics = _metrics(y_test, lower, raw_median, upper)

    # Seasonal-naive point baseline: same hour one day earlier; no fitted model.
    original = np.asarray(series, dtype=float)
    naive = original[target_idx[calibration_end:] - 24]
    naive_error = y_test - naive
    naive_metrics = PointMetrics(
        mae=float(np.mean(np.abs(naive_error))),
        rmse=float(np.sqrt(np.mean(naive_error**2))),
    )
    return BacktestResult(
        train_rows=len(y_train),
        calibration_rows=len(y_cal),
        test_rows=len(y_test),
        conformal_adjustment=adjustment,
        model=model_metrics,
        seasonal_naive=naive_metrics,
    )
