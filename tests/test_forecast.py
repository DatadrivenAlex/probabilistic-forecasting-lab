import numpy as np
import pytest

from forecast_lab import make_synthetic_series, run_backtest
from forecast_lab.forecast import make_supervised


def test_synthetic_series_is_reproducible_and_finite() -> None:
    first = make_synthetic_series(900, seed=8)
    second = make_synthetic_series(900, seed=8)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_supervised_features_use_only_past_values() -> None:
    series = make_synthetic_series(900)
    X1, y1, indices = make_supervised(series)
    changed_future = series.copy()
    changed_future[700:] += 10_000
    X2, y2, _ = make_supervised(changed_future)
    past = indices < 700
    assert np.array_equal(X1[past], X2[past])
    assert np.array_equal(y1[past], y2[past])
    assert np.all(indices[1:] > indices[:-1])


def test_backtest_has_chronological_nonempty_blocks_and_finite_metrics() -> None:
    result = run_backtest(make_synthetic_series(24 * 50, seed=17))
    assert result.train_rows > result.calibration_rows > 0
    assert result.test_rows > 0
    assert result.conformal_adjustment >= 0
    assert result.model.mae >= 0
    assert result.model.rmse >= 0
    assert 0 <= result.model.interval_coverage <= 1
    assert result.model.interval_width >= 0
    assert result.seasonal_naive.mae >= 0
    assert result.seasonal_naive.rmse >= 0


def test_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least 300"):
        make_synthetic_series(100)
    with pytest.raises(ValueError, match="one-dimensional"):
        make_supervised(np.zeros((20, 2)))
    with pytest.raises(ValueError, match="alpha"):
        run_backtest(make_synthetic_series(900), alpha=1.0)
