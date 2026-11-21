"""Deterministic synthetic hourly demand series for offline experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_synthetic_series(n_hours: int = 24 * 120, *, seed: int = 42) -> NDArray[np.float64]:
    """Build synthetic demand with daily/weekly seasonality, trend, events and noise."""
    if n_hours < 300:
        raise ValueError("n_hours must be at least 300 for a meaningful backtest")
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours, dtype=float)
    hour = t % 24
    weekday = (t // 24) % 7
    daily = 8.0 * np.sin(2 * np.pi * (hour - 7) / 24) + 3.0 * np.cos(4 * np.pi * hour / 24)
    weekend = np.where(weekday >= 5, -7.0, 2.5)
    trend = 0.008 * t
    event = np.zeros(n_hours)
    # Reproducible demand pulses; event times are part of the synthetic generator only.
    event[(t.astype(int) % (24 * 19)) == 9] = 13.0
    event[(t.astype(int) % (24 * 31)) == 18] = -8.0
    noise = rng.normal(0.0, 2.2, n_hours)
    return (45.0 + daily + weekend + trend + event + noise).astype(float)
