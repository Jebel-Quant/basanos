"""Unit tests for taipan.math.taipan.optimize.

These tests validate that the optimizer:
- returns a Polars DataFrame with the expected schema (date + asset columns),
- yields finite risk positions after a warmup period when correlations become defined,
- and produces all-zero positions when the expected returns (mu) are zero.

Tests are skipped if pandas is not available since taipan.optimize relies on
pandas' EWM correlation internally.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from basanos.math import TaipanConfig, TaipanEngine


@pytest.fixture
def prices() -> pl.DataFrame:
    """Create a price frame with a date column and two assets A, B.

    The first ``corr - 1`` rows are set to null (None) for numeric assets to
    avoid undefined EWM correlation during warmup; the optimizer will skip
    those rows via its row mask. Subsequent rows follow smooth random walks.
    """
    n = 120
    rng = np.random.default_rng(0)

    # Build base series
    r_a = rng.normal(0.0, 0.5, size=n)
    r_b = rng.normal(0.0, 0.7, size=n)
    p_a = 100.0 + np.cumsum(r_a)
    p_b = 200.0 + np.cumsum(r_b)

    # Date range
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True)

    return pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(p_a, dtype=pl.Float64),
            "B": pl.Series(p_b, dtype=pl.Float64),
        }
    )


@pytest.fixture
def mu(prices) -> pl.DataFrame:
    """Create a deterministic mu DataFrame aligned with prices.

    Uses simple sinusoidal patterns for A/B; the 'date' column is included
    for alignment but ignored by optimize (it selects only numeric columns).
    """
    n = prices.height
    theta = np.linspace(0.0, 4.0 * np.pi, num=n)
    mu_a = np.tanh(np.sin(theta))
    mu_b = np.tanh(np.cos(theta))
    return pl.DataFrame(
        {
            "date": prices["date"],
            "A": pl.Series(mu_a, dtype=pl.Float64),
            "B": pl.Series(mu_b, dtype=pl.Float64),
        }
    )


def test_optimize_returns_frame_with_expected_schema_and_finite_after_warmup(prices, mu):
    """Optimize should return a DataFrame with date + asset cols and finite values after warmup.

    We construct prices with leading nulls for the first ``corr-1`` rows, so those
    rows are skipped. After that warmup, the resulting risk positions should be finite.
    """
    n = 120

    cfg = TaipanConfig(vola=16, corr=20, clip=3.5, shrink=0.5, aum=1e6)

    cp = TaipanEngine(prices=prices, cfg=cfg, mu=mu).cash_position

    # Basic type/shape checks
    assert isinstance(cp, pl.DataFrame)
    assert cp.height == prices.height
    assert cp.columns[0] == "date"
    assert set(cp.columns[1:]) == {"A", "B"}

    # Finite values after warmup window (skip first corr rows)
    tail = cp.tail(n - cfg.corr)
    for c in ("A", "B"):
        s = tail[c]
        # Some values should be non-null and finite
        assert s.null_count() == 0
        assert s.is_finite().all()


def test_optimize_with_zero_mu_returns_zero_positions(prices):
    """When mu is all zeros, the optimizer should return zero risk positions for assets.

    This serves as a sanity check that the linear solver responds to a zero RHS
    with a zero solution when scaled/normalized appropriately.
    """
    n = 80
    cfg = TaipanConfig(corr=20, vola=12, clip=4.0, shrink=0.7, aum=1e6)

    # Zero mu for numeric assets; keep date for alignment
    mu = pl.DataFrame(
        {
            "date": prices["date"],
            "A": pl.Series([0.0] * prices.height, dtype=pl.Float64),
            "B": pl.Series([0.0] * prices.height, dtype=pl.Float64),
        }
    )

    cp = TaipanEngine(prices=prices, cfg=cfg, mu=mu).cash_position

    # After warmup, positions should be exactly zero (within numerical tolerance)
    tail = cp.tail(n - cfg.corr)
    for c in ("A", "B"):
        vals = tail[c].to_numpy()
        assert np.allclose(vals, 0.0, rtol=0, atol=0)
