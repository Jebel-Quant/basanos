"""Tests for Portfolio.lead_lag_ir_plot in taipan.math.portfolio.

These tests ensure that the bar chart is produced with the expected number
of bars (lags -10..+19 inclusive), correct labeling, and that selected bar
heights match direct Sharpe computations from the Stats facade on lagged
portfolios.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl

from basanos.analytics import Portfolio


def _make_prices_positions(n: int = 30) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build a simple deterministic dataset with varying returns and weights.

    We create two assets with smooth but distinct dynamics to avoid zero-variance
    edge cases in Sharpe calculation.
    """
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    # Asset A grows approximately exponentially, Asset B oscillates
    a = pl.Series([100.0 * (1.02**i) for i in range(n)], dtype=pl.Float64)
    b = pl.Series([200.0 + 10.0 * np.sin(0.2 * i) for i in range(n)], dtype=pl.Float64)

    prices = pl.DataFrame({"date": dates, "A": a, "B": b})

    # Positions vary to ensure non-trivial profits
    pos_a = pl.Series([1000.0 + 10.0 * i for i in range(n)], dtype=pl.Float64)
    pos_b = pl.Series([500.0 + 5.0 * ((-1) ** i) for i in range(n)], dtype=pl.Float64)
    positions = pl.DataFrame({"date": dates, "A": pos_a, "B": pos_b})

    return prices, positions


def test_lead_lag_ir_plot_basic_structure_and_values():
    """lead_lag_ir_plot returns a Figure with bars for lags -10..+19 and valid values."""
    prices, positions = _make_prices_positions(n=60)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    fig = pf.plots.lead_lag_ir_plot()  # defaults to -10..+19
    assert isinstance(fig, go.Figure)

    # One trace with 30 bars (inclusive range -10..+19)
    assert len(fig.data) == 1
    bar = fig.data[0]
    x = list(bar.x)
    y = list(bar.y)

    expected_lags = list(range(-10, 20))
    assert x == expected_lags
    assert len(y) == len(expected_lags)

    # Cross-check a couple of lags against direct Stats computation
    for n in (-10, 0, 5, 19):
        pf_n = pf if n == 0 else pf.lag(n)
        sharpe_n = pf_n.stats.sharpe()["returns"]
        idx = expected_lags.index(n)
        assert np.isclose(y[idx], sharpe_n, rtol=1e-12, atol=1e-12)

    # Figure serializes
    _ = fig.to_dict()
