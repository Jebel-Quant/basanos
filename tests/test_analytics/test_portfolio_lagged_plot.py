"""Tests for Portfolio.lagged_performance_plot in taipan.math.portfolio.

These tests ensure the method returns a Plotly Figure with a line per
lag (default 0..4), proper naming, and that it serializes without error.
"""

from __future__ import annotations

from datetime import timedelta

import plotly.graph_objects as go
import polars as pl

from basanos.analytics import Portfolio


def _make_prices_positions(n: int = 6) -> tuple[pl.DataFrame, pl.DataFrame]:
    start = pl.date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([100.0 + 1.0 * i for i in range(n)], dtype=pl.Float64),
            "B": pl.Series([50.0 + 0.5 * i for i in range(n)], dtype=pl.Float64),
        }
    )
    positions = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([1000.0] * n, dtype=pl.Float64),
            "B": pl.Series([500.0] * n, dtype=pl.Float64),
        }
    )
    return prices, positions


def test_lagged_performance_plot_default_returns_five_traces_and_serializes():
    """Default call should plot lags 0..4 as separate traces and be serializable."""
    prices, positions = _make_prices_positions(8)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)

    fig = pf.plots.lagged_performance_plot()

    assert isinstance(fig, go.Figure)
    # Expect five traces named 'lag 0'..'lag 4'
    assert len(fig.data) == 5
    expected_names = {f"lag {i}" for i in range(5)}
    assert {tr.name for tr in fig.data} == expected_names

    # Serialization should not raise
    _ = fig.to_dict()


def test_lagged_performance_plot_log_scale_sets_axis():
    """log_scale=True should set the first y-axis to logarithmic scale."""
    prices, positions = _make_prices_positions(10)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)

    fig = pf.plots.lagged_performance_plot(log_scale=True)
    assert isinstance(fig, go.Figure)
    assert fig.layout.yaxis.type == "log"
