"""Comprehensive tests for taipan.analytics.plots.Plots.

These tests exercise all plotting methods exposed via the Portfolio.plots
facade:
- snapshot
- lagged_performance_plot
- smoothed_holdings_performance_plot
- lead_lag_ir_plot

The goal is to ensure each method returns a valid Plotly Figure with expected
trace structure, titles, and that optional log scaling is applied.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl

from basanos.analytics import Portfolio


def _make_prices_positions(n: int = 40) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create a small deterministic price/position dataset with a date column.

    Prices: two assets A (upward trend) and B (oscillatory) to avoid degenerate
    variance. Positions vary over time to produce non-trivial profits.
    """
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    a = pl.Series([100.0 * (1.01**i) for i in range(n)], dtype=pl.Float64)
    b = pl.Series([200.0 + 5.0 * np.sin(0.15 * i) for i in range(n)], dtype=pl.Float64)
    prices = pl.DataFrame({"date": dates, "A": a, "B": b})

    pos_a = pl.Series([1000.0 + 2.0 * i for i in range(n)], dtype=pl.Float64)
    pos_b = pl.Series([500.0 + (i % 3) for i in range(n)], dtype=pl.Float64)
    positions = pl.DataFrame({"date": dates, "A": pos_a, "B": pos_b})

    return prices, positions


def test_snapshot_returns_figure_with_expected_traces_and_log_scale():
    """Snapshot should return a 2-trace figure and honor log_scale on y-axis."""
    prices, positions = _make_prices_positions(n=60)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    # Default scale
    fig = pf.plots.snapshot()
    assert isinstance(fig, go.Figure)
    # Two traces: NAV (row1) and Drawdown (row2)
    assert len(fig.data) == 4
    names = {trace.name for trace in fig.data}
    assert {"NAV", "Drawdown"}.issubset(names)
    # Layout basics
    assert fig.layout.title.text
    assert "Performance" in fig.layout.title.text
    assert fig.layout.hovermode in ("x unified", "x", "x unified")

    # Log scale branch
    fig_log = pf.plots.snapshot(log_scale=True)
    assert isinstance(fig_log, go.Figure)
    # Primary y-axis should be log
    assert getattr(fig_log.layout.yaxis, "type", None) == "log"
    _ = fig_log.to_dict()  # serializes


def test_lagged_performance_plot_traces_and_log_scale():
    """lagged_performance_plot returns 5 traces by default and supports log scale."""
    prices, positions = _make_prices_positions(n=50)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    fig = pf.plots.lagged_performance_plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 5
    names = [t.name for t in fig.data]
    assert names == [f"lag {i}" for i in range(5)]
    # Ensure series exist and are finite (first points may be equal to AUM)
    for tr in fig.data:
        assert len(tr.x) == len(tr.y) > 0

    # Log scale branch
    fig_log = pf.plots.lagged_performance_plot(log_scale=True)
    assert getattr(fig_log.layout.yaxis, "type", None) == "log"
    _ = fig_log.to_dict()


def test_smoothed_holdings_performance_plot_traces_and_log_scale():
    """smoothed_holdings_performance_plot returns 5 traces and supports log scale."""
    prices, positions = _make_prices_positions(n=50)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    fig = pf.plots.smoothed_holdings_performance_plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 5
    names = [t.name for t in fig.data]
    assert names == [f"smooth {i}" for i in range(5)]
    for tr in fig.data:
        assert len(tr.x) == len(tr.y) > 0

    fig_log = pf.plots.smoothed_holdings_performance_plot(log_scale=True)
    assert getattr(fig_log.layout.yaxis, "type", None) == "log"
    _ = fig_log.to_dict()


def test_lead_lag_ir_plot_basic_integration_from_plots_facade():
    """lead_lag_ir_plot accessible via plots facade returns one Bar trace for default lags."""
    prices, positions = _make_prices_positions(n=60)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    fig = pf.plots.lead_lag_ir_plot()  # defaults -10..+19
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)
    x = list(fig.data[0].x)
    assert x == list(range(-10, 20))
    _ = fig.to_dict()


def test_lead_lag_ir_plot_swaps_when_start_greater_than_end():
    """When start > end, the function swaps them and still plots inclusive range."""
    prices, positions = _make_prices_positions(n=40)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    fig = pf.plots.lead_lag_ir_plot(start=5, end=-5)
    xs = list(fig.data[0].x)
    assert xs == list(range(-5, 6))  # inclusive after swap


def test_lead_lag_ir_plot_type_validation_raises():
    """Non-integer start/end should raise TypeError in lead_lag_ir_plot."""
    prices, positions = _make_prices_positions(n=20)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    import pytest

    with pytest.raises(TypeError):
        _ = pf.plots.lead_lag_ir_plot(start=-10.0, end=10)
    with pytest.raises(TypeError):
        _ = pf.plots.lead_lag_ir_plot(start=-10, end="19")


essential_ints = [0, 1, 2]


def test_lagged_performance_plot_type_validation_raises():
    """Lags must be a list of ints; other types or contents should raise TypeError."""
    prices, positions = _make_prices_positions(n=20)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)
    import pytest

    with pytest.raises(TypeError):
        _ = pf.plots.lagged_performance_plot(lags=(0, 1, 2))  # not a list
    with pytest.raises(TypeError):
        _ = pf.plots.lagged_performance_plot(lags=[0, "1", 2])  # non-int element


def test_smoothed_holdings_performance_plot_type_validation_raises():
    """Windows must be a list of non-negative ints; invalid inputs raise TypeError."""
    prices, positions = _make_prices_positions(n=20)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)
    import pytest

    with pytest.raises(TypeError):
        _ = pf.plots.smoothed_holdings_performance_plot(windows=(0, 1, 2))  # not a list
    with pytest.raises(TypeError):
        _ = pf.plots.smoothed_holdings_performance_plot(windows=[0, -1, 2])  # negative element
    with pytest.raises(TypeError):
        _ = pf.plots.smoothed_holdings_performance_plot(windows=[0, "2"])  # non-int element
