"""Tests for Plots.correlation_heatmap.

These tests validate that the correlation heatmap method returns a
Plotly Figure with a Heatmap trace, reasonable axis labels, and that
it serializes. We also exercise custom arguments (frame, name, title)
so the new code paths are covered.
"""

from __future__ import annotations

from datetime import date, timedelta

import plotly.graph_objects as go
import polars as pl

from basanos.analytics import Portfolio


def _make_prices_positions(n: int = 30) -> tuple[pl.DataFrame, pl.DataFrame]:
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    # Two assets with simple but distinct dynamics
    a = pl.Series([100.0 + 0.5 * i for i in range(n)], dtype=pl.Float64)
    b = pl.Series([200.0 + 5.0 * ((-1) ** i) for i in range(n)], dtype=pl.Float64)

    prices = pl.DataFrame({"date": dates, "A": a, "B": b})
    positions = pl.DataFrame({"date": dates, "A": pl.Series([1000.0] * n), "B": pl.Series([500.0] * n)})

    return prices, positions


def test_correlation_heatmap_default_trace_and_serialize():
    """Default call returns Heatmap trace and is serializable; axes align."""
    prices, positions = _make_prices_positions(40)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    fig = pf.plots.correlation_heatmap()  # use defaults (frame=None, name="portfolio")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    # The first trace should be a Heatmap
    assert isinstance(fig.data[0], go.Heatmap)

    # Axis labels should match the correlation matrix dimension
    x_labels = list(fig.data[0].x)
    y_labels = list(fig.data[0].y)
    assert x_labels == y_labels
    assert len(x_labels) >= 2  # at least the assets and the portfolio

    # Correlation values fall within [-1, 1]
    z_vals = fig.data[0].z
    flat = [v for row in z_vals for v in row]
    assert all(-1.000001 <= float(v) <= 1.000001 for v in flat)

    # Should be serializable
    _ = fig.to_dict()


def test_correlation_heatmap_custom_args_title_and_name():
    """Custom frame/name/title are respected and output remains a Heatmap."""
    prices, positions = _make_prices_positions(25)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    custom_title = "My Correlations"
    fig = pf.plots.correlation_heatmap(frame=pf.prices, name="my_port", title=custom_title)

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == custom_title

    # Still a Heatmap and serializable
    assert isinstance(fig.data[0], go.Heatmap)
    _ = fig.to_dict()
