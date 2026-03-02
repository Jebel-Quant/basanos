"""Validation tests for Portfolio.lag and Portfolio.smoothed_holding.

These tests explicitly exercise input validation branches in
src/taipan/analytics/portfolio.py to ensure full coverage:
- Portfolio.lag raises TypeError when n is not an int.
- Portfolio.smoothed_holding raises ValueError when n < 0.
- Portfolio.smoothed_holding raises TypeError when n is not an int.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from basanos.analytics import Portfolio


def _make_prices_positions(n: int = 5) -> tuple[pl.DataFrame, pl.DataFrame]:
    start = date(2020, 1, 1)
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


def test_portfolio_lag_type_error_on_non_int():
    """Portfolio.lag should raise TypeError when n is not an int."""
    prices, positions = _make_prices_positions(n=6)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    with pytest.raises(TypeError):
        _ = pf.lag("1")  # type: ignore[arg-type]


def test_portfolio_smoothed_holding_negative_raises_value_error():
    """Portfolio.smoothed_holding should raise ValueError when n < 0."""
    prices, positions = _make_prices_positions(n=6)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    with pytest.raises(ValueError, match=r".*"):
        _ = pf.smoothed_holding(-1)


def test_portfolio_smoothed_holding_type_error_on_non_int():
    """Portfolio.smoothed_holding should raise TypeError when n is not an int."""
    prices, positions = _make_prices_positions(n=6)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    with pytest.raises(TypeError):
        _ = pf.smoothed_holding(1.5)  # type: ignore[arg-type]
