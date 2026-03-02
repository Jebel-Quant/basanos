"""Tests for Portfolio.lag in taipan.math.portfolio.

These tests verify that lagging cash positions by positive, zero, and negative
steps behaves as expected and that downstream profit computation remains
consistent with shifted weights.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
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
            # Simple price path with constant daily return of +10% then +10%
            "A": pl.Series([100.0 * (1.1**i) for i in range(n)], dtype=pl.Float64),
            # Asset B oscillates to test sign/zero effects
            "B": pl.Series([200.0 + 5.0 * ((-1) ** i) for i in range(n)], dtype=pl.Float64),
        }
    )
    # Positions: constant for A, increasing for B
    positions = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([1000.0] * n, dtype=pl.Float64),
            "B": pl.Series([100.0 * i for i in range(n)], dtype=pl.Float64),
        }
    )
    return prices, positions


def test_lag_positive_shifts_weights_down_and_preserves_date():
    """lag(+1) should shift numeric columns down by one and preserve 'date'."""
    prices, positions = _make_prices_positions(n=5)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    pf_lag1 = pf.lag(1)
    assert isinstance(pf_lag1, Portfolio)
    assert pf_lag1.aum == pf.aum

    # Date column preserved
    assert pf_lag1.cashposition.columns[0] == "date"

    # Column-wise shift: first row becomes null after shift for numeric cols
    assets = [col for col in positions.columns if col != "date" and positions[col].dtype.is_numeric()]
    for c in assets:
        s0 = pf.cashposition[c]
        s1 = pf_lag1.cashposition[c]
        assert s1.null_count() == 1
        # After dropping the first null, values should equal original[:-1]
        assert np.allclose(s1.drop_nulls().to_numpy(), s0[:-1].to_numpy(), rtol=0, atol=0)

    # Profits compute without error on lagged portfolio
    _ = pf_lag1.profit


def test_lag_negative_leads_weights_and_last_becomes_null():
    """lag(-1) should lead numeric columns; last element becomes null."""
    prices, positions = _make_prices_positions(n=5)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    pf_lead1 = pf.lag(-1)
    assets = [col for col in positions.columns if col != "date" and positions[col].dtype.is_numeric()]
    for c in assets:
        s0 = pf.cashposition[c]
        s1 = pf_lead1.cashposition[c]
        assert s1.null_count() == 1
        # After dropping the last null (implicitly by comparing upto -1), should match original[1:]
        assert np.allclose(s1.head(len(s1) - 1).to_numpy(), s0[1:].to_numpy(), rtol=0, atol=0)

    # Profits compute without error on leading portfolio
    _ = pf_lead1.profit


def test_lag_zero_returns_same_portfolio_object_or_equal_data():
    """lag(0) should be a no-op: same object or equal data content and AUM preserved."""
    prices, positions = _make_prices_positions(n=4)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)

    pf0 = pf.lag(0)
    # Returning the same instance is acceptable; otherwise, data equality must hold
    if pf0 is pf:
        assert True
    else:
        assert pf0.aum == pf.aum
        assert pf0.prices.frame_equal(pf.prices)
        assert pf0.cashposition.frame_equal(pf.cashposition)


def test_lag_raises_typeerror_for_non_int():
    """Passing a non-int to lag should raise TypeError."""
    prices, positions = _make_prices_positions(n=3)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)

    with pytest.raises(TypeError):
        _ = pf.lag(1.5)  # type: ignore[arg-type]
