"""Property-based tests for basanos.analytics.portfolio (Portfolio class).

Uses Hypothesis to systematically explore invariants for the Portfolio
dataclass and its derived series.

Properties under test
---------------------
profits
  - Shape matches the input prices shape.
  - All numeric columns are finite (no NaN, no Inf).

highwater
  - The cumulative maximum is always non-decreasing.

drawdown
  - Always non-negative (highwater - NAV_accumulated >= 0).
  - Equals highwater - NAV_accumulated element-wise.

Returns:
  - Equals profit / aum element-wise.

truncate
  - Preserves aum.
  - Returns a portfolio with height <= the original.

lag
  - lag(0) returns the original Portfolio object unchanged.

tilt / timing
  - tilt.cashposition + timing.cashposition equals the original cashposition.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st

from basanos.analytics import Portfolio

# ─── Shared strategies ────────────────────────────────────────────────────────

_PRICE = st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False)
_POSITION = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
_AUM = st.floats(min_value=1e3, max_value=1e9, allow_nan=False, allow_infinity=False)


@st.composite
def _portfolio(draw: st.DrawFn) -> Portfolio:
    """Strategy that draws a valid Portfolio without a date column."""
    n_rows = draw(st.integers(min_value=3, max_value=20))
    n_assets = draw(st.integers(min_value=1, max_value=3))
    aum = draw(_AUM)
    col_names = [f"asset_{i}" for i in range(n_assets)]
    price_arrays = [draw(np_st.arrays(dtype=np.float64, shape=n_rows, elements=_PRICE)) for _ in range(n_assets)]
    pos_arrays = [draw(np_st.arrays(dtype=np.float64, shape=n_rows, elements=_POSITION)) for _ in range(n_assets)]
    prices_df = pl.DataFrame(
        {name: pl.Series(arr, dtype=pl.Float64) for name, arr in zip(col_names, price_arrays, strict=False)}
    )
    pos_df = pl.DataFrame(
        {name: pl.Series(arr, dtype=pl.Float64) for name, arr in zip(col_names, pos_arrays, strict=False)}
    )
    return Portfolio.from_cash_position(prices=prices_df, cash_position=pos_df, aum=aum)


# ─── profits ──────────────────────────────────────────────────────────────────


@given(pf=_portfolio())
@settings(max_examples=200)
def test_profits_shape_matches_prices(pf: Portfolio) -> None:
    """profits() must have the same shape as the input prices DataFrame."""
    assert pf.profits.shape == pf.prices.shape


@given(pf=_portfolio())
@settings(max_examples=200)
def test_profits_all_values_are_finite(pf: Portfolio) -> None:
    """All numeric profit values must be finite (no NaN, no Inf)."""
    for col in pf.assets:
        assert pf.profits[col].is_finite().all(), f"Non-finite values in profits column '{col}'"


# ─── highwater ────────────────────────────────────────────────────────────────


@given(pf=_portfolio())
@settings(max_examples=200)
def test_highwater_is_non_decreasing(pf: Portfolio) -> None:
    """Highwater must be monotonically non-decreasing."""
    hw = pf.highwater["highwater"].to_list()
    for i in range(1, len(hw)):
        assert hw[i] >= hw[i - 1], f"highwater decreased at index {i}: {hw[i - 1]} -> {hw[i]}"


# ─── drawdown ─────────────────────────────────────────────────────────────────


@given(pf=_portfolio())
@settings(max_examples=200)
def test_drawdown_is_non_negative(pf: Portfolio) -> None:
    """Drawdown must always be >= 0 (NAV cannot exceed the high-water mark)."""
    dd = pf.drawdown["drawdown"].to_list()
    assert all(v >= -1e-10 for v in dd), f"Negative drawdown values: {[v for v in dd if v < -1e-10]}"


@given(pf=_portfolio())
@settings(max_examples=200)
def test_drawdown_equals_highwater_minus_nav(pf: Portfolio) -> None:
    """Drawdown must equal highwater - NAV_accumulated element-wise."""
    df = pf.drawdown
    for i, (d, h, nav) in enumerate(
        zip(df["drawdown"].to_list(), df["highwater"].to_list(), df["NAV_accumulated"].to_list(), strict=False)
    ):
        assert d == pytest.approx(h - nav, abs=1e-8), (
            f"drawdown mismatch at row {i}: drawdown={d}, highwater={h}, NAV={nav}"
        )


# ─── returns ──────────────────────────────────────────────────────────────────


@given(pf=_portfolio())
@settings(max_examples=200)
def test_returns_equal_profit_over_aum(pf: Portfolio) -> None:
    """Returns must equal profit / aum element-wise."""
    r_df = pf.returns
    for p, r in zip(r_df["profit"].to_list(), r_df["returns"].to_list(), strict=False):
        assert r == pytest.approx(p / pf.aum, rel=1e-9, abs=1e-14)


# ─── truncate ─────────────────────────────────────────────────────────────────


@given(pf=_portfolio(), data=st.data())
@settings(max_examples=200)
def test_truncate_height_le_original(pf: Portfolio, data: st.DataObject) -> None:
    """truncate() must return a portfolio whose height does not exceed the original."""
    n = pf.prices.height
    start_idx = data.draw(st.integers(min_value=0, max_value=n - 1))
    end_idx = data.draw(st.integers(min_value=start_idx, max_value=n - 1))
    truncated = pf.truncate(start=start_idx, end=end_idx)
    assert truncated.prices.height <= pf.prices.height


@given(pf=_portfolio())
@settings(max_examples=200)
def test_truncate_preserves_aum(pf: Portfolio) -> None:
    """truncate() must leave aum unchanged."""
    truncated = pf.truncate(start=0, end=pf.prices.height - 1)
    assert truncated.aum == pf.aum


# ─── lag ──────────────────────────────────────────────────────────────────────


@given(pf=_portfolio())
@settings(max_examples=200)
def test_lag_zero_returns_same_portfolio(pf: Portfolio) -> None:
    """lag(0) must return exactly the original Portfolio object."""
    assert pf.lag(0) is pf


# ─── tilt / timing decomposition ─────────────────────────────────────────────


@given(pf=_portfolio())
@settings(max_examples=200)
def test_tilt_plus_timing_equals_original(pf: Portfolio) -> None:
    """tilt.cashposition + timing.cashposition must equal the original cashposition."""
    for col in pf.assets:
        orig = pf.cashposition[col].to_numpy()
        tilt_col = pf.tilt.cashposition[col].to_numpy()
        timing_col = pf.timing.cashposition[col].to_numpy()
        np.testing.assert_allclose(tilt_col + timing_col, orig, rtol=1e-9, atol=1e-9)
