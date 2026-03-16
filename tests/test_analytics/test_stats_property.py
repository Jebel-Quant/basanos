"""Property-based tests for basanos.analytics._stats (Stats class).

Uses Hypothesis to systematically explore edge cases for the statistical
metrics exposed by ``Stats``.

Properties under test
---------------------
volatility
  - Always non-negative for any finite returns series.
  - Zero for a constant series.

best / worst
  - ``best() >= worst()`` for any series with at least one finite value.

avg_win / avg_loss
  - ``avg_win >= 0``: the mean of positive returns cannot be negative.
  - ``avg_loss <= 0``: the mean of negative returns cannot be positive.

sharpe
  - Returns NaN for a constant (zero-variance) series.

conditional_value_at_risk
  - Bug fix: when no empirical observations fall below the parametric VaR
    threshold, the function must return NaN rather than the previous
    incorrect fallback of 0.0.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from basanos.analytics._stats import Stats

# ─── Helpers ─────────────────────────────────────────────────────────────────

_FINITE_RETURN = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_POSITIVE_RETURN = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)


def _make_stats(returns: list[float]) -> Stats:
    """Build a minimal Stats object with a daily date column and one asset column."""
    n = len(returns)
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)
    df = pl.DataFrame({"date": dates, "returns": pl.Series(returns, dtype=pl.Float64)})
    return Stats(data=df)


# ─── volatility ───────────────────────────────────────────────────────────────


@given(returns=st.lists(_FINITE_RETURN, min_size=2, max_size=50))
@settings(max_examples=300)
def test_volatility_is_non_negative(returns: list[float]) -> None:
    """volatility() must always be >= 0 for any finite returns series."""
    s = _make_stats(returns)
    vol = s.volatility()["returns"]
    assert vol >= 0.0


@given(
    constant=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    n=st.integers(min_value=2, max_value=30),
)
@settings(max_examples=200)
def test_volatility_of_constant_series_is_zero(constant: float, n: int) -> None:
    """A constant series has zero variance; annualized volatility must be 0."""
    s = _make_stats([constant] * n)
    vol = s.volatility()["returns"]
    assert vol == pytest.approx(0.0, abs=1e-12)


# ─── best / worst ────────────────────────────────────────────────────────────


@given(returns=st.lists(_FINITE_RETURN, min_size=1, max_size=50))
@settings(max_examples=300)
def test_best_is_at_least_worst(returns: list[float]) -> None:
    """best() must always be >= worst() for any non-empty finite series."""
    s = _make_stats(returns)
    best_val = s.best()["returns"]
    worst_val = s.worst()["returns"]
    assert best_val is not None
    assert worst_val is not None
    assert best_val >= worst_val


# ─── avg_win / avg_loss ───────────────────────────────────────────────────────


@given(returns=st.lists(_FINITE_RETURN, min_size=2, max_size=50))
@settings(max_examples=300)
def test_avg_win_is_non_negative(returns: list[float]) -> None:
    """avg_win() is the mean of positive returns: it must always be >= 0."""
    s = _make_stats(returns)
    avg_win = s.avg_win()["returns"]
    assert avg_win >= 0.0


@given(returns=st.lists(_FINITE_RETURN, min_size=2, max_size=50))
@settings(max_examples=300)
def test_avg_loss_is_non_positive(returns: list[float]) -> None:
    """avg_loss() is the mean of negative returns: it must always be <= 0."""
    s = _make_stats(returns)
    avg_loss = s.avg_loss()["returns"]
    assert avg_loss <= 0.0


# ─── sharpe ──────────────────────────────────────────────────────────────────


@given(
    constant=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    n=st.integers(min_value=2, max_value=30),
)
@settings(max_examples=200)
def test_sharpe_of_constant_series_is_nan(constant: float, n: int) -> None:
    """sharpe() must return NaN for a zero-variance (constant) series."""
    s = _make_stats([constant] * n)
    sharpe_val = s.sharpe()["returns"]
    assert math.isnan(sharpe_val)


# ─── conditional_value_at_risk (bug fix) ─────────────────────────────────────


@given(returns=st.lists(_POSITIVE_RETURN, min_size=5, max_size=30))
@settings(max_examples=300)
def test_cvar_is_nan_when_no_obs_below_var(returns: list[float]) -> None:
    """CVaR must be NaN (not 0.0) when no empirical observations fall below VaR.

    This property tests the fix for a bug where ``conditional_value_at_risk``
    returned ``0.0`` instead of ``nan`` when the parametric VaR threshold was
    below the minimum of the empirical series (i.e., the empirical filter was
    empty).

    Using strictly positive returns with a very low alpha (0.001) reliably
    produces a parametric VaR that lies far below the minimum observation,
    triggering the empty-filter path.
    """
    s = _make_stats(returns)
    var_val = s.value_at_risk(alpha=0.001)["returns"]
    min_return = min(returns)

    # Only assert when the condition is actually triggered.
    if var_val < min_return:
        cvar_val = s.conditional_value_at_risk(alpha=0.001)["returns"]
        assert math.isnan(cvar_val), f"Expected NaN when VaR={var_val:.4f} < min={min_return:.4f}, got {cvar_val}"


def test_cvar_is_nan_for_empty_empirical_tail() -> None:
    """Deterministic: CVaR must be NaN when all observations exceed the parametric VaR.

    Constructs a series of large positive returns where the parametric VaR at
    alpha=0.001 is guaranteed to fall below the minimum observation (verified by
    an explicit assertion), so the empirical filter is always empty.
    """
    returns = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0]
    s = _make_stats(returns)
    var_val = s.value_at_risk(alpha=0.001)["returns"]
    # Guard: confirm the precondition holds for this fixed input.
    assert var_val < min(returns), f"Precondition failed: VaR={var_val} >= min={min(returns)}"
    cvar_val = s.conditional_value_at_risk(alpha=0.001)["returns"]
    assert math.isnan(cvar_val), f"Expected NaN for empty empirical tail, got {cvar_val}"
