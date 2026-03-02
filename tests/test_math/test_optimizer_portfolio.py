"""Tests for BasanosEngine.portfolio and BasanosConfig validation.

These tests cover the construction of a Portfolio from BasanosEngine as well as
validation logic in BasanosConfig (corr >= vola). Keeping them lightweight with
small synthetic datasets ensures determinism and speed.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from basanos.math import BasanosConfig, BasanosEngine


def _make_prices_mu(n: int = 64) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create synthetic prices and a bounded signal (mu) aligned by date.

    - Prices: two assets with smooth trends/oscillations.
    - Mu: tanh of sin/cos waves to keep values in [-1, 1].
    """
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    a = pl.Series([100.0 + 2.0 * np.cos(0.20 * i) for i in range(n)], dtype=pl.Float64)
    b = pl.Series([200.0 + 5.0 * np.sin(0.15 * i) for i in range(n)], dtype=pl.Float64)
    prices = pl.DataFrame({"date": dates, "A": a, "B": b})

    theta = np.linspace(0.0, 3.0 * np.pi, num=n)
    mu_a = np.tanh(np.sin(theta))
    mu_b = np.tanh(np.cos(theta))
    mu = pl.DataFrame({"date": dates, "A": pl.Series(mu_a, dtype=pl.Float64), "B": pl.Series(mu_b, dtype=pl.Float64)})

    return prices, mu


def test_basanos_portfolio_builds_portfolio_with_finite_nav_and_positions():
    """BasanosEngine.portfolio should return a Portfolio with sane outputs.

    We verify that:
    - a Portfolio object is returned,
    - its risk positions DataFrame has expected columns and finite values after warmup,
    - NAV series is finite and of expected length.
    """
    prices, mu = _make_prices_mu(96)

    cfg = BasanosConfig(vola=16, corr=24, clip=3.5, shrink=0.5, aum=1e6)
    engine = BasanosEngine(prices=prices, cfg=cfg, mu=mu)

    assert engine.cfg == cfg
    assert engine.assets == ["A", "B"]

    portfolio = engine.portfolio

    # Basic Portfolio interface checks (avoid importing plots/stats heavy parts)
    rp = portfolio.cashposition

    assert isinstance(rp, pl.DataFrame)
    assert rp.columns[0] == "date"
    assert set(rp.columns[1:]) == {"A", "B"}
    # After correlation warmup, positions should be finite
    tail = rp.tail(rp.height - cfg.corr)
    for c in ("A", "B"):
        assert tail[c].null_count() == 0
        assert tail[c].is_finite().all()

    # NAV and returns should be finite as well
    nav = portfolio.nav_accumulated
    assert nav.height == prices.height
    assert nav["NAV_accumulated"].is_finite().all()


def test_basanos_config_validator_enforces_corr_ge_vola():
    """BasanosConfig should raise when corr < vola and accept equal/greater.

    This covers the Pydantic v2 field_validator that inspects ValidationInfo.
    """
    # Acceptable: corr == vola
    _ = BasanosConfig(vola=20, corr=20, clip=3.0, shrink=0.5, aum=1e6)
    # Acceptable: corr > vola
    _ = BasanosConfig(vola=12, corr=24, clip=2.0, shrink=0.25, aum=1e6)

    # Invalid: corr < vola -> ValueError
    with pytest.raises(ValueError, match=r".*"):
        _ = BasanosConfig(vola=30, corr=10, clip=4.0, shrink=0.7, aum=1e6)
