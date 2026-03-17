"""Tests for basanos.math._async (AsyncBasanosEngine)."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from basanos.exceptions import MissingDateColumnError, ShapeMismatchError
from basanos.math import AsyncBasanosEngine, BasanosConfig

# ─── Helpers ─────────────────────────────────────────────────────────────────

_N = 80
_START = date(2020, 1, 1)


def _make_engine(n: int = _N) -> AsyncBasanosEngine:
    """Return a small AsyncBasanosEngine for testing."""
    rng = np.random.default_rng(42)
    p_a = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=n))
    p_b = 200.0 + np.cumsum(rng.normal(0.0, 0.7, size=n))
    dates = pl.date_range(start=_START, end=_START + timedelta(days=n - 1), interval="1d", eager=True)
    prices = pl.DataFrame({"date": dates, "A": pl.Series(p_a, dtype=pl.Float64), "B": pl.Series(p_b, dtype=pl.Float64)})
    theta = np.linspace(0.0, 4.0 * np.pi, num=n)
    mu = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.tanh(np.sin(theta)), dtype=pl.Float64),
            "B": pl.Series(np.tanh(np.cos(theta)), dtype=pl.Float64),
        }
    )
    cfg = BasanosConfig(vola=16, corr=32, clip=3.0, shrink=0.5, aum=1e6)
    return AsyncBasanosEngine(prices=prices, mu=mu, cfg=cfg)


# ─── Construction ─────────────────────────────────────────────────────────────


def test_async_engine_construction_succeeds():
    """AsyncBasanosEngine can be constructed with valid inputs."""
    engine = _make_engine()
    assert isinstance(engine, AsyncBasanosEngine)


def test_async_engine_validation_raises_missing_date():
    """AsyncBasanosEngine raises MissingDateColumnError when prices lacks 'date'."""
    rng = np.random.default_rng(0)
    prices = pl.DataFrame({"A": rng.lognormal(size=10)})
    mu = pl.DataFrame({"date": list(range(10)), "A": rng.normal(size=10)})
    cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
    with pytest.raises(MissingDateColumnError):
        AsyncBasanosEngine(prices=prices, mu=mu, cfg=cfg)


def test_async_engine_validation_raises_shape_mismatch():
    """AsyncBasanosEngine raises ShapeMismatchError when shapes differ."""
    rng = np.random.default_rng(0)
    dates = pl.Series("date", list(range(10)))
    prices = pl.DataFrame({"date": dates, "A": rng.lognormal(size=10)})
    mu_dates = pl.Series("date", list(range(5)))
    mu = pl.DataFrame({"date": mu_dates, "A": rng.normal(size=5)})
    cfg = BasanosConfig(vola=2, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    with pytest.raises(ShapeMismatchError):
        AsyncBasanosEngine(prices=prices, mu=mu, cfg=cfg)


# ─── Synchronous passthrough properties ───────────────────────────────────────


def test_async_engine_assets_property():
    """Assets returns a list of numeric column names."""
    engine = _make_engine()
    assert engine.assets == ["A", "B"]


def test_async_engine_prices_property():
    """Prices property returns a Polars DataFrame."""
    engine = _make_engine()
    assert isinstance(engine.prices, pl.DataFrame)


def test_async_engine_mu_property():
    """Mu property returns a Polars DataFrame."""
    engine = _make_engine()
    assert isinstance(engine.mu, pl.DataFrame)


def test_async_engine_cfg_property():
    """Cfg property returns a BasanosConfig instance."""
    engine = _make_engine()
    assert isinstance(engine.cfg, BasanosConfig)


# ─── Async methods ─────────────────────────────────────────────────────────────


def test_async_engine_ret_adj():
    """ret_adj() returns a DataFrame with the same shape as prices."""
    engine = _make_engine()
    result = asyncio.run(engine.ret_adj())
    assert isinstance(result, pl.DataFrame)
    assert result.shape == engine.prices.shape


def test_async_engine_vola():
    """vola() returns a DataFrame with the same shape as prices."""
    engine = _make_engine()
    result = asyncio.run(engine.vola())
    assert isinstance(result, pl.DataFrame)
    assert result.shape == engine.prices.shape


def test_async_engine_cor():
    """cor() returns a dict mapping each date to a square NumPy array."""
    engine = _make_engine()
    result = asyncio.run(engine.cor())
    assert isinstance(result, dict)
    assert len(result) == engine.prices.height
    for mat in result.values():
        assert isinstance(mat, np.ndarray)
        assert mat.ndim == 2
        assert mat.shape[0] == mat.shape[1] == len(engine.assets)


def test_async_engine_cor_tensor():
    """cor_tensor() returns an (T, N, N) NumPy array."""
    engine = _make_engine()
    result = asyncio.run(engine.cor_tensor())
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    n = len(engine.assets)
    t = engine.prices.height
    assert result.shape == (t, n, n)


def test_async_engine_cash_position():
    """cash_position() returns a DataFrame with the same shape as prices."""
    engine = _make_engine()
    result = asyncio.run(engine.cash_position())
    assert isinstance(result, pl.DataFrame)
    assert result.shape == engine.prices.shape


def test_async_engine_portfolio():
    """portfolio() returns a Portfolio with a positive NAV."""
    from basanos.analytics import Portfolio

    engine = _make_engine()
    result = asyncio.run(engine.portfolio())
    assert isinstance(result, Portfolio)
    nav = result.nav_accumulated
    assert "NAV_accumulated" in nav.columns


def test_async_engine_cor_equals_sync():
    """Async cor() result matches the synchronous BasanosEngine.cor."""
    from basanos.math import BasanosEngine

    engine = _make_engine()
    sync_engine = BasanosEngine(prices=engine.prices, mu=engine.mu, cfg=engine.cfg)
    async_cor = asyncio.run(engine.cor())
    sync_cor = sync_engine.cor
    assert set(async_cor.keys()) == set(sync_cor.keys())
    for key in sync_cor:
        np.testing.assert_array_equal(async_cor[key], sync_cor[key])


def test_async_engine_cash_position_equals_sync():
    """Async cash_position() result matches the synchronous BasanosEngine.cash_position."""
    from basanos.math import BasanosEngine

    engine = _make_engine()
    sync_engine = BasanosEngine(prices=engine.prices, mu=engine.mu, cfg=engine.cfg)
    async_cp = asyncio.run(engine.cash_position())
    sync_cp = sync_engine.cash_position
    assert async_cp.equals(sync_cp)


def test_async_engine_concurrent_calls():
    """Multiple async calls can be gathered concurrently without error."""

    async def run_concurrent() -> tuple:
        engine = _make_engine(n=60)
        vola, ret_adj = await asyncio.gather(engine.vola(), engine.ret_adj())
        return vola, ret_adj

    vola, ret_adj = asyncio.run(run_concurrent())
    assert isinstance(vola, pl.DataFrame)
    assert isinstance(ret_adj, pl.DataFrame)
