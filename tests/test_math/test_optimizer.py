"""Tests for basanos.math.optimizer (BasanosEngine and BasanosConfig)."""

from __future__ import annotations

import logging
import pathlib
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from basanos.exceptions import (
    ColumnMismatchError,
    ExcessiveNullsError,
    MissingDateColumnError,
    MonotonicPricesError,
    NonPositivePricesError,
    ShapeMismatchError,
)
from basanos.math import BasanosConfig, BasanosEngine

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def optimizer_prices() -> pl.DataFrame:
    """120-day, 2-asset price frame built from a seeded random walk."""
    n = 120
    rng = np.random.default_rng(0)
    p_a = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=n))
    p_b = 200.0 + np.cumsum(rng.normal(0.0, 0.7, size=n))
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    return pl.DataFrame({"date": dates, "A": pl.Series(p_a, dtype=pl.Float64), "B": pl.Series(p_b, dtype=pl.Float64)})


@pytest.fixture
def optimizer_mu(optimizer_prices) -> pl.DataFrame:
    """Sinusoidal mu aligned with optimizer_prices."""
    n = optimizer_prices.height
    theta = np.linspace(0.0, 4.0 * np.pi, num=n)
    return pl.DataFrame(
        {
            "date": optimizer_prices["date"],
            "A": pl.Series(np.tanh(np.sin(theta)), dtype=pl.Float64),
            "B": pl.Series(np.tanh(np.cos(theta)), dtype=pl.Float64),
        }
    )


@pytest.fixture
def small_prices() -> pl.DataFrame:
    """10-day deterministic prices frame with two non-monotonic assets."""
    n = 10
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    return pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([100.0, 101.5, 99.8, 103.2, 101.7, 104.5, 102.3, 106.1, 103.9, 107.2], dtype=pl.Float64),
            "B": pl.Series([200.0, 203.1, 198.5, 206.4, 203.7, 209.2, 204.6, 212.3, 207.8, 214.5], dtype=pl.Float64),
        }
    )


@pytest.fixture
def small_mu(small_prices: pl.DataFrame) -> pl.DataFrame:
    """Bounded sinusoidal mu aligned with small_prices."""
    n = small_prices.height
    theta = np.linspace(0.0, np.pi, num=n)
    return pl.DataFrame(
        {
            "date": small_prices["date"],
            "A": pl.Series(np.tanh(np.sin(theta)), dtype=pl.Float64),
            "B": pl.Series(np.tanh(np.cos(theta)), dtype=pl.Float64),
        }
    )


# ─── Year boundary constants (correlation matrix tests) ───────────────────────

_Y1 = date(2020, 1, 1)
_Y2 = date(2021, 1, 1)
_Y3 = date(2022, 1, 1)
_Y4 = date(2023, 1, 1)
_END = date(2023, 12, 31)
_IDX_1, _IDX_2, _IDX_3, _IDX_4 = 0, 1, 2, 3


def _mask(series: np.ndarray, dates: list[date], *, from_: date | None = None, before: date | None = None) -> list:
    """Return a list of floats with None outside the [from_, before) window."""
    return [
        float(v) if (from_ is None or d >= from_) and (before is None or d < before) else None
        for d, v in zip(dates, series, strict=False)
    ]


@pytest.fixture(scope="module")
def prices() -> pl.DataFrame:
    """Four-year, four-asset price frame with staggered availability."""
    dates_series = pl.date_range(start=_Y1, end=_END, interval="1d", eager=True)
    n = len(dates_series)
    rng = np.random.default_rng(42)
    base = {k: 100.0 + np.cumsum(rng.normal(0.0, 0.3, size=n)) for k in range(1, 5)}
    dl = dates_series.to_list()
    return pl.DataFrame(
        {
            "date": dates_series,
            "asset_1": pl.Series(_mask(base[1], dl, before=_Y3), dtype=pl.Float64),
            "asset_2": pl.Series(list(base[2].astype(float)), dtype=pl.Float64),
            "asset_3": pl.Series(_mask(base[3], dl, from_=_Y2), dtype=pl.Float64),
            "asset_4": pl.Series(_mask(base[4], dl, from_=_Y3), dtype=pl.Float64),
        }
    )


@pytest.fixture(scope="module")
def mu(prices: pl.DataFrame) -> pl.DataFrame:
    """Simple bounded expected-return signal aligned with prices."""
    n = prices.height
    theta = np.linspace(0.0, 4.0 * np.pi, num=n)
    return pl.DataFrame(
        {
            "date": prices["date"],
            "asset_1": pl.Series(np.tanh(np.sin(theta)), dtype=pl.Float64),
            "asset_2": pl.Series(np.tanh(np.cos(theta)), dtype=pl.Float64),
            "asset_3": pl.Series(np.tanh(np.sin(2.0 * theta)), dtype=pl.Float64),
            "asset_4": pl.Series(np.tanh(np.cos(2.0 * theta)), dtype=pl.Float64),
        }
    )


@pytest.fixture(scope="module")
def cfg() -> BasanosConfig:
    """Basanos config with moderate EWMA windows suitable for 4-year data."""
    return BasanosConfig(vola=16, corr=32, clip=3.5, shrink=0.5, aum=1e6)


@pytest.fixture(scope="module")
def cor(prices: pl.DataFrame, mu: pl.DataFrame, cfg: BasanosConfig) -> dict:
    """Pre-computed correlation matrices for the full 4-year scenario."""
    return BasanosEngine(prices=prices, mu=mu, cfg=cfg).cor


@pytest.fixture(scope="module")
def tensor_engine(prices: pl.DataFrame, mu: pl.DataFrame, cfg: BasanosConfig) -> BasanosEngine:
    """A single BasanosEngine instance shared across tensor tests."""
    return BasanosEngine(prices=prices, mu=mu, cfg=cfg)


# ─── BasanosConfig validation ─────────────────────────────────────────────────


def test_basanos_config_validator_enforces_corr_ge_vola():
    """BasanosConfig should raise when corr < vola and accept equal/greater."""
    _ = BasanosConfig(vola=20, corr=20, clip=3.0, shrink=0.5, aum=1e6)
    _ = BasanosConfig(vola=12, corr=24, clip=2.0, shrink=0.25, aum=1e6)

    with pytest.raises(ValueError, match=r".*"):
        _ = BasanosConfig(vola=30, corr=10, clip=4.0, shrink=0.7, aum=1e6)


def test_basanos_config_new_fields_have_correct_defaults():
    """New optimizer fields should default to their hardcoded values."""
    cfg = BasanosConfig(vola=16, corr=32, clip=3.0, shrink=0.5, aum=1e6)
    assert cfg.profit_variance_init == 1.0
    assert cfg.profit_variance_decay == 0.99
    assert cfg.denom_tol == 1e-12
    assert cfg.position_scale == 1e6


def test_basanos_config_new_fields_accept_custom_values():
    """New optimizer fields should accept valid custom values."""
    cfg = BasanosConfig(
        vola=16,
        corr=32,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        profit_variance_init=2.0,
        profit_variance_decay=0.95,
        denom_tol=1e-8,
        position_scale=1e4,
    )
    assert cfg.profit_variance_init == 2.0
    assert cfg.profit_variance_decay == 0.95
    assert cfg.denom_tol == 1e-8
    assert cfg.position_scale == 1e4


def test_basanos_config_new_fields_validation():
    """New optimizer fields should reject invalid values."""
    base = {"vola": 16, "corr": 32, "clip": 3.0, "shrink": 0.5, "aum": 1e6}

    with pytest.raises(ValueError, match=r".*"):
        BasanosConfig(**base, profit_variance_init=-1.0)

    with pytest.raises(ValueError, match=r".*"):
        BasanosConfig(**base, profit_variance_decay=0.0)

    with pytest.raises(ValueError, match=r".*"):
        BasanosConfig(**base, profit_variance_decay=1.0)

    with pytest.raises(ValueError, match=r".*"):
        BasanosConfig(**base, denom_tol=0.0)

    with pytest.raises(ValueError, match=r".*"):
        BasanosConfig(**base, position_scale=-1.0)


# ─── BasanosEngine construction validation ────────────────────────────────────


def test_post_init_shape_mismatch_raises(small_prices: pl.DataFrame, small_mu: pl.DataFrame) -> None:
    """Prices and mu must have identical shapes (rows and columns)."""
    cfg = BasanosConfig(vola=4, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    with pytest.raises(ShapeMismatchError) as exc_info:
        _ = BasanosEngine(prices=small_prices, mu=small_mu.slice(0, small_mu.height - 1), cfg=cfg)
    assert exc_info.value.prices_shape == small_prices.shape
    assert exc_info.value.mu_shape == small_mu.slice(0, small_mu.height - 1).shape


def test_post_init_missing_date_raises() -> None:
    """Both prices and mu must contain a 'date' column."""
    cfg = BasanosConfig(vola=4, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    prices_no_date = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    mu_ok = pl.DataFrame(
        {
            "date": pl.date_range(date(2020, 1, 1), date(2020, 1, 2), interval="1d", eager=True),
            "A": [0.1, 0.2],
            "B": [0.3, 0.4],
        }
    )
    with pytest.raises(MissingDateColumnError) as exc_info:
        _ = BasanosEngine(prices=prices_no_date, mu=mu_ok, cfg=cfg)
    assert exc_info.value.frame_name == "prices"

    prices_ok = pl.DataFrame(
        {
            "date": pl.date_range(date(2020, 1, 1), date(2020, 1, 2), interval="1d", eager=True),
            "A": [1.0, 2.0],
            "B": [3.0, 4.0],
        }
    )
    with pytest.raises(MissingDateColumnError) as exc_info:
        _ = BasanosEngine(prices=prices_ok, mu=pl.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]}), cfg=cfg)
    assert exc_info.value.frame_name == "mu"


def test_post_init_column_mismatch_raises(small_prices: pl.DataFrame, small_mu: pl.DataFrame) -> None:
    """Prices and mu must have identical column sets."""
    cfg = BasanosConfig(vola=4, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    with pytest.raises(ColumnMismatchError) as exc_info:
        _ = BasanosEngine(prices=small_prices, mu=small_mu.rename({"B": "BB"}), cfg=cfg)
    assert "B" in exc_info.value.prices_columns
    assert "BB" in exc_info.value.mu_columns


def test_post_init_nonpositive_prices_raises(small_mu: pl.DataFrame) -> None:
    """Prices must be strictly positive; zero or negative values should raise."""
    cfg = BasanosConfig(vola=4, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    dates = small_mu["date"]

    # price of zero
    a_zero = pl.Series([100.0, 101.5, 0.0, 103.2, 101.7, 104.5, 102.3, 106.1, 103.9, 107.2], dtype=pl.Float64)
    b_ok = pl.Series([200.0, 203.1, 198.5, 206.4, 203.7, 209.2, 204.6, 212.3, 207.8, 214.5], dtype=pl.Float64)
    with pytest.raises(NonPositivePricesError) as exc_info:
        _ = BasanosEngine(
            prices=pl.DataFrame({"date": dates, "A": a_zero, "B": b_ok}),
            mu=small_mu,
            cfg=cfg,
        )
    assert exc_info.value.asset == "A"

    # negative price
    a_neg = pl.Series([100.0, 101.5, -5.0, 103.2, 101.7, 104.5, 102.3, 106.1, 103.9, 107.2], dtype=pl.Float64)
    with pytest.raises(NonPositivePricesError) as exc_info:
        _ = BasanosEngine(
            prices=pl.DataFrame({"date": dates, "A": a_neg, "B": b_ok}),
            mu=small_mu,
            cfg=cfg,
        )
    assert exc_info.value.asset == "A"

    # null values should be ignored; only non-null values are checked
    a_with_null = pl.Series([100.0, None, 99.8, 103.2, 101.7, 104.5, 102.3, 106.1, 103.9, 107.2], dtype=pl.Float64)
    _ = BasanosEngine(
        prices=pl.DataFrame({"date": dates, "A": a_with_null, "B": b_ok}),
        mu=small_mu,
        cfg=cfg,
    )


def test_post_init_excessive_nan_raises(small_mu: pl.DataFrame) -> None:
    """Any asset column with more than 90% null values should raise."""
    cfg = BasanosConfig(vola=4, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    n = small_mu.height
    dates = small_mu["date"]

    # 9 out of 10 values are null (90% nulls → exactly at boundary, not over)
    a_ninety = pl.Series([100.0] + [None] * (n - 1), dtype=pl.Float64)
    b_ok = pl.Series([200.0, 203.1, 198.5, 206.4, 203.7, 209.2, 204.6, 212.3, 207.8, 214.5], dtype=pl.Float64)
    # exactly 90% null: should NOT raise (threshold is strictly greater than)
    _ = BasanosEngine(
        prices=pl.DataFrame({"date": dates, "A": a_ninety, "B": b_ok}),
        mu=small_mu,
        cfg=cfg,
    )

    # 10 out of 10 values are null (100% nulls → over threshold)
    a_all_null = pl.Series([None] * n, dtype=pl.Float64)
    with pytest.raises(ExcessiveNullsError) as exc_info:
        _ = BasanosEngine(
            prices=pl.DataFrame({"date": dates, "A": a_all_null, "B": b_ok}),
            mu=small_mu,
            cfg=cfg,
        )
    assert exc_info.value.asset == "A"
    assert exc_info.value.null_fraction == pytest.approx(1.0)
    assert exc_info.value.max_fraction == pytest.approx(0.9)


def test_post_init_monotonic_prices_raises(small_mu: pl.DataFrame) -> None:
    """Strictly monotonic price series should raise; non-monotonic should pass."""
    cfg = BasanosConfig(vola=4, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    n = small_mu.height
    dates = small_mu["date"]
    b_ok = pl.Series([200.0, 203.1, 198.5, 206.4, 203.7, 209.2, 204.6, 212.3, 207.8, 214.5], dtype=pl.Float64)

    # strictly increasing prices
    a_increasing = pl.Series([100.0 + i for i in range(n)], dtype=pl.Float64)
    with pytest.raises(MonotonicPricesError) as exc_info:
        _ = BasanosEngine(
            prices=pl.DataFrame({"date": dates, "A": a_increasing, "B": b_ok}),
            mu=small_mu,
            cfg=cfg,
        )
    assert exc_info.value.asset == "A"

    # strictly decreasing prices
    a_decreasing = pl.Series([100.0 - i for i in range(n)], dtype=pl.Float64)
    with pytest.raises(MonotonicPricesError) as exc_info:
        _ = BasanosEngine(
            prices=pl.DataFrame({"date": dates, "A": a_decreasing, "B": b_ok}),
            mu=small_mu,
            cfg=cfg,
        )
    assert exc_info.value.asset == "A"

    # constant prices
    a_constant = pl.Series([100.0] * n, dtype=pl.Float64)
    with pytest.raises(MonotonicPricesError) as exc_info:
        _ = BasanosEngine(
            prices=pl.DataFrame({"date": dates, "A": a_constant, "B": b_ok}),
            mu=small_mu,
            cfg=cfg,
        )
    assert exc_info.value.asset == "A"

    # non-monotonic prices should pass
    a_ok = pl.Series([100.0, 101.5, 99.8, 103.2, 101.7, 104.5, 102.3, 106.1, 103.9, 107.2], dtype=pl.Float64)
    _ = BasanosEngine(
        prices=pl.DataFrame({"date": dates, "A": a_ok, "B": b_ok}),
        mu=small_mu,
        cfg=cfg,
    )


# ─── cash_position: basic schema and finite values ────────────────────────────────────────────


def test_optimize_returns_frame_with_expected_schema_and_finite_after_warmup(optimizer_prices, optimizer_mu):
    """cash_position should have date + asset cols and finite values after warmup."""
    cfg = BasanosConfig(vola=16, corr=20, clip=3.5, shrink=0.5, aum=1e6)
    cp = BasanosEngine(prices=optimizer_prices, cfg=cfg, mu=optimizer_mu).cash_position

    assert isinstance(cp, pl.DataFrame)
    assert cp.height == optimizer_prices.height
    assert cp.columns[0] == "date"
    assert set(cp.columns[1:]) == {"A", "B"}

    tail = cp.tail(optimizer_prices.height - cfg.corr)
    for c in ("A", "B"):
        assert tail[c].null_count() == 0
        assert tail[c].is_finite().all()


def test_optimize_with_zero_mu_returns_zero_positions(optimizer_prices):
    """When mu is all zeros, the optimizer should return zero risk positions."""
    cfg = BasanosConfig(corr=20, vola=12, clip=4.0, shrink=0.7, aum=1e6)
    mu_zero = pl.DataFrame(
        {
            "date": optimizer_prices["date"],
            "A": pl.Series([0.0] * optimizer_prices.height, dtype=pl.Float64),
            "B": pl.Series([0.0] * optimizer_prices.height, dtype=pl.Float64),
        }
    )
    cp = BasanosEngine(prices=optimizer_prices, cfg=cfg, mu=mu_zero).cash_position

    tail = cp.tail(80 - cfg.corr)
    for c in ("A", "B"):
        assert np.allclose(tail[c].to_numpy(), 0.0, rtol=0, atol=0)


# ─── cash_position: edge cases ────────────────────────────────────────────────


def test_cash_position_skips_profit_update_when_returns_all_nan():
    """Branch where ret_mask.any() is False should be exercised without error."""
    dates = pl.date_range(start=pl.date(2020, 1, 1), end=pl.date(2020, 1, 5), interval="1d", eager=True)
    prices = pl.DataFrame(
        {"date": dates, "A": [None, 101.0, 103.5, 101.8, 104.2], "B": [None, 201.0, 203.5, 200.8, 204.2]}
    ).with_columns(pl.col(["A", "B"]).cast(pl.Float64))
    mu = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([0.1, 0.2, 0.1, -0.1, 0.0], dtype=pl.Float64),
            "B": pl.Series([0.0, -0.2, -0.1, 0.2, 0.1], dtype=pl.Float64),
        }
    )
    cfg = BasanosConfig(vola=1, corr=2, clip=3.0, shrink=0.5, aum=1e6)
    cp = BasanosEngine(prices=prices, mu=mu, cfg=cfg).cash_position

    assert cp.columns[0] == "date"
    assert cp.columns[1:] == ["A", "B"]
    assert isinstance(cp, pl.DataFrame)
    tail = cp.tail(cp.height - cfg.corr)
    for c in ("A", "B"):
        assert tail.schema[c] == pl.Float64


def test_cash_position_skips_rows_with_all_nan_prices() -> None:
    """cash_position should skip timestamps where all asset prices are non-finite."""
    n = 12
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    rng = np.random.default_rng(7)
    a = list(100.0 + np.cumsum(rng.normal(0.0, 0.5, size=n)))
    b = list(200.0 + np.cumsum(rng.normal(0.0, 0.7, size=n)))
    idx_nan = 6
    a[idx_nan] = None  # type: ignore[assignment]
    b[idx_nan] = None  # type: ignore[assignment]
    prices = pl.DataFrame({"date": dates, "A": a, "B": b}).with_columns(
        pl.col("A").cast(pl.Float64), pl.col("B").cast(pl.Float64)
    )
    theta = np.linspace(0.0, np.pi, num=n)
    mu = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.tanh(np.sin(theta)), dtype=pl.Float64),
            "B": pl.Series(np.tanh(np.cos(theta)), dtype=pl.Float64),
        }
    )
    cfg = BasanosConfig(vola=3, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    cp = BasanosEngine(prices=prices, mu=mu, cfg=cfg).cash_position

    row = cp.row(idx_nan, named=True)
    assert row["A"] is None or (isinstance(row["A"], float) and np.isnan(row["A"]))
    assert row["B"] is None or (isinstance(row["B"], float) and np.isnan(row["B"]))


def test_cash_position_warns_when_denom_is_near_zero(optimizer_prices: pl.DataFrame, caplog) -> None:
    """A warning should be emitted when the normalisation denominator is degenerate.

    A large denom_tol forces all non-zero-mu timestamps to trigger the guard, so
    at least one warning is logged for the non-trivial (non-zero mu) case.
    """
    n = optimizer_prices.height
    theta = np.linspace(0.0, 4.0 * np.pi, num=n)
    mu = pl.DataFrame(
        {
            "date": optimizer_prices["date"],
            "A": pl.Series(np.tanh(np.sin(theta)), dtype=pl.Float64),
            "B": pl.Series(np.tanh(np.cos(theta)), dtype=pl.Float64),
        }
    )
    # denom_tol set absurdly large so every timestamp triggers the guard
    cfg = BasanosConfig(corr=20, vola=12, clip=4.0, shrink=0.7, aum=1e6, denom_tol=1e6)

    with caplog.at_level(logging.WARNING, logger="basanos.math.optimizer"):
        _ = BasanosEngine(prices=optimizer_prices, cfg=cfg, mu=mu).cash_position

    warnings = [r for r in caplog.records if "normalisation denominator is degenerate" in r.message]
    assert warnings, "Expected at least one warning about a degenerate normalisation denominator"
    assert all("denom=" in r.message for r in warnings)
    assert all("denom_tol=" in r.message for r in warnings)


# ─── ret_adj, vola, cor properties ────────────────────────────────────────────


def test_ret_adj_and_vola_return_frames_with_asset_columns(small_prices: pl.DataFrame, small_mu: pl.DataFrame) -> None:
    """ret_adj and vola should return DataFrames aligned with asset columns."""
    cfg = BasanosConfig(vola=3, corr=4, clip=2.5, shrink=0.5, aum=1e6)
    engine = BasanosEngine(prices=small_prices, mu=small_mu, cfg=cfg)

    for prop in (engine.ret_adj, engine.vola):
        assert isinstance(prop, pl.DataFrame)
        assert prop.columns[0] == "date"
        assert set(prop.columns[1:]) == {"A", "B"}


def test_cor_returns_square_matrices(small_prices: pl.DataFrame, small_mu: pl.DataFrame) -> None:
    """Cor should return a dict of square correlation matrices of size n_assets."""
    cfg = BasanosConfig(vola=2, corr=2, clip=2.5, shrink=0.5, aum=1e6)
    engine = BasanosEngine(prices=small_prices, mu=small_mu, cfg=cfg)
    cor = engine.cor

    assert isinstance(cor, dict)
    assert len(cor) > 0
    n_assets = len(engine.assets)
    for _, mat in cor.items():
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (n_assets, n_assets)


# ─── portfolio property ───────────────────────────────────────────────────────


def _make_prices_mu(n: int = 64) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create synthetic prices and a bounded mu aligned by date."""
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)
    a = pl.Series([100.0 + 2.0 * np.cos(0.20 * i) for i in range(n)], dtype=pl.Float64)
    b = pl.Series([200.0 + 5.0 * np.sin(0.15 * i) for i in range(n)], dtype=pl.Float64)
    prices = pl.DataFrame({"date": dates, "A": a, "B": b})
    theta = np.linspace(0.0, 3.0 * np.pi, num=n)
    mu = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.tanh(np.sin(theta)), dtype=pl.Float64),
            "B": pl.Series(np.tanh(np.cos(theta)), dtype=pl.Float64),
        }
    )
    return prices, mu


def test_basanos_portfolio_builds_portfolio_with_finite_nav_and_positions():
    """BasanosEngine.portfolio should return a Portfolio with sane outputs."""
    prices, mu = _make_prices_mu(96)
    cfg = BasanosConfig(vola=16, corr=24, clip=3.5, shrink=0.5, aum=1e6)
    engine = BasanosEngine(prices=prices, cfg=cfg, mu=mu)

    assert engine.cfg == cfg
    assert engine.assets == ["A", "B"]

    portfolio = engine.portfolio
    rp = portfolio.cashposition

    assert isinstance(rp, pl.DataFrame)
    assert rp.columns[0] == "date"
    assert set(rp.columns[1:]) == {"A", "B"}
    tail = rp.tail(rp.height - cfg.corr)
    for c in ("A", "B"):
        assert tail[c].null_count() == 0
        assert tail[c].is_finite().all()

    nav = portfolio.nav_accumulated
    assert nav.height == prices.height
    assert nav["NAV_accumulated"].is_finite().all()


# ─── Correlation matrix shape ─────────────────────────────────────────────────


class TestCorMatrixShape:
    """Structural properties of the returned correlation dict."""

    def test_returns_dict(self, cor: dict) -> None:
        """Cor should return a plain Python dict."""
        assert isinstance(cor, dict)

    def test_has_one_entry_per_day(self, prices: pl.DataFrame, cor: dict) -> None:
        """The dict should have exactly one entry for each row in the prices frame."""
        assert len(cor) == prices.height

    def test_each_matrix_is_4x4_numpy_array(self, cor: dict) -> None:
        """Every value in the dict should be a (4, 4) NumPy ndarray."""
        for _d, mat in cor.items():
            assert isinstance(mat, np.ndarray)
            assert mat.shape == (4, 4), f"Expected (4, 4), got {mat.shape} at {_d}"

    def test_finite_correlations_are_in_unit_interval(self, cor: dict) -> None:
        """Finite correlation values must lie within [−1, 1]."""
        for _d, mat in cor.items():
            finite = mat[np.isfinite(mat)]
            if finite.size:
                assert np.all(finite >= -1.0 - 1e-9), f"Correlation below -1 at {_d}"
                assert np.all(finite <= 1.0 + 1e-9), f"Correlation above +1 at {_d}"

    def test_matrices_are_symmetric(self, cor: dict) -> None:
        """Sample every 30th date to keep the test fast."""
        for d in list(cor.keys())[::30]:
            mat = cor[d]
            both_finite = np.isfinite(mat) & np.isfinite(mat.T)
            if both_finite.any():
                np.testing.assert_allclose(
                    mat[both_finite], mat.T[both_finite], atol=1e-10, err_msg=f"Matrix not symmetric at {d}"
                )

    def test_diagonal_is_one_where_finite(self, cor: dict, cfg: BasanosConfig) -> None:
        """After a generous warmup period, every finite diagonal entry equals 1."""
        warmup = cfg.corr + cfg.vola
        for d in list(cor.keys())[warmup::30]:
            mat = cor[d]
            for i, val in enumerate(np.diag(mat)):
                if np.isfinite(val):
                    assert abs(val - 1.0) < 1e-6, f"Diagonal [{i},{i}] = {val} at {d}; expected 1.0"


# ─── Correlation matrix asset availability ────────────────────────────────────


class TestCorMatrixAssetAvailability:
    """The correlation matrices should reflect which assets have price data."""

    def test_asset3_and_asset4_rows_are_nan_in_year1(self, cor: dict, cfg: BasanosConfig) -> None:
        """asset_3 and asset_4 have no price data in year 1; rows/cols must be non-finite."""
        year1_post_warmup = [d for d in cor if d.year == 2020][cfg.corr :]
        for d in year1_post_warmup[:10]:
            mat = cor[d]
            assert not np.any(np.isfinite(mat[_IDX_3, :])), f"asset_3 row not all-NaN at {d}"
            assert not np.any(np.isfinite(mat[:, _IDX_3])), f"asset_3 col not all-NaN at {d}"
            assert not np.any(np.isfinite(mat[_IDX_4, :])), f"asset_4 row not all-NaN at {d}"
            assert not np.any(np.isfinite(mat[:, _IDX_4])), f"asset_4 col not all-NaN at {d}"

    def test_asset1_asset2_cross_correlation_finite_in_late_year1(self, cor: dict, cfg: BasanosConfig) -> None:
        """Both asset_1 and asset_2 are present in year 1; cross-correlation should be finite after warmup."""
        warmup = cfg.corr + cfg.vola + 10
        late_year1 = sorted(d for d in cor if d.year == 2020)[warmup:]
        assert late_year1, "Need year-1 dates beyond warmup"
        for d in late_year1[-5:]:
            mat = cor[d]
            assert np.isfinite(mat[_IDX_1, _IDX_2]), f"asset_1 vs asset_2 not finite at {d}"
            assert np.isfinite(mat[_IDX_2, _IDX_1]), f"asset_2 vs asset_1 not finite at {d}"

    def test_asset3_correlations_become_finite_in_late_year2(self, cor: dict, cfg: BasanosConfig) -> None:
        """asset_3 enters year 2; after warmup its correlation with asset_2 should be finite."""
        warmup_after_entry = cfg.corr + 10
        late_year2 = sorted(d for d in cor if d.year == 2021)[warmup_after_entry:]
        assert late_year2, "Need year-2 dates beyond warmup"
        for d in late_year2[-5:]:
            mat = cor[d]
            assert np.isfinite(mat[_IDX_2, _IDX_3]), f"asset_2 vs asset_3 not finite at {d}"
            assert np.isfinite(mat[_IDX_3, _IDX_2]), f"asset_3 vs asset_2 not finite at {d}"

    def test_asset4_correlations_become_finite_in_late_year3(self, cor: dict, cfg: BasanosConfig) -> None:
        """asset_4 enters year 3; after warmup its correlations should be finite."""
        warmup_after_entry = cfg.corr + 10
        late_year3 = sorted(d for d in cor if d.year == 2022)[warmup_after_entry:]
        assert late_year3, "Need year-3 dates beyond warmup"
        for d in late_year3[-5:]:
            mat = cor[d]
            assert np.isfinite(mat[_IDX_2, _IDX_4]), f"asset_2 vs asset_4 not finite at {d}"
            assert np.isfinite(mat[_IDX_4, _IDX_2]), f"asset_4 vs asset_2 not finite at {d}"
            assert np.isfinite(mat[_IDX_3, _IDX_4]), f"asset_3 vs asset_4 not finite at {d}"
            assert np.isfinite(mat[_IDX_4, _IDX_3]), f"asset_4 vs asset_3 not finite at {d}"

    def test_active_assets_have_finite_diagonal_in_late_year3(self, cor: dict, cfg: BasanosConfig) -> None:
        """In year 3, asset_2, asset_3, and asset_4 all have live price data; diagonals must be 1."""
        warmup_after_entry = cfg.corr + cfg.vola + 10
        late_year3 = sorted(d for d in cor if d.year == 2022)[warmup_after_entry:]
        if not late_year3:
            pytest.skip("Not enough year-3 dates after warmup")
        for d in late_year3[-5:]:
            mat = cor[d]
            for idx in (_IDX_2, _IDX_3, _IDX_4):
                diag = mat[idx, idx]
                assert np.isfinite(diag), f"Diagonal [{idx},{idx}] is NaN at {d}"
                assert abs(diag - 1.0) < 1e-6, f"Diagonal [{idx},{idx}] = {diag} at {d}"


# ─── Cash position with staggered assets ─────────────────────────────────────


class TestCashPositionWithStaggeredAssets:
    """BasanosEngine.cash_position must be NaN exactly where prices are absent."""

    @pytest.fixture(scope="class")
    def cash_pos(self, prices: pl.DataFrame, mu: pl.DataFrame, cfg: BasanosConfig) -> pl.DataFrame:
        """Build the cash_position DataFrame once per test class."""
        return BasanosEngine(prices=prices, mu=mu, cfg=cfg).cash_position

    def test_schema_matches_prices(self, cash_pos: pl.DataFrame, prices: pl.DataFrame) -> None:
        """cash_position should have the same row count and columns as prices."""
        assert isinstance(cash_pos, pl.DataFrame)
        assert cash_pos.height == prices.height
        assert cash_pos.columns[0] == "date"
        assert set(cash_pos.columns[1:]) == {"asset_1", "asset_2", "asset_3", "asset_4"}

    def test_asset1_positions_nan_from_year3(self, cash_pos: pl.DataFrame) -> None:
        """asset_1 price is null from year 3 onward; positions must be NaN."""
        dates = cash_pos["date"].to_list()
        vals = cash_pos["asset_1"].to_numpy()
        for i in [i for i, d in enumerate(dates) if d >= _Y3][:20]:
            assert not np.isfinite(vals[i]), f"Expected NaN for asset_1 at {dates[i]}, got {vals[i]}"

    def test_asset3_positions_nan_in_year1(self, cash_pos: pl.DataFrame) -> None:
        """asset_3 price is null in year 1; positions must be NaN."""
        dates = cash_pos["date"].to_list()
        vals = cash_pos["asset_3"].to_numpy()
        for i in [i for i, d in enumerate(dates) if d < _Y2]:
            assert not np.isfinite(vals[i]), f"Expected NaN for asset_3 at {dates[i]}, got {vals[i]}"

    def test_asset4_positions_nan_in_years_1_and_2(self, cash_pos: pl.DataFrame) -> None:
        """asset_4 price is null in years 1-2; positions must be NaN."""
        dates = cash_pos["date"].to_list()
        vals = cash_pos["asset_4"].to_numpy()
        for i in [i for i, d in enumerate(dates) if d < _Y3]:
            assert not np.isfinite(vals[i]), f"Expected NaN for asset_4 at {dates[i]}, got {vals[i]}"

    def test_asset2_positions_finite_after_warmup(self, cash_pos: pl.DataFrame, cfg: BasanosConfig) -> None:
        """asset_2 is present for all 4 years; after warmup its positions are finite."""
        warmup = cfg.corr + cfg.vola
        assert np.all(np.isfinite(cash_pos["asset_2"].to_numpy()[warmup:])), (
            "asset_2 has non-finite positions after warmup"
        )


# ─── cor_tensor flat-file round-trip ─────────────────────────────────────────


class TestCorTensorFlatFile:
    """Store all correlation matrices in a tensor and round-trip via a flat file."""

    def test_tensor_shape_and_type(self, tensor_engine: BasanosEngine, prices: pl.DataFrame) -> None:
        """cor_tensor should return a 3-D NumPy ndarray of shape (T, N, N)."""
        tensor = tensor_engine.cor_tensor
        n_assets = len(tensor_engine.assets)
        assert isinstance(tensor, np.ndarray)
        assert tensor.ndim == 3
        assert tensor.shape == (prices.height, n_assets, n_assets)

    def test_tensor_slices_match_cor_dict(self, tensor_engine: BasanosEngine) -> None:
        """Each slice tensor[t] must equal the corresponding matrix in cor."""
        tensor = tensor_engine.cor_tensor
        for t, mat in enumerate(tensor_engine.cor.values()):
            np.testing.assert_array_equal(
                tensor[t], mat, err_msg=f"tensor[{t}] does not match cor dict entry at index {t}"
            )

    def test_tensor_saves_and_loads_from_flat_file(
        self, tensor_engine: BasanosEngine, resource_dir: pathlib.Path
    ) -> None:
        """NumPy EWM implementation must reproduce the reference tensor within 1e-10."""
        tensor = tensor_engine.cor_tensor
        loaded = np.load(resource_dir / "cor_tensor.npy")
        assert loaded.shape == tensor.shape
        np.testing.assert_allclose(loaded, tensor, rtol=0, atol=1e-10)
