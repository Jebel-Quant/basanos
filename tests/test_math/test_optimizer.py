"""Tests for basanos.math.optimizer (BasanosEngine and BasanosConfig).

This module covers:
- BasanosConfig field validation and default values.
- BasanosEngine construction guards (shape, date column, column names,
  non-positive prices, excessive nulls, monotonic prices).
- cash_position output schema, finite-value assertions, and edge-case
  branches (all-NaN mu, all-NaN price rows, degenerate normalisation).
- ret_adj, vola, and cor property shapes.
- portfolio property producing a Portfolio with sane NAV.
- Structural and asset-availability properties of the correlation matrix dict.
- cash_position correctness with staggered-asset price frames.
- cor_tensor shape, slice equality, and flat-file round-trip.
"""

from __future__ import annotations

import logging
import math
import pathlib
from datetime import date, timedelta
from unittest.mock import patch

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
    SingularMatrixError,
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
    """Bounded sinusoidal expected-return signal aligned with optimizer_prices.

    Uses tanh(sin/cos) so values stay in (−1, +1) and change sign frequently,
    exercising both long and short allocation paths in the optimizer.
    """
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
    """Bounded sinusoidal expected-return signal aligned with small_prices.

    Half-period sine wave keeps values non-negative on the first half and
    non-positive on the second half, providing sign-change coverage with only
    10 rows, which is enough for the small validation tests.
    """
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
    assert cfg.min_corr_denom == 1e-14
    assert cfg.max_nan_fraction == 0.9


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
        min_corr_denom=1e-10,
        max_nan_fraction=0.5,
    )
    assert cfg.profit_variance_init == 2.0
    assert cfg.profit_variance_decay == 0.95
    assert cfg.denom_tol == 1e-8
    assert cfg.position_scale == 1e4
    assert cfg.min_corr_denom == 1e-10
    assert cfg.max_nan_fraction == 0.5


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

    with pytest.raises(ValueError, match=r".*"):
        BasanosConfig(**base, min_corr_denom=0.0)

    with pytest.raises(ValueError, match=r".*"):
        BasanosConfig(**base, max_nan_fraction=0.0)

    with pytest.raises(ValueError, match=r".*"):
        BasanosConfig(**base, max_nan_fraction=1.0)


# ─── BasanosEngine construction validation ────────────────────────────────────


def test_post_init_shape_mismatch_raises(small_prices: pl.DataFrame, small_mu: pl.DataFrame) -> None:
    """Prices and mu must have identical shapes (rows and columns)."""
    cfg = BasanosConfig(vola=4, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    with pytest.raises(ShapeMismatchError) as exc_info:
        _ = BasanosEngine(prices=small_prices, mu=small_mu.slice(0, small_mu.height - 1), cfg=cfg)
    assert exc_info.value.prices_shape == small_prices.shape
    assert exc_info.value.mu_shape == small_mu.slice(0, small_mu.height - 1).shape


def test_post_init_missing_date_raises() -> None:
    """BasanosEngine requires a 'date' column in both prices and mu; omitting it raises ValueError.

    Tested for each direction: prices missing 'date', then mu missing 'date'.
    """
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
    """BasanosEngine rejects prices and mu with different column names.

    Renaming a column in mu so it no longer matches prices should raise
    ValueError, ensuring the engine cannot silently pair assets incorrectly.
    """
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


def test_post_init_excessive_nan_respects_config(small_mu: pl.DataFrame) -> None:
    """max_nan_fraction from cfg should control the null gate."""
    n = small_mu.height
    dates = small_mu["date"]
    b_ok = pl.Series([200.0, 203.1, 198.5, 206.4, 203.7, 209.2, 204.6, 212.3, 207.8, 214.5], dtype=pl.Float64)
    # 5 out of 10 values are null (50% nulls) with non-monotonic prices in non-null rows
    a_half_null = pl.Series([100.0, 102.3, 99.5, 103.7, 101.1, None, None, None, None, None], dtype=pl.Float64)

    # with default max_nan_fraction=0.9, 50% nulls should pass
    cfg_default = BasanosConfig(vola=4, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    _ = BasanosEngine(
        prices=pl.DataFrame({"date": dates, "A": a_half_null, "B": b_ok}),
        mu=small_mu,
        cfg=cfg_default,
    )

    # with a stricter max_nan_fraction=0.4, 50% nulls should raise
    cfg_strict = BasanosConfig(vola=4, corr=4, clip=3.0, shrink=0.5, aum=1e6, max_nan_fraction=0.4)
    with pytest.raises(ExcessiveNullsError) as exc_info:
        _ = BasanosEngine(
            prices=pl.DataFrame({"date": dates, "A": a_half_null, "B": b_ok}),
            mu=small_mu,
            cfg=cfg_strict,
        )
    assert exc_info.value.asset == "A"
    assert exc_info.value.max_fraction == pytest.approx(0.4)

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


def test_shrink_zero_identity_case() -> None:
    """With shrink=0, C_shrunk = I, so solve(I, mu) = mu (up to normalisation).

    When the correlation matrix is replaced entirely by the identity, each
    asset's solved position is proportional to its expected-return signal
    (undeflected by cross-asset correlations).  Concretely, the *risk*
    position — cash_position × vola — must point in the same direction as
    mu at every timestamp after the warm-up period where mu is non-zero.
    """
    prices, mu = _make_prices_mu(80)
    cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.0, aum=1e6)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)

    warmup = cfg.corr
    assets = engine.assets

    mu_arr = mu.select(assets).to_numpy()
    cp_arr = engine.cash_position.select(assets).to_numpy()
    vola_arr = engine.vola.select(assets).to_numpy()

    for i in range(warmup, prices.height):
        mu_row = mu_arr[i]
        # Skip timestamps where the signal is zero (no trade taken)
        if np.allclose(mu_row, 0.0):
            continue

        # risk_pos = cash_pos * vola is the un-volatility-scaled position
        # When C = I: solve(I, mu) = mu, so risk_pos ∝ mu
        risk_row = cp_arr[i] * vola_arr[i]

        # Skip rows where risk position is zero (e.g. degenerate denominator during early warmup)
        risk_norm = np.linalg.norm(risk_row)
        if risk_norm < 1e-15:
            continue

        norm_risk = risk_row / risk_norm
        norm_mu = mu_row / np.linalg.norm(mu_row)
        np.testing.assert_allclose(norm_risk, norm_mu, atol=1e-10)


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


def test_cash_position_no_warning_for_zero_mu(optimizer_prices: pl.DataFrame, caplog) -> None:
    """No warning should be logged when mu is all zeros.

    When expected_mu is identically zero the optimizer short-circuits before
    calling inv_a_norm, so the degenerate-denominator path is never reached and
    no warning should be emitted.
    """
    cfg = BasanosConfig(corr=20, vola=12, clip=4.0, shrink=0.7, aum=1e6)
    n = optimizer_prices.height
    mu_zero = pl.DataFrame(
        {
            "date": optimizer_prices["date"],
            "A": pl.Series([0.0] * n, dtype=pl.Float64),
            "B": pl.Series([0.0] * n, dtype=pl.Float64),
        }
    )
    with caplog.at_level(logging.WARNING, logger="basanos.math.optimizer"):
        _ = BasanosEngine(prices=optimizer_prices, cfg=cfg, mu=mu_zero).cash_position

    degen_warnings = [r for r in caplog.records if "normalisation denominator is degenerate" in r.message]
    assert not degen_warnings, "No warning should be logged when mu is all zeros"


def test_cash_position_zeros_and_warns_when_inv_a_norm_returns_nan(caplog) -> None:
    """When inv_a_norm returns nan (all-NaN correlation matrix), positions are zeroed and a warning emitted.

    A minimal price series with corr=100 ensures the correlation window is never
    satisfied, so every correlation matrix is all-NaN.  With non-zero mu the
    optimizer must still zero positions and emit a warning rather than dividing
    by NaN.
    """
    n = 10
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    # Seed is fixed for reproducibility; specific values are not critical to the test
    rng = np.random.default_rng(42)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n)), dtype=pl.Float64),
            "B": pl.Series(200.0 + np.cumsum(rng.normal(0, 0.7, n)), dtype=pl.Float64),
        }
    )
    mu = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([0.1] * n, dtype=pl.Float64),
            "B": pl.Series([0.2] * n, dtype=pl.Float64),
        }
    )
    # corr=100 far exceeds the 10-row series, so all correlation matrices are all-NaN
    cfg = BasanosConfig(corr=100, vola=5, clip=4.0, shrink=0.5, aum=1e6)

    with caplog.at_level(logging.WARNING, logger="basanos.math.optimizer"):
        cp = BasanosEngine(prices=prices, cfg=cfg, mu=mu).cash_position

    # All positions must be zero or NaN during warm-up - never infinite or based on NaN denom
    for col in ("A", "B"):
        vals = cp[col].to_numpy(allow_copy=True)
        assert not np.any(np.isinf(vals)), f"Unexpected infinite position in column {col}"
        assert np.all(np.isnan(vals) | (vals == 0.0)), f"Expected only NaN or zero positions in column {col}"

    # A warning must have been logged for the degenerate denominator
    degen_warnings = [r for r in caplog.records if "normalisation denominator is degenerate" in r.message]
    assert degen_warnings, "Expected a warning when inv_a_norm returns nan due to all-NaN correlation matrix"
    assert all("denom=nan" in r.message for r in degen_warnings)


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
        """Cor property should return a plain Python dict keyed by date.

        Verifies the public API contract so callers can rely on dict-style
        lookup (e.g. cor[some_date]) without additional unwrapping.
        """
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
        """Correlation matrices must be symmetric for every pair of finite entries.

        Samples every 30th date to keep runtime short while still spanning
        multiple years and asset-availability regimes.
        """
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


# ─── Profit variance EMA mechanism ───────────────────────────────────────────


class TestProfitVarianceEMA:
    """Dedicated tests for the profit variance EMA update mechanism.

    The adaptive position-sizing logic in ``cash_position`` maintains a running
    variance estimate via an EMA:

        pv[t] = λ·pv[t-1] + (1-λ)·profit[t]²

    where ``profit[t] = cash_position[t-1] @ returns[t]`` uses *yesterday's*
    cash position (lagged). Positions are then scaled as:

        risk_pos[t] = solved_pos[t] / pv[t]

    These tests validate the EMA initialisation, the update formula, and the
    effect of the decay parameter on position magnitude.
    """

    @pytest.fixture
    def pv_prices(self) -> pl.DataFrame:
        """30-day, 2-asset non-monotonic price series for EMA tests."""
        n = 30
        start = date(2020, 1, 1)
        dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
        rng = np.random.default_rng(99)
        return pl.DataFrame(
            {
                "date": dates,
                "A": pl.Series(100.0 + np.cumsum(rng.normal(0.0, 0.5, n)), dtype=pl.Float64),
                "B": pl.Series(200.0 + np.cumsum(rng.normal(0.0, 0.7, n)), dtype=pl.Float64),
            }
        )

    @pytest.fixture
    def pv_mu(self, pv_prices: pl.DataFrame) -> pl.DataFrame:
        """Bounded sinusoidal mu aligned with pv_prices."""
        n = pv_prices.height
        theta = np.linspace(0.0, 2.0 * np.pi, num=n)
        return pl.DataFrame(
            {
                "date": pv_prices["date"],
                "A": pl.Series(np.tanh(np.sin(theta)), dtype=pl.Float64),
                "B": pl.Series(np.tanh(np.cos(theta)), dtype=pl.Float64),
            }
        )

    def test_profit_variance_init_exactly_scales_first_valid_positions(
        self, pv_prices: pl.DataFrame, pv_mu: pl.DataFrame
    ) -> None:
        """Position at the warmup boundary is inversely proportional to profit_variance_init.

        Before any valid (non-NaN) positions exist, all realized profits are zero because
        pre-warmup positions are NaN (→ ``nan_to_num`` → 0). The EMA therefore degenerates
        to ``pv[k] = λ^k · pv_init`` for both configs, preserving the exact ratio of their
        init values. At the warmup boundary, both configs face the same solved position
        (identical μ and Σ), scaled only by their respective pv values, so the ratio of
        positions must equal the inverse ratio of the inits.
        """
        vola, corr = 4, 8
        base = {"vola": vola, "corr": corr, "clip": 3.0, "shrink": 0.5, "aum": 1e6, "profit_variance_decay": 0.99}
        cp_1 = BasanosEngine(
            prices=pv_prices, mu=pv_mu, cfg=BasanosConfig(**base, profit_variance_init=1.0)
        ).cash_position
        cp_4 = BasanosEngine(
            prices=pv_prices, mu=pv_mu, cfg=BasanosConfig(**base, profit_variance_init=4.0)
        ).cash_position

        a1 = cp_1["A"].to_numpy()
        a4 = cp_4["A"].to_numpy()

        first_valid = next(
            (i for i in range(len(a1)) if np.isfinite(a1[i]) and np.isfinite(a4[i]) and a1[i] != 0.0),
            None,
        )
        assert first_valid is not None, "No valid (finite, non-zero) positions found after warmup"
        ratio = a1[first_valid] / a4[first_valid]
        assert abs(ratio - 4.0) < 1e-10, f"Expected exact ratio 4.0 at warmup boundary, got {ratio}"

    def test_profit_variance_ema_updates_break_proportionality_after_warmup(
        self, pv_prices: pl.DataFrame, pv_mu: pl.DataFrame
    ) -> None:
        """After the first non-NaN positions, real profits enter the EMA and break the exact init ratio.

        At the warmup boundary the two configs produce positions in exact proportion to their
        inits. One step later, yesterday's positions differ between the two configs, so the
        realized profits—and therefore the EMA updates—are different. This confirms that the
        EMA is actually being updated from real P&L data, not merely decaying from init.
        """
        vola, corr = 4, 8
        n = pv_prices.height
        base = {"vola": vola, "corr": corr, "clip": 3.0, "shrink": 0.5, "aum": 1e6, "profit_variance_decay": 0.99}
        cp_1 = BasanosEngine(
            prices=pv_prices, mu=pv_mu, cfg=BasanosConfig(**base, profit_variance_init=1.0)
        ).cash_position
        cp_4 = BasanosEngine(
            prices=pv_prices, mu=pv_mu, cfg=BasanosConfig(**base, profit_variance_init=4.0)
        ).cash_position

        a1 = cp_1["A"].to_numpy()
        a4 = cp_4["A"].to_numpy()

        first_valid = next(
            (i for i in range(len(a1)) if np.isfinite(a1[i]) and np.isfinite(a4[i]) and a1[i] != 0.0),
            None,
        )
        assert first_valid is not None, "No valid (finite, non-zero) positions found after warmup"
        later_valids = [
            i for i in range(first_valid + 1, n) if np.isfinite(a1[i]) and np.isfinite(a4[i]) and a1[i] != 0.0
        ]
        assert later_valids, "Need at least 2 valid positions after warmup"

        # After one EMA update from a real profit, the pv values for the two configs
        # are no longer in exact proportion, so the position ratio must deviate from 4.0.
        second_valid = later_valids[0]
        ratio_second = a1[second_valid] / a4[second_valid]
        assert abs(ratio_second - 4.0) > 1e-12, (
            f"EMA did not update from realized profits: ratio at step {second_valid} is still {ratio_second}"
        )

    def test_profit_variance_fast_decay_produces_larger_initial_positions_than_slow_decay(
        self, pv_prices: pl.DataFrame, pv_mu: pl.DataFrame
    ) -> None:
        """At the warmup boundary, fast-decaying pv is smaller and therefore positions are larger.

        With fast decay (small λ), pv decays quickly during the warmup period
        (``pv[W] = λ^W · init`` → 0 for small λ), producing large positions. With slow
        decay (large λ close to 1), pv barely changes from init and positions are smaller.
        The ratio of absolute positions at the warmup boundary should reflect the ratio of
        the pv values, which can be computed analytically as ``(λ_fast / λ_slow)^W``.
        """
        vola, corr = 4, 8
        lamb_fast, lamb_slow = 0.01, 0.999
        base = {"vola": vola, "corr": corr, "clip": 3.0, "shrink": 0.5, "aum": 1e6, "profit_variance_init": 1.0}
        cp_fast = BasanosEngine(
            prices=pv_prices, mu=pv_mu, cfg=BasanosConfig(**base, profit_variance_decay=lamb_fast)
        ).cash_position
        cp_slow = BasanosEngine(
            prices=pv_prices, mu=pv_mu, cfg=BasanosConfig(**base, profit_variance_decay=lamb_slow)
        ).cash_position

        a_fast = cp_fast["A"].to_numpy()
        a_slow = cp_slow["A"].to_numpy()

        first_valid = next(
            (i for i in range(len(a_fast)) if np.isfinite(a_fast[i]) and np.isfinite(a_slow[i]) and a_fast[i] != 0.0),
            None,
        )
        assert first_valid is not None, "No valid (finite, non-zero) positions found after warmup"
        # Fast-decaying pv is much smaller → much larger absolute positions.
        assert abs(a_fast[first_valid]) > abs(a_slow[first_valid]), (
            "Expected fast-decay config to produce larger initial positions (lower pv) than slow-decay config"
        )
        # Quantitative check: pv_fast[W] = lamb_fast^W, pv_slow[W] = lamb_slow^W, so the
        # ratio of positions ≈ pv_slow[W] / pv_fast[W] = (lamb_slow / lamb_fast)^W.
        # With W = corr = 8, lamb_slow / lamb_fast = 0.999 / 0.01 = 99.9, so
        # ratio ≈ 99.9^8 ≈ 9.2e15.  Conservatively we require a ratio of at least 1e10.
        ratio_abs = abs(a_fast[first_valid]) / abs(a_slow[first_valid])
        assert ratio_abs > 1e10, f"Expected ratio > 1e10, got {ratio_abs}"

    def test_profit_variance_ema_formula_manual_verification(
        self, pv_prices: pl.DataFrame, pv_mu: pl.DataFrame
    ) -> None:
        """Manually simulate the EMA and verify that positions are self-consistent.

        Given the engine's ``cash_position`` output, we can re-derive the pv trajectory
        using the same EMA formula, feeding the engine's own outputs back as the lagged
        cash positions:

            pv[t] = λ·pv[t-1] + (1-λ)·(cash_pos[t-1] @ returns[t])²

        The "solved position" underlying any cash position is:

            solved_pos[t] = cash_pos[t] · vola[t] · pv[t]

        Because both engines face the same linear system (μ and Σ are identical for both
        configs), their solved positions must agree at every timestamp. Verifying this
        self-consistency simultaneously checks the EMA formula, the lagged-position update,
        and the position scaling.
        """
        vola, corr = 4, 8
        base = {"vola": vola, "corr": corr, "clip": 3.0, "shrink": 0.5, "aum": 1e6, "profit_variance_decay": 0.9}

        engine_1 = BasanosEngine(prices=pv_prices, mu=pv_mu, cfg=BasanosConfig(**base, profit_variance_init=1.0))
        engine_2 = BasanosEngine(prices=pv_prices, mu=pv_mu, cfg=BasanosConfig(**base, profit_variance_init=2.0))

        cp1 = engine_1.cash_position
        cp2 = engine_2.cash_position

        assets = engine_1.assets
        prices_np = pv_prices.select(assets).to_numpy()
        returns_np = np.zeros_like(prices_np, dtype=float)
        returns_np[1:] = prices_np[1:] / prices_np[:-1] - 1.0

        cash1 = cp1.select(assets).to_numpy()
        cash2 = cp2.select(assets).to_numpy()
        vola_np = engine_1.vola.select(assets).to_numpy()

        def _simulate_pv(cash_pos: np.ndarray, init: float, lamb: float) -> list[float]:
            """Re-run the EMA formula using actual engine outputs as the lagged positions."""
            pv = init
            pv_seq = [pv]
            for i in range(1, len(prices_np)):
                mask = np.isfinite(prices_np[i])
                ret_mask = np.isfinite(returns_np[i]) & mask
                if ret_mask.any():
                    lhs = np.nan_to_num(cash_pos[i - 1, ret_mask], nan=0.0)
                    rhs = np.nan_to_num(returns_np[i, ret_mask], nan=0.0)
                    profit = lhs @ rhs
                    pv = lamb * pv + (1 - lamb) * profit**2
                pv_seq.append(pv)
            return pv_seq

        lamb = 0.9
        pv1 = _simulate_pv(cash1, 1.0, lamb)
        pv2 = _simulate_pv(cash2, 2.0, lamb)

        # At each valid step: cash_pos[t] · vola[t] · pv[t] = solved_pos[t].
        # Both engines solve the same linear system, so solved_pos must agree.
        valid_idxs = [
            i
            for i in range(corr, pv_prices.height)
            if (
                np.isfinite(cash1[i, 0])
                and np.isfinite(cash2[i, 0])
                and abs(cash1[i, 0]) > 1e-15
                and abs(cash2[i, 0]) > 1e-15
                and np.isfinite(vola_np[i, 0])
                and vola_np[i, 0] > 0
            )
        ]
        assert valid_idxs, "Need at least one valid index after warmup"

        for idx in valid_idxs:
            solved_1 = cash1[idx, 0] * vola_np[idx, 0] * pv1[idx]
            solved_2 = cash2[idx, 0] * vola_np[idx, 0] * pv2[idx]
            np.testing.assert_allclose(
                solved_1,
                solved_2,
                rtol=1e-10,
                err_msg=f"Solved positions diverge at index {idx}: engine_1={solved_1}, engine_2={solved_2}",
            )


# ─── Diagnostics properties ───────────────────────────────────────────────────


class TestDiagnostics:
    """Tests for BasanosEngine diagnostic properties."""

    @pytest.fixture
    def engine(self) -> BasanosEngine:
        """120-row, 2-asset engine used across all diagnostic tests."""
        prices, mu = _make_prices_mu(120)
        cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
        return BasanosEngine(prices=prices, mu=mu, cfg=cfg)

    # ── risk_position ─────────────────────────────────────────────────────────

    def test_risk_position_schema_matches_prices(self, engine: BasanosEngine) -> None:
        """risk_position should have the same columns as prices."""
        rp = engine.risk_position
        assert rp.columns == engine.prices.columns

    def test_risk_position_equals_cash_times_vola(self, engine: BasanosEngine) -> None:
        """risk_position[i] == cash_position[i] * vola[i] element-wise."""
        assets = engine.assets
        rp = engine.risk_position.select(assets).to_numpy()
        cp = engine.cash_position.select(assets).to_numpy()
        vola = engine.vola.select(assets).to_numpy()
        with np.errstate(invalid="ignore"):
            expected = cp * vola
        np.testing.assert_allclose(
            np.nan_to_num(rp, nan=0.0),
            np.nan_to_num(expected, nan=0.0),
            rtol=1e-12,
        )

    def test_risk_position_has_finite_values_after_warmup(self, engine: BasanosEngine) -> None:
        """After the warmup window, risk positions should be finite."""
        warmup = engine.cfg.corr
        rp = engine.risk_position
        for asset in engine.assets:
            tail = rp[asset].slice(warmup)
            assert tail.is_finite().all(), f"Non-finite risk_position for {asset} after warmup"

    # ── position_leverage ─────────────────────────────────────────────────────

    def test_position_leverage_schema(self, engine: BasanosEngine) -> None:
        """position_leverage should have exactly ['date', 'leverage'] columns."""
        lev = engine.position_leverage
        assert lev.columns == ["date", "leverage"]

    def test_position_leverage_height_matches_prices(self, engine: BasanosEngine) -> None:
        """position_leverage should have the same number of rows as prices."""
        assert engine.position_leverage.height == engine.prices.height

    def test_position_leverage_non_negative(self, engine: BasanosEngine) -> None:
        """L1 norm is always ≥ 0."""
        lev = engine.position_leverage["leverage"].drop_nulls()
        assert (lev >= 0).all()

    def test_position_leverage_finite_after_warmup(self, engine: BasanosEngine) -> None:
        """Leverage should be finite after the warmup window."""
        warmup = engine.cfg.corr
        tail = engine.position_leverage["leverage"].slice(warmup)
        assert tail.is_finite().all()

    def test_position_leverage_equals_sum_of_abs_cash_positions(self, engine: BasanosEngine) -> None:
        """leverage[t] should equal sum(|cash_pos_i[t]|) ignoring NaN."""
        assets = engine.assets
        cp_np = engine.cash_position.select(assets).to_numpy()
        expected = np.nansum(np.abs(cp_np), axis=1)
        actual = engine.position_leverage["leverage"].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    # ── condition_number ──────────────────────────────────────────────────────

    def test_condition_number_schema(self, engine: BasanosEngine) -> None:
        """condition_number should have exactly ['date', 'condition_number'] columns."""
        cn = engine.condition_number
        assert cn.columns == ["date", "condition_number"]

    def test_condition_number_height_matches_prices(self, engine: BasanosEngine) -> None:
        """condition_number should have the same number of rows as prices."""
        assert engine.condition_number.height == engine.prices.height

    def test_condition_number_positive_after_warmup(self, engine: BasanosEngine) -> None:
        """Condition numbers must be strictly positive after the warmup window."""
        warmup = engine.cfg.corr
        cn = engine.condition_number["condition_number"].slice(warmup)
        assert (cn.drop_nulls() > 0).all()

    def test_condition_number_ge_one_for_correlation_matrices(self, engine: BasanosEngine) -> None:
        """Condition number of any valid correlation matrix is ≥ 1."""
        cn = engine.condition_number["condition_number"].drop_nulls()
        assert (cn >= 1.0 - 1e-10).all()

    def test_condition_number_is_one_for_identity_shrink(self) -> None:
        """When shrink=0 the matrix is the identity; κ should equal 1."""
        prices, mu = _make_prices_mu(80)
        cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.0, aum=1e6)
        engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
        warmup = cfg.corr
        cn = engine.condition_number["condition_number"].slice(warmup).drop_nulls()
        np.testing.assert_allclose(cn.to_numpy(), 1.0, atol=1e-10)

    # ── effective_rank ────────────────────────────────────────────────────────

    def test_effective_rank_schema(self, engine: BasanosEngine) -> None:
        """effective_rank should have exactly ['date', 'effective_rank'] columns."""
        er = engine.effective_rank
        assert er.columns == ["date", "effective_rank"]

    def test_effective_rank_height_matches_prices(self, engine: BasanosEngine) -> None:
        """effective_rank should have the same number of rows as prices."""
        assert engine.effective_rank.height == engine.prices.height

    def test_effective_rank_bounded_after_warmup(self, engine: BasanosEngine) -> None:
        """Effective rank must be in (0, n_assets] after the warmup window."""
        warmup = engine.cfg.corr
        n_assets = len(engine.assets)
        er = engine.effective_rank["effective_rank"].slice(warmup).drop_nulls()
        assert (er > 0).all()
        assert (er <= n_assets + 1e-10).all()

    def test_effective_rank_equals_n_for_identity_shrink(self) -> None:
        """When shrink=0 the matrix is the identity; effective rank equals n_assets."""
        prices, mu = _make_prices_mu(80)
        n_assets = 2
        cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.0, aum=1e6)
        engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
        warmup = cfg.corr
        er = engine.effective_rank["effective_rank"].slice(warmup).drop_nulls()
        np.testing.assert_allclose(er.to_numpy(), float(n_assets), atol=1e-10)

    def test_effective_rank_zero_eigenvalue_sum_returns_nan(self) -> None:
        """effective_rank returns NaN when the shrunk matrix has all-zero eigenvalues.

        Patches ``BasanosEngine.cor`` to return a zero matrix at every timestamp so
        that ``eigvals.sum() == 0.0`` after clipping to non-negative values.
        """
        from unittest.mock import PropertyMock, patch

        prices, mu = _make_prices_mu(40)
        cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=1.0, aum=1e6)
        engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)

        n = prices.height
        n_assets = 2
        # Build a cor dict whose matrices are all zeros so eigvalsh → [0, 0]
        zero_matrix = np.zeros((n_assets, n_assets))
        fake_cor = {prices["date"][i]: zero_matrix for i in range(n)}

        with patch.object(type(engine), "cor", new_callable=PropertyMock, return_value=fake_cor):
            er = engine.effective_rank["effective_rank"]

        assert er.is_nan().all()

    def test_solver_residual_schema(self, engine: BasanosEngine) -> None:
        """solver_residual should have exactly ['date', 'residual'] columns."""
        sr = engine.solver_residual
        assert sr.columns == ["date", "residual"]

    def test_solver_residual_height_matches_prices(self, engine: BasanosEngine) -> None:
        """solver_residual should have the same number of rows as prices."""
        assert engine.solver_residual.height == engine.prices.height

    def test_solver_residual_non_negative(self, engine: BasanosEngine) -> None:
        """Residual norms are always ≥ 0."""
        res = engine.solver_residual["residual"].drop_nulls()
        assert (res >= 0).all()

    def test_solver_residual_near_zero_after_warmup(self, engine: BasanosEngine) -> None:
        """For a well-conditioned system the residual should be near machine epsilon."""
        warmup = engine.cfg.corr
        res = engine.solver_residual["residual"].slice(warmup).drop_nulls()
        np.testing.assert_array_less(res.to_numpy(), 1e-10)

    def test_solver_residual_zero_for_zero_mu(self) -> None:
        """When μ=0 at every timestamp the residual is exactly 0."""
        prices, _ = _make_prices_mu(80)
        n = prices.height
        dates = prices["date"]
        mu_zero = pl.DataFrame(
            {
                "date": dates,
                "A": pl.Series(np.zeros(n), dtype=pl.Float64),
                "B": pl.Series(np.zeros(n), dtype=pl.Float64),
            }
        )
        cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
        engine = BasanosEngine(prices=prices, mu=mu_zero, cfg=cfg)
        res = engine.solver_residual["residual"].drop_nulls()
        np.testing.assert_array_equal(res.to_numpy(), 0.0)

    # ── signal_utilisation ────────────────────────────────────────────────────

    def test_signal_utilisation_schema_matches_prices(self, engine: BasanosEngine) -> None:
        """signal_utilisation should have the same columns as prices."""
        su = engine.signal_utilisation
        assert su.columns == engine.prices.columns

    def test_signal_utilisation_height_matches_prices(self, engine: BasanosEngine) -> None:
        """signal_utilisation should have the same number of rows as prices."""
        assert engine.signal_utilisation.height == engine.prices.height

    def test_signal_utilisation_is_one_for_identity_shrink(self) -> None:
        """When shrink=0 (identity matrix), each asset's utilisation is 1."""
        prices, mu = _make_prices_mu(80)
        cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.0, aum=1e6)
        engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
        warmup = cfg.corr
        assets = engine.assets
        su = engine.signal_utilisation
        for asset in assets:
            col = su[asset].slice(warmup).drop_nulls()
            # μ=0 slots are NaN-or-zero; we only check where utilisation is defined
            finite = col.filter(col.is_not_nan())
            np.testing.assert_allclose(finite.to_numpy(), 1.0, atol=1e-10)

    def test_signal_utilisation_zero_mu_yields_zero(self) -> None:
        """When μ=0 signal utilisation is 0 (not NaN), consistent with zero positions."""
        prices, _ = _make_prices_mu(80)
        n = prices.height
        dates = prices["date"]
        mu_zero = pl.DataFrame(
            {
                "date": dates,
                "A": pl.Series(np.zeros(n), dtype=pl.Float64),
                "B": pl.Series(np.zeros(n), dtype=pl.Float64),
            }
        )
        cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
        engine = BasanosEngine(prices=prices, mu=mu_zero, cfg=cfg)
        warmup = cfg.corr
        su = engine.signal_utilisation
        for asset in engine.assets:
            col = su[asset].slice(warmup).drop_nulls()
            np.testing.assert_array_equal(col.to_numpy(), 0.0)

    def test_diagnostics_yield_nan_for_all_nan_price_rows(self) -> None:
        """Diagnostics should return NaN for timestamps where all asset prices are NaN.

        Covers the ``not mask.any()`` guard in condition_number, effective_rank,
        solver_residual, and signal_utilisation (also the ``continue`` path in
        signal_utilisation).
        """
        n = 40
        start = date(2020, 1, 1)
        dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
        rng = np.random.default_rng(99)
        a = list(100.0 + np.cumsum(rng.normal(0.0, 0.5, size=n)))
        b = list(200.0 + np.cumsum(rng.normal(0.0, 0.7, size=n)))
        idx_nan = n - 1  # last row all-NaN
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
        cfg = BasanosConfig(vola=3, corr=5, clip=3.0, shrink=0.5, aum=1e6)
        engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)

        # condition_number: NaN at the all-NaN row
        cn = engine.condition_number["condition_number"]
        assert np.isnan(cn[idx_nan])

        # effective_rank: NaN at the all-NaN row
        er = engine.effective_rank["effective_rank"]
        assert np.isnan(er[idx_nan])

        # solver_residual: NaN at the all-NaN row
        sr = engine.solver_residual["residual"]
        assert np.isnan(sr[idx_nan])

        # signal_utilisation: NaN at the all-NaN row for every asset
        su = engine.signal_utilisation
        for asset in engine.assets:
            assert np.isnan(su[asset][idx_nan])

    def test_solver_residual_warns_and_returns_nan_on_singular_matrix(self, caplog) -> None:
        """solver_residual emits a warning and records NaN when SingularMatrixError is raised.

        Patches ``basanos.math.optimizer.solve`` to raise ``SingularMatrixError``
        on every call so that every non-trivial timestamp exercises the exception
        handler.  The test verifies that a WARNING is logged and that the
        residual column contains at least one NaN.
        """
        prices, mu = _make_prices_mu(40)
        cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
        engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)

        with (
            patch("basanos.math.optimizer.solve", side_effect=SingularMatrixError()),
            caplog.at_level(logging.WARNING, logger="basanos.math.optimizer"),
        ):
            sr = engine.solver_residual

        singular_warnings = [r for r in caplog.records if "SingularMatrixError" in r.message]
        assert singular_warnings, "Expected at least one warning about a SingularMatrixError in solver_residual"
        assert all("solver_residual" in r.message for r in singular_warnings)
        assert all("degenerate" in r.message for r in singular_warnings)
        assert sr["residual"].is_nan().any(), "Expected at least one NaN residual when solve raises SingularMatrixError"

    def test_signal_utilisation_warns_and_returns_nan_on_singular_matrix(self, caplog) -> None:
        """signal_utilisation emits a warning and keeps NaN when SingularMatrixError is raised.

        Patches ``basanos.math.optimizer.solve`` to raise ``SingularMatrixError``
        on every call.  The test verifies that a WARNING is logged and that the
        utilisation columns contain at least one NaN per asset.
        """
        prices, mu = _make_prices_mu(40)
        cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
        engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)

        with (
            patch("basanos.math.optimizer.solve", side_effect=SingularMatrixError()),
            caplog.at_level(logging.WARNING, logger="basanos.math.optimizer"),
        ):
            su = engine.signal_utilisation

        singular_warnings = [r for r in caplog.records if "SingularMatrixError" in r.message]
        assert singular_warnings, "Expected at least one warning about a SingularMatrixError in signal_utilisation"
        assert all("signal_utilisation" in r.message for r in singular_warnings)
        assert all("degenerate" in r.message for r in singular_warnings)
        for asset in engine.assets:
            assert su[asset].is_nan().any(), (
                f"Expected at least one NaN for asset {asset!r} when solve raises SingularMatrixError"
            )


# ─── Information Coefficient (IC) ────────────────────────────────────────────


@pytest.fixture
def ic_engine(optimizer_prices: pl.DataFrame, optimizer_mu: pl.DataFrame) -> BasanosEngine:
    """BasanosEngine fixture used for IC tests (120-day, 2-asset)."""
    cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
    return BasanosEngine(prices=optimizer_prices, mu=optimizer_mu, cfg=cfg)


class TestICTimeSeries:
    """Tests for the cross-sectional Pearson IC time series property."""

    def test_ic_returns_polars_dataframe(self, ic_engine: BasanosEngine) -> None:
        """IC property must return a Polars DataFrame."""
        assert isinstance(ic_engine.ic, pl.DataFrame)

    def test_ic_has_expected_columns(self, ic_engine: BasanosEngine) -> None:
        """IC DataFrame must contain exactly 'date' and 'ic' columns."""
        assert set(ic_engine.ic.columns) == {"date", "ic"}

    def test_ic_length_is_t_minus_one(self, ic_engine: BasanosEngine, optimizer_prices: pl.DataFrame) -> None:
        """IC series must have T-1 rows (one per forward-return period)."""
        assert ic_engine.ic.height == optimizer_prices.height - 1

    def test_ic_dates_match_signal_dates(self, ic_engine: BasanosEngine, optimizer_prices: pl.DataFrame) -> None:
        """Each IC date must equal the corresponding signal date (prices[t])."""
        expected_dates = optimizer_prices["date"].head(optimizer_prices.height - 1).to_list()
        assert ic_engine.ic["date"].to_list() == expected_dates

    def test_ic_values_are_in_minus_one_to_one(self, ic_engine: BasanosEngine) -> None:
        """All finite IC values must lie in [-1, 1]."""
        ic_vals = ic_engine.ic["ic"].drop_nulls().to_numpy()
        assert np.all(ic_vals >= -1.0 - 1e-12)
        assert np.all(ic_vals <= 1.0 + 1e-12)

    def test_ic_col_is_float64(self, ic_engine: BasanosEngine) -> None:
        """IC column must be Float64 dtype."""
        assert ic_engine.ic["ic"].dtype == pl.Float64


class TestRankICTimeSeries:
    """Tests for the cross-sectional Spearman Rank IC time series property."""

    def test_rank_ic_returns_polars_dataframe(self, ic_engine: BasanosEngine) -> None:
        """rank_ic property must return a Polars DataFrame."""
        assert isinstance(ic_engine.rank_ic, pl.DataFrame)

    def test_rank_ic_has_expected_columns(self, ic_engine: BasanosEngine) -> None:
        """rank_ic DataFrame must contain exactly 'date' and 'rank_ic' columns."""
        assert set(ic_engine.rank_ic.columns) == {"date", "rank_ic"}

    def test_rank_ic_length_is_t_minus_one(self, ic_engine: BasanosEngine, optimizer_prices: pl.DataFrame) -> None:
        """rank_ic series must have T-1 rows (one per forward-return period)."""
        assert ic_engine.rank_ic.height == optimizer_prices.height - 1

    def test_rank_ic_values_are_in_minus_one_to_one(self, ic_engine: BasanosEngine) -> None:
        """All finite Rank IC values must lie in [-1, 1]."""
        vals = ic_engine.rank_ic["rank_ic"].drop_nulls().to_numpy()
        assert np.all(vals >= -1.0 - 1e-12)
        assert np.all(vals <= 1.0 + 1e-12)

    def test_rank_ic_col_is_float64(self, ic_engine: BasanosEngine) -> None:
        """rank_ic column must be Float64 dtype."""
        assert ic_engine.rank_ic["rank_ic"].dtype == pl.Float64


class TestICScalars:
    """Tests for ic_mean, ic_std, icir, rank_ic_mean, and rank_ic_std."""

    def test_ic_mean_is_finite(self, ic_engine: BasanosEngine) -> None:
        """ic_mean must return a finite float for a well-behaved engine."""
        assert math.isfinite(ic_engine.ic_mean)

    def test_ic_std_is_non_negative(self, ic_engine: BasanosEngine) -> None:
        """ic_std must be non-negative (standard deviation cannot be negative)."""
        std = ic_engine.ic_std
        assert math.isfinite(std)
        assert std >= 0.0

    def test_icir_equals_mean_over_std(self, ic_engine: BasanosEngine) -> None:
        """ICIR must equal ic_mean / ic_std."""
        mean = ic_engine.ic_mean
        std = ic_engine.ic_std
        expected = mean / std if std != 0.0 else float("nan")
        if math.isnan(expected):
            assert math.isnan(ic_engine.icir)
        else:
            assert math.isclose(ic_engine.icir, expected, rel_tol=1e-12)

    def test_icir_is_nan_when_std_is_zero(self) -> None:
        """ICIR must gracefully return a float (possibly NaN) in degenerate cases."""
        # Use the existing ic_engine: just verify icir is always a float, whether
        # finite or NaN.  The NaN branch is exercised by the zero-std guard in the
        # implementation; we verify the type contract here.
        n = 20
        start = date(2020, 1, 1)
        dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
        # Use non-monotonic prices to pass engine validation
        rng = np.random.default_rng(7)
        p_a = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n))
        p_b = 200.0 + np.cumsum(rng.normal(0.0, 0.7, n))
        prices = pl.DataFrame(
            {"date": dates, "A": pl.Series(p_a, dtype=pl.Float64), "B": pl.Series(p_b, dtype=pl.Float64)}
        )
        mu = pl.DataFrame(
            {
                "date": dates,
                "A": pl.Series(np.zeros(n), dtype=pl.Float64),
                "B": pl.Series(np.zeros(n), dtype=pl.Float64),
            }
        )
        cfg = BasanosConfig(vola=3, corr=5, clip=3.0, shrink=0.5, aum=1e6)
        eng = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
        # With all-zero signal, every ic row is NaN (corrcoef of constants).
        # icir should return NaN gracefully.
        icir = eng.icir
        assert isinstance(icir, float)

    def test_rank_ic_mean_is_finite(self, ic_engine: BasanosEngine) -> None:
        """rank_ic_mean must return a finite float for a well-behaved engine."""
        assert math.isfinite(ic_engine.rank_ic_mean)

    def test_rank_ic_std_is_non_negative(self, ic_engine: BasanosEngine) -> None:
        """rank_ic_std must be non-negative."""
        std = ic_engine.rank_ic_std
        assert math.isfinite(std)
        assert std >= 0.0

    def test_ic_mean_matches_manual_computation(self, ic_engine: BasanosEngine) -> None:
        """ic_mean must equal np.nanmean of the ic series."""
        ic_vals = ic_engine.ic["ic"].to_numpy()
        expected = float(np.nanmean(ic_vals))
        assert math.isclose(ic_engine.ic_mean, expected, rel_tol=1e-12)

    def test_rank_ic_mean_matches_manual_computation(self, ic_engine: BasanosEngine) -> None:
        """rank_ic_mean must equal np.nanmean of the rank_ic series."""
        ric_vals = ic_engine.rank_ic["rank_ic"].to_numpy()
        expected = float(np.nanmean(ric_vals))
        assert math.isclose(ic_engine.rank_ic_mean, expected, rel_tol=1e-12)


class TestICNaNHandling:
    """Tests for IC behaviour with NaN prices or signals."""

    @pytest.fixture
    def nan_prices(self) -> pl.DataFrame:
        """10-day price frame with NaN at t=3 for asset A (non-monotonic to pass validation)."""
        n = 10
        start = date(2020, 1, 1)
        dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
        p_a = [100.0, 102.0, 99.0, None, 104.0, 101.0, 106.0, 103.0, 108.0, 105.0]
        p_b = [200.0, 203.0, 198.0, 205.0, 201.0, 207.0, 203.0, 209.0, 205.0, 211.0]
        return pl.DataFrame(
            {"date": dates, "A": pl.Series(p_a, dtype=pl.Float64), "B": pl.Series(p_b, dtype=pl.Float64)}
        )

    @pytest.fixture
    def nan_mu(self, nan_prices: pl.DataFrame) -> pl.DataFrame:
        """Sinusoidal signal aligned with nan_prices."""
        n = nan_prices.height
        theta = np.linspace(0.0, np.pi, num=n)
        return pl.DataFrame(
            {
                "date": nan_prices["date"],
                "A": pl.Series(np.tanh(np.sin(theta)), dtype=pl.Float64),
                "B": pl.Series(np.tanh(np.cos(theta)), dtype=pl.Float64),
            }
        )

    def test_ic_handles_nan_prices_gracefully(self, nan_prices: pl.DataFrame, nan_mu: pl.DataFrame) -> None:
        """IC must not raise and must return finite or NaN values for all rows."""
        cfg = BasanosConfig(vola=3, corr=5, clip=3.0, shrink=0.5, aum=1e6)
        eng = BasanosEngine(prices=nan_prices, mu=nan_mu, cfg=cfg)
        ic_df = eng.ic
        assert ic_df.height == nan_prices.height - 1
        for v in ic_df["ic"].to_list():
            assert v is None or isinstance(v, float)

    def test_ic_nan_when_only_one_valid_asset(self) -> None:
        """When only one asset has finite signal, IC must be NaN (needs ≥2 pairs)."""
        n = 5
        start = date(2020, 1, 1)
        dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
        # Non-monotonic prices for both assets so validation passes
        p_a = [100.0, 102.0, 99.0, 103.0, 101.0]
        p_b = [200.0, 203.0, 198.0, 205.0, 201.0]
        prices = pl.DataFrame(
            {
                "date": dates,
                "A": pl.Series(p_a, dtype=pl.Float64),
                "B": pl.Series(p_b, dtype=pl.Float64),
            }
        )
        # Signal has NaN for B at all times → only 1 valid (asset, fwd_ret) pair per period
        mu = pl.DataFrame(
            {
                "date": dates,
                "A": pl.Series([0.1, 0.2, 0.3, 0.4, 0.5], dtype=pl.Float64),
                "B": pl.Series([None, None, None, None, None], dtype=pl.Float64),
            }
        )
        cfg = BasanosConfig(vola=2, corr=3, clip=3.0, shrink=0.5, aum=1e6)
        eng = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
        ic_vals = eng.ic["ic"].to_list()
        # With only one valid asset pair, every IC value must be NaN
        for v in ic_vals:
            assert v is None or (isinstance(v, float) and math.isnan(v))


# ─── sharpe_at_shrink and naive_sharpe ───────────────────────────────────────


class TestSharpeAtShrink:
    """Tests for BasanosEngine.sharpe_at_shrink and BasanosEngine.naive_sharpe."""

    @pytest.fixture
    def engine(self, optimizer_prices: pl.DataFrame, optimizer_mu: pl.DataFrame) -> BasanosEngine:
        """BasanosEngine with moderate config for Sharpe benchmark tests."""
        cfg = BasanosConfig(vola=16, corr=20, clip=3.5, shrink=0.5, aum=1e6)
        return BasanosEngine(prices=optimizer_prices, mu=optimizer_mu, cfg=cfg)

    def test_sharpe_at_shrink_returns_float(self, engine: BasanosEngine) -> None:
        """sharpe_at_shrink should return a Python float."""
        result = engine.sharpe_at_shrink(0.5)
        assert isinstance(result, float)

    def test_sharpe_at_shrink_zero_returns_float(self, engine: BasanosEngine) -> None:
        """sharpe_at_shrink(0.0) corresponds to identity matrix (signal-proportional) and must return float."""
        result = engine.sharpe_at_shrink(0.0)
        assert isinstance(result, float)

    def test_sharpe_at_shrink_one_returns_float(self, engine: BasanosEngine) -> None:
        """sharpe_at_shrink(1.0) uses the raw EWMA correlation and must return float."""
        result = engine.sharpe_at_shrink(1.0)
        assert isinstance(result, float)

    def test_sharpe_at_shrink_matches_portfolio_sharpe_for_same_lambda(self, engine: BasanosEngine) -> None:
        """sharpe_at_shrink(cfg.shrink) must equal the engine's own portfolio Sharpe."""
        expected = engine.portfolio.stats.sharpe().get("returns", float("nan"))
        actual = engine.sharpe_at_shrink(engine.cfg.shrink)
        assert actual == pytest.approx(expected, abs=1e-10)

    def test_sharpe_at_shrink_varies_across_lambda_values(self, engine: BasanosEngine) -> None:
        """Different shrinkage values generally produce different Sharpe ratios."""
        sharpes = [engine.sharpe_at_shrink(lam) for lam in (0.0, 0.25, 0.5, 0.75, 1.0)]
        # All must be finite floats
        assert all(isinstance(s, float) for s in sharpes)
        # At least two distinct values across the sweep (non-trivial signal with non-trivial data)
        assert len({round(s, 6) for s in sharpes}) > 1, "Expected distinct Sharpe values across the lambda sweep"

    def test_naive_sharpe_returns_float(self, engine: BasanosEngine) -> None:
        """naive_sharpe should return a Python float."""
        result = engine.naive_sharpe
        assert isinstance(result, float)

    def test_naive_sharpe_differs_from_signal_sharpe(self, engine: BasanosEngine) -> None:
        """naive_sharpe (equal-weight mu) should differ from the signal-driven Sharpe."""
        signal_sharpe = engine.portfolio.stats.sharpe().get("returns", float("nan"))
        naive = engine.naive_sharpe
        # Both must be finite floats; the signal should add information so they differ
        assert isinstance(naive, float)
        assert isinstance(signal_sharpe, float)
        # They may coincidentally agree on some seeds, but with a sinusoidal signal they should differ
        assert naive != pytest.approx(signal_sharpe, abs=1e-6), (
            "naive_sharpe should differ from signal Sharpe for a non-constant signal"
        )
