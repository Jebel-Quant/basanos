"""Integration and smoke tests for the full BasanosEngine pipeline.

Tests
-----
test_smoke_all_public_properties
    Constructs a BasanosEngine with a ~100-row, 5-asset dataset, exercises
    every public property, and asserts no exceptions and sensible shapes/types.

test_regression_cash_position
    Fixes a small synthetic dataset (seed=0, 80 rows, 3 assets) and asserts
    that ``cash_position`` values match known-good reference values so that
    numerical regressions are immediately visible.

test_regression_position_leverage
    Same fixed dataset; asserts ``position_leverage`` matches reference values.

test_realistic_scale
    Marked ``@pytest.mark.slow``.  Constructs a 252-row x 20-asset engine and
    asserts that the full pipeline completes without error within a reasonable
    wall-clock bound.
"""

from __future__ import annotations

import math
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from basanos.analytics import Portfolio
from basanos.math import BasanosConfig, BasanosEngine

# ─── shared helpers ──────────────────────────────────────────────────────────


def _make_engine(n: int, n_assets: int, seed: int = 42) -> BasanosEngine:
    """Return a BasanosEngine with ``n`` daily rows and ``n_assets`` assets."""
    rng = np.random.default_rng(seed)
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    assets = [f"A{i}" for i in range(n_assets)]
    prices_data: dict[str, pl.Series] = {"date": dates}
    mu_data: dict[str, pl.Series] = {"date": dates}
    for asset in assets:
        log_ret = rng.normal(0.0, 0.01, size=n)
        price = 100.0 * np.exp(np.cumsum(log_ret))
        prices_data[asset] = pl.Series(price.tolist())
        signal = rng.normal(0, 1, size=n)
        mu_data[asset] = pl.Series(signal.tolist())

    prices = pl.DataFrame(prices_data)
    mu = pl.DataFrame(mu_data)
    cfg = BasanosConfig(vola=16, corr=32, clip=3.0, shrink=0.5, aum=1e6)
    return BasanosEngine(prices=prices, mu=mu, cfg=cfg)


# ─── regression fixture ───────────────────────────────────────────────────────

# Known-good reference values generated from seed=0, n=80, 3 assets.
# To regenerate: instantiate the regression_engine fixture, call cash_position
# and position_leverage, and read off the values at the rows listed below.
# Row indices are into cash_position / position_leverage DataFrames.
_REGRESSION_ROWS = {
    # row_index: {"A": ..., "B": ..., "C": ..., "leverage": ...}
    40: {
        "A": 80.147377420387,
        "B": 37.062861385912,
        "C": 57.240820014736,
        "leverage": 174.451058821035,
    },
    50: {
        "A": -70.835872743568,
        "B": -80.359228347184,
        "C": 32.067213993143,
        "leverage": 183.262315083894,
    },
    60: {
        "A": 32.655350383737,
        "B": 60.901627409479,
        "C": -81.870898890859,
        "leverage": 175.427876684075,
    },
    79: {
        "A": -84.622435391405,
        "B": 19.379347467408,
        "C": 24.031828098832,
        "leverage": 128.033610957646,
    },
}

_REGRESSION_SEED = 0
_REGRESSION_N = 80
_REGRESSION_ASSETS = ["A", "B", "C"]


@pytest.fixture(scope="module")
def regression_engine() -> BasanosEngine:
    """Small fixed-seed engine used for numerical regression checks."""
    rng = np.random.default_rng(_REGRESSION_SEED)
    n = _REGRESSION_N
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    prices_data: dict[str, pl.Series] = {"date": dates}
    mu_data: dict[str, pl.Series] = {"date": dates}
    for asset in _REGRESSION_ASSETS:
        log_ret = rng.normal(0.0, 0.01, size=n)
        price = 100.0 * np.exp(np.cumsum(log_ret))
        prices_data[asset] = pl.Series(price.tolist())
        signal = rng.normal(0, 1, size=n)
        mu_data[asset] = pl.Series(signal.tolist())

    prices = pl.DataFrame(prices_data)
    mu = pl.DataFrame(mu_data)
    cfg = BasanosConfig(vola=16, corr=32, clip=3.0, shrink=0.5, aum=1e6)
    return BasanosEngine(prices=prices, mu=mu, cfg=cfg)


# ─── 1. Smoke test ────────────────────────────────────────────────────────────


class TestSmoke:
    """Smoke-test every public property of BasanosEngine on a realistic dataset.

    Uses a ~100-row, 5-asset engine so that EWM warm-up completes and all
    properties return non-trivially empty results.
    """

    @pytest.fixture(scope="class")
    def engine(self) -> BasanosEngine:
        """Provide a 100-row, 5-asset BasanosEngine for smoke tests."""
        return _make_engine(n=100, n_assets=5)

    # --- core data properties ---

    def test_assets(self, engine: BasanosEngine) -> None:
        """Assets must return the expected list of column names."""
        assert engine.assets == ["A0", "A1", "A2", "A3", "A4"]

    def test_ret_adj_shape(self, engine: BasanosEngine) -> None:
        """ret_adj must have the same shape as the input prices DataFrame."""
        result = engine.ret_adj
        assert result.shape == engine.prices.shape

    def test_vola_shape(self, engine: BasanosEngine) -> None:
        """Vola must have the same shape as the input prices DataFrame."""
        result = engine.vola
        assert result.shape == engine.prices.shape

    def test_cor_length(self, engine: BasanosEngine) -> None:
        """Cor must contain one entry per timestamp (100 rows)."""
        result = engine.cor
        assert len(result) == 100

    def test_cor_matrix_shape(self, engine: BasanosEngine) -> None:
        """Each correlation matrix in cor must be (n_assets x n_assets)."""
        matrices = list(engine.cor.values())
        assert all(m.shape == (5, 5) for m in matrices)

    def test_cor_tensor_shape(self, engine: BasanosEngine) -> None:
        """cor_tensor must stack all matrices as a (T, N, N) array."""
        tensor = engine.cor_tensor
        assert tensor.shape == (100, 5, 5)

    # --- position properties ---

    def test_cash_position_shape(self, engine: BasanosEngine) -> None:
        """cash_position must have the same shape as the input prices DataFrame."""
        result = engine.cash_position
        assert result.shape == engine.prices.shape

    def test_cash_position_columns(self, engine: BasanosEngine) -> None:
        """cash_position must have ['date'] + asset columns."""
        result = engine.cash_position
        assert result.columns == ["date", *engine.assets]

    def test_risk_position_shape(self, engine: BasanosEngine) -> None:
        """risk_position must have the same shape as the input prices DataFrame."""
        result = engine.risk_position
        assert result.shape == engine.prices.shape

    def test_position_leverage_shape(self, engine: BasanosEngine) -> None:
        """position_leverage must be a two-column DataFrame with the correct columns."""
        result = engine.position_leverage
        assert result.shape == (100, 2)
        assert result.columns == ["date", "leverage"]

    def test_position_leverage_non_negative(self, engine: BasanosEngine) -> None:
        """All leverage values must be non-negative (L1 norm >= 0)."""
        values = engine.position_leverage["leverage"].to_list()
        assert all(v >= 0.0 for v in values)

    # --- diagnostic properties ---

    def test_condition_number_shape(self, engine: BasanosEngine) -> None:
        """condition_number must be a two-column DataFrame."""
        result = engine.condition_number
        assert result.shape == (100, 2)
        assert "condition_number" in result.columns

    def test_condition_number_finite_greater_than_or_equal_one(self, engine: BasanosEngine) -> None:
        """All finite condition numbers must be >= 1.0 (max_sv / min_sv >= 1)."""
        values = [
            v for v in engine.condition_number["condition_number"].to_list() if v is not None and math.isfinite(v)
        ]
        assert all(v >= 1.0 - 1e-9 for v in values)

    def test_effective_rank_shape(self, engine: BasanosEngine) -> None:
        """effective_rank must be a two-column DataFrame."""
        result = engine.effective_rank
        assert result.shape == (100, 2)
        assert "effective_rank" in result.columns

    def test_effective_rank_bounded(self, engine: BasanosEngine) -> None:
        """All finite effective-rank values must lie in [1, n_assets]."""
        n_assets = len(engine.assets)
        values = [v for v in engine.effective_rank["effective_rank"].to_list() if v is not None and math.isfinite(v)]
        assert all(1.0 - 1e-9 <= v <= n_assets + 1e-9 for v in values)

    def test_solver_residual_shape(self, engine: BasanosEngine) -> None:
        """solver_residual must be a two-column DataFrame."""
        result = engine.solver_residual
        assert result.shape == (100, 2)
        assert "residual" in result.columns

    def test_solver_residual_non_negative(self, engine: BasanosEngine) -> None:
        """All non-NaN solver residuals must be non-negative (Euclidean norm >= 0)."""
        values = [v for v in engine.solver_residual["residual"].to_list() if v is not None and math.isfinite(v)]
        assert all(v >= 0.0 for v in values)

    def test_signal_utilisation_shape(self, engine: BasanosEngine) -> None:
        """signal_utilisation must have the same shape as the input prices DataFrame."""
        result = engine.signal_utilisation
        assert result.shape == engine.prices.shape

    # --- IC / signal quality properties ---

    def test_ic_shape(self, engine: BasanosEngine) -> None:
        """Ic must have one row per timestamp except the last (no forward return available)."""
        result = engine.ic
        assert result.shape == (99, 2)
        assert "ic" in result.columns

    def test_rank_ic_shape(self, engine: BasanosEngine) -> None:
        """rank_ic must have one row per timestamp except the last."""
        result = engine.rank_ic
        assert result.shape == (99, 2)
        assert "rank_ic" in result.columns

    def test_ic_mean_is_float(self, engine: BasanosEngine) -> None:
        """ic_mean must be a float scalar."""
        assert isinstance(engine.ic_mean, float)

    def test_ic_std_is_float(self, engine: BasanosEngine) -> None:
        """ic_std must be a float scalar."""
        assert isinstance(engine.ic_std, float)

    def test_icir_is_float(self, engine: BasanosEngine) -> None:
        """Icir must be a float scalar."""
        assert isinstance(engine.icir, float)

    def test_rank_ic_mean_is_float(self, engine: BasanosEngine) -> None:
        """rank_ic_mean must be a float scalar."""
        assert isinstance(engine.rank_ic_mean, float)

    def test_rank_ic_std_is_float(self, engine: BasanosEngine) -> None:
        """rank_ic_std must be a float scalar."""
        assert isinstance(engine.rank_ic_std, float)

    def test_naive_sharpe_is_float(self, engine: BasanosEngine) -> None:
        """naive_sharpe must be a float scalar."""
        assert isinstance(engine.naive_sharpe, float)

    def test_sharpe_at_shrink_is_float(self, engine: BasanosEngine) -> None:
        """sharpe_at_shrink must return a float for a valid shrinkage value."""
        result = engine.sharpe_at_shrink(0.5)
        assert isinstance(result, float)

    # --- portfolio property ---

    def test_portfolio_type(self, engine: BasanosEngine) -> None:
        """Portfolio must return a Portfolio instance."""
        result = engine.portfolio
        assert isinstance(result, Portfolio)

    def test_portfolio_assets_match(self, engine: BasanosEngine) -> None:
        """portfolio.assets must match the engine asset list."""
        assert engine.portfolio.assets == engine.assets


# ─── 2. Regression tests ─────────────────────────────────────────────────────


class TestRegression:
    """Numerical regression tests against known-good reference values.

    Any change to the optimizer pipeline that alters numerical output will be
    caught here.  The reference values were generated from:
        seed=0, n=80 rows, 3 assets (A, B, C),
        BasanosConfig(vola=16, corr=32, clip=3.0, shrink=0.5, aum=1e6).
    """

    _ATOL = 1e-6  # absolute tolerance for floating-point comparisons

    def test_cash_position_reference_rows(self, regression_engine: BasanosEngine) -> None:
        """cash_position values at selected rows must match stored references."""
        cp = regression_engine.cash_position
        for row_idx, expected in _REGRESSION_ROWS.items():
            for asset in _REGRESSION_ASSETS:
                actual = cp[asset][row_idx]
                assert actual is not None, f"cash_position[{asset}][{row_idx}] is None"
                assert not math.isnan(actual), f"cash_position[{asset}][{row_idx}] is NaN"
                assert abs(actual - expected[asset]) < self._ATOL, (
                    f"cash_position[{asset}][{row_idx}]: expected {expected[asset]:.10f}, got {actual:.10f}"
                )

    def test_position_leverage_reference_rows(self, regression_engine: BasanosEngine) -> None:
        """position_leverage values at selected rows must match stored references."""
        lev = regression_engine.position_leverage
        for row_idx, expected in _REGRESSION_ROWS.items():
            actual = lev["leverage"][row_idx]
            assert actual is not None, f"leverage[{row_idx}] is None"
            assert not math.isnan(actual), f"leverage[{row_idx}] is NaN"
            assert abs(actual - expected["leverage"]) < self._ATOL, (
                f"leverage[{row_idx}]: expected {expected['leverage']:.10f}, got {actual:.10f}"
            )

    def test_cash_position_overall_shape(self, regression_engine: BasanosEngine) -> None:
        """cash_position must have (n_rows, n_assets + 1) shape."""
        cp = regression_engine.cash_position
        assert cp.shape == (_REGRESSION_N, len(_REGRESSION_ASSETS) + 1)

    def test_position_leverage_overall_shape(self, regression_engine: BasanosEngine) -> None:
        """position_leverage must have (n_rows, 2) shape."""
        lev = regression_engine.position_leverage
        assert lev.shape == (_REGRESSION_N, 2)


# ─── 3. Realistic-scale test ─────────────────────────────────────────────────


@pytest.mark.slow
def test_realistic_scale_completes_within_time_bound() -> None:
    """252-row x 20-asset engine must complete the full pipeline within 60 s.

    This is an optional stress test that exercises the optimizer at a scale
    representative of one year of daily data across a medium-sized universe.
    """
    engine = _make_engine(n=252, n_assets=20)

    start = time.monotonic()

    # Exercise the full pipeline end-to-end
    _ = engine.cash_position
    _ = engine.position_leverage
    _ = engine.portfolio

    elapsed = time.monotonic() - start

    assert elapsed < 60.0, f"Realistic-scale pipeline took {elapsed:.1f}s (limit: 60s)"

    # Sanity-check shapes
    assert engine.cash_position.shape == (252, 21)  # date + 20 assets
    assert engine.position_leverage.shape == (252, 2)
