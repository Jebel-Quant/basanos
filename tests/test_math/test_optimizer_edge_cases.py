"""Explicit edge-case tests for basanos.math.optimizer (BasanosEngine).

Covers four targeted edge cases that may not be fully exercised by the broad
random input space of the property tests:

1. All-zero ``mu`` vector — short-circuit path at ``optimizer.py:661``
   (``np.allclose(expected_mu, 0.0)`` branch).
2. Single-asset universe (``N=1``) — 1×1 correlation matrix; condition number
   is trivially 1.0 and effective rank is always 1.
3. ``T=1`` timestep — no EWM warm-up; EWMA volatility and all positions must
   be NaN/null.
4. Single non-null asset per row — alternating availability forces the
   per-timestamp mask to select a 1×1 sub-matrix.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import pytest

from basanos.math import BasanosConfig, BasanosEngine

# ─── Shared helpers ───────────────────────────────────────────────────────────


def _non_monotonic_prices(n_rows: int, n_assets: int, *, seed: int = 42) -> pl.DataFrame:
    """Return a DataFrame of non-monotonic positive prices.

    Uses alternating percentage changes so the price series is guaranteed to
    change sign from one step to the next, satisfying the monotonicity guard in
    ``BasanosEngine.__post_init__``.

    Args:
        n_rows: Number of timestep rows.
        n_assets: Number of asset columns (named ``"A"``, ``"B"``, …).
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    cols: dict[str, object] = {"date": list(range(n_rows))}
    for i in range(n_assets):
        pct = rng.uniform(0.01, 0.05, size=max(n_rows - 1, 0))
        signs = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(max(n_rows - 1, 0))])
        prices = np.empty(n_rows)
        prices[0] = 100.0 + i * 50.0
        for j in range(1, n_rows):
            prices[j] = prices[j - 1] * (1.0 + signs[j - 1] * pct[j - 1])
        cols[chr(ord("A") + i)] = pl.Series(prices.tolist(), dtype=pl.Float64)
    return pl.DataFrame(cols)


def _zero_mu(prices: pl.DataFrame) -> pl.DataFrame:
    """Return a ``mu`` DataFrame aligned with *prices* containing all zeros."""
    assets = [c for c in prices.columns if c != "date"]
    return pl.DataFrame(
        {
            "date": prices["date"],
            **{a: pl.Series([0.0] * prices.height, dtype=pl.Float64) for a in assets},
        }
    )


def _sinusoidal_mu(prices: pl.DataFrame, *, seed: int = 42) -> pl.DataFrame:
    """Return a ``mu`` DataFrame aligned with *prices* using bounded sinusoidal values."""
    rng = np.random.default_rng(seed)
    assets = [c for c in prices.columns if c != "date"]
    return pl.DataFrame(
        {
            "date": prices["date"],
            **{
                a: pl.Series(np.tanh(rng.normal(0.0, 0.5, size=prices.height)).tolist(), dtype=pl.Float64)
                for a in assets
            },
        }
    )


# ─── Edge case 1: All-zero mu vector ─────────────────────────────────────────


@pytest.mark.parametrize("n_assets", [1, 2])
def test_zero_mu_positions_are_zero_after_warmup(n_assets: int) -> None:
    """After EWM warm-up, all-zero mu must yield zero cash positions.

    Verifies the short-circuit branch at ``optimizer.py:661`` where
    ``np.allclose(expected_mu, 0.0)`` causes the optimizer to assign a zero
    position vector directly, skipping the matrix solve.
    """
    cfg = BasanosConfig(vola=5, corr=10, clip=2.0, shrink=0.5, aum=1e6)
    n_rows = 40
    prices = _non_monotonic_prices(n_rows, n_assets)
    mu = _zero_mu(prices)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)

    warmup = cfg.corr
    cp = engine.cash_position
    for asset in engine.assets:
        tail = cp[asset].slice(warmup).drop_nulls().to_numpy()
        assert np.allclose(tail, 0.0), f"Non-zero cash position for asset {asset!r} with zero mu"


@pytest.mark.parametrize("n_assets", [1, 2])
def test_zero_mu_emits_no_degenerate_denominator_warning(n_assets: int, caplog: pytest.LogCaptureFixture) -> None:
    """Zero mu must not trigger the degenerate-denominator warning.

    When ``expected_mu`` is identically zero the optimizer short-circuits
    before calling ``inv_a_norm``, so the degenerate-denominator guard is never
    reached.
    """
    cfg = BasanosConfig(vola=5, corr=10, clip=2.0, shrink=0.5, aum=1e6)
    prices = _non_monotonic_prices(40, n_assets)
    mu = _zero_mu(prices)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)

    with caplog.at_level(logging.WARNING, logger="basanos.math.optimizer"):
        _ = engine.cash_position

    degen = [r for r in caplog.records if "normalisation denominator is degenerate" in r.message]
    assert not degen, f"Unexpected degenerate-denominator warnings with zero mu: {degen}"


@pytest.mark.parametrize("n_assets", [1, 2])
def test_zero_mu_solver_residual_is_zero(n_assets: int) -> None:
    """Solver residual must be exactly 0.0 everywhere when mu is all zeros.

    The zero-mu short-circuit stores a zero-position vector; the corresponding
    residual ``‖C x − μ‖₂ = ‖0 − 0‖₂ = 0`` should be returned by
    ``solver_residual`` for every non-NaN row.
    """
    cfg = BasanosConfig(vola=5, corr=10, clip=2.0, shrink=0.5, aum=1e6)
    prices = _non_monotonic_prices(40, n_assets)
    mu = _zero_mu(prices)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)

    residuals = engine.solver_residual["residual"].drop_nulls().to_numpy()
    np.testing.assert_array_equal(residuals, 0.0)


@pytest.mark.parametrize("n_assets", [1, 2])
def test_zero_mu_position_leverage_is_zero_after_warmup(n_assets: int) -> None:
    """With all-zero mu, position leverage must be zero after the EWM warm-up."""
    cfg = BasanosConfig(vola=5, corr=10, clip=2.0, shrink=0.5, aum=1e6)
    prices = _non_monotonic_prices(40, n_assets)
    mu = _zero_mu(prices)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)

    warmup = cfg.corr
    leverage = engine.position_leverage["leverage"].slice(warmup).drop_nulls().to_numpy()
    assert np.allclose(leverage, 0.0), f"Non-zero leverage with zero mu: {leverage}"


# ─── Edge case 2: Single-asset universe (N=1) ─────────────────────────────────


class TestSingleAsset:
    """BasanosEngine with exactly one asset (N=1)."""

    @pytest.fixture(scope="class")
    def engine(self) -> BasanosEngine:
        """Build a 30-row single-asset engine."""
        cfg = BasanosConfig(vola=3, corr=5, clip=2.0, shrink=0.5, aum=1e6)
        prices = _non_monotonic_prices(30, n_assets=1)
        mu = _sinusoidal_mu(prices)
        return BasanosEngine(prices=prices, mu=mu, cfg=cfg)

    def test_assets_has_one_element(self, engine: BasanosEngine) -> None:
        """engine.assets must contain exactly one element for N=1."""
        assert len(engine.assets) == 1

    def test_cash_position_shape(self, engine: BasanosEngine) -> None:
        """cash_position must have shape (T, 2) — date plus one asset column."""
        cp = engine.cash_position
        assert cp.shape == (engine.prices.height, 2), f"Unexpected shape: {cp.shape}"

    def test_condition_number_is_one_after_warmup(self, engine: BasanosEngine) -> None:
        """For N=1 the correlation matrix is the scalar 1.0; κ must equal 1.0."""
        warmup = engine.cfg.corr
        kappas = engine.condition_number["condition_number"].slice(warmup).drop_nulls().to_numpy()
        assert len(kappas) > 0, "No finite condition-number values found after warmup"
        np.testing.assert_allclose(kappas, 1.0, atol=1e-9)

    def test_effective_rank_is_one_after_warmup(self, engine: BasanosEngine) -> None:
        """For N=1 the effective rank of a 1×1 matrix is always 1.0."""
        warmup = engine.cfg.corr
        ranks = engine.effective_rank["effective_rank"].slice(warmup).drop_nulls().to_numpy()
        assert len(ranks) > 0, "No finite effective-rank values found after warmup"
        np.testing.assert_allclose(ranks, 1.0, atol=1e-9)

    def test_position_leverage_is_non_negative(self, engine: BasanosEngine) -> None:
        """Leverage (L1 norm) must remain non-negative for the N=1 case."""
        leverage = engine.position_leverage["leverage"].drop_nulls().to_numpy()
        assert np.all(leverage >= 0.0), f"Negative leverage for N=1: {leverage[leverage < 0]}"

    def test_cor_tensor_shape(self, engine: BasanosEngine) -> None:
        """cor_tensor for N=1 must have shape (T, 1, 1)."""
        tensor = engine.cor_tensor
        assert tensor.shape == (engine.prices.height, 1, 1), f"Unexpected tensor shape: {tensor.shape}"

    def test_cor_diagonal_is_one_after_warmup(self, engine: BasanosEngine) -> None:
        """For N=1 the single correlation value is always 1.0 after warm-up."""
        warmup = engine.cfg.corr
        tensor = engine.cor_tensor[warmup:]  # (T-warmup, 1, 1)
        finite_mask = np.isfinite(tensor[:, 0, 0])
        np.testing.assert_allclose(tensor[finite_mask, 0, 0], 1.0, atol=1e-9)


# ─── Edge case 3: T=1 timestep ────────────────────────────────────────────────


class TestSingleTimestep:
    """BasanosEngine with exactly one timestep (T=1).

    With a single row there are no percentage returns to feed into EWMA, so:
    - EWMA volatility cannot be computed → all null.
    - EWM correlation has zero observations → all NaN (below min_periods).
    - cash_position uses NaN vola as denominator → all NaN.
    """

    @pytest.fixture(scope="class")
    def engine(self) -> BasanosEngine:
        """Minimal 1-row, 1-asset engine."""
        prices = pl.DataFrame({"date": [0], "A": pl.Series([100.0], dtype=pl.Float64)})
        mu = pl.DataFrame({"date": [0], "A": pl.Series([0.5], dtype=pl.Float64)})
        cfg = BasanosConfig(vola=3, corr=5, clip=2.0, shrink=0.5, aum=1e6)
        return BasanosEngine(prices=prices, mu=mu, cfg=cfg)

    def test_vola_is_all_null(self, engine: BasanosEngine) -> None:
        """With T=1, EWMA volatility requires at least cfg.vola samples; output is null."""
        for asset in engine.assets:
            non_null = engine.vola[asset].drop_nulls()
            assert non_null.len() == 0, f"Unexpected non-null vola for T=1 in {asset!r}: {non_null}"

    def test_cash_position_is_all_nan(self, engine: BasanosEngine) -> None:
        """With T=1, no EWM warm-up means cash positions are NaN for every asset."""
        cp = engine.cash_position
        for asset in engine.assets:
            vals = cp[asset].to_numpy()
            assert np.all(np.isnan(vals)), f"Expected all-NaN cash position for T=1, got {vals}"

    def test_cash_position_has_correct_shape(self, engine: BasanosEngine) -> None:
        """cash_position must still return shape (1, n_assets+1) for T=1."""
        cp = engine.cash_position
        expected_cols = 1 + len(engine.assets)  # date + assets
        assert cp.shape == (1, expected_cols), f"Unexpected shape for T=1: {cp.shape}"

    def test_ret_adj_shape_matches_prices(self, engine: BasanosEngine) -> None:
        """ret_adj must have the same shape as prices even for T=1."""
        assert engine.ret_adj.shape == engine.prices.shape

    def test_cor_tensor_has_single_nan_matrix(self, engine: BasanosEngine) -> None:
        """The single correlation matrix for T=1 must be NaN (below min_periods)."""
        tensor = engine.cor_tensor  # (1, N, N)
        assert tensor.shape[0] == 1, f"Expected 1 matrix, got {tensor.shape[0]}"
        assert not np.any(np.isfinite(tensor)), f"Expected all-NaN cor tensor for T=1, got {tensor}"


# ─── Edge case 4: Single non-null asset per row ───────────────────────────────


class TestSingleNonNullAssetPerRow:
    """When alternating rows expose exactly one asset at a time.

    Each row has only one finite price, so the per-timestamp ``mask`` selects a
    1×1 sub-matrix (scalar) from the full correlation matrix.  The optimizer
    must handle this gracefully, returning finite positions for available assets
    and NaN for unavailable ones.
    """

    @pytest.fixture(scope="class")
    def engine(self) -> BasanosEngine:
        """60-row engine where asset A is null in odd rows, B in even rows."""
        n = 60
        rng = np.random.default_rng(7)
        cfg = BasanosConfig(vola=3, corr=5, clip=2.0, shrink=0.5, aum=1e6)

        # Build full non-monotonic price walks, then apply alternating mask.
        def _walk(start: float, pct: np.ndarray, signs: np.ndarray) -> np.ndarray:
            p = np.empty(n)
            p[0] = start
            for j in range(1, n):
                p[j] = p[j - 1] * (1.0 + signs[j - 1] * pct[j - 1])
            return p

        signs = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(n - 1)])
        prices_a = _walk(100.0, rng.uniform(0.01, 0.05, n - 1), signs)
        prices_b = _walk(150.0, rng.uniform(0.01, 0.05, n - 1), signs)

        # Even rows: only A is non-null; odd rows: only B is non-null.
        a_masked = [float(v) if i % 2 == 0 else None for i, v in enumerate(prices_a.tolist())]
        b_masked = [float(v) if i % 2 == 1 else None for i, v in enumerate(prices_b.tolist())]

        prices = pl.DataFrame(
            {
                "date": list(range(n)),
                "A": pl.Series(a_masked, dtype=pl.Float64),
                "B": pl.Series(b_masked, dtype=pl.Float64),
            }
        )
        dummy_prices = pl.DataFrame(
            {
                "date": list(range(n)),
                "A": pl.Series([100.0] * n, dtype=pl.Float64),
                "B": pl.Series([100.0] * n, dtype=pl.Float64),
            }
        )
        mu = _sinusoidal_mu(dummy_prices)
        return BasanosEngine(prices=prices, mu=mu, cfg=cfg)

    def test_cash_position_shape(self, engine: BasanosEngine) -> None:
        """cash_position must have shape (n_rows, 3) for this 2-asset frame."""
        cp = engine.cash_position
        assert cp.shape == (engine.prices.height, 3), f"Unexpected shape: {cp.shape}"

    def test_position_leverage_is_non_negative(self, engine: BasanosEngine) -> None:
        """Leverage must be non-negative with alternating single-asset availability."""
        leverage = engine.position_leverage["leverage"].drop_nulls().to_numpy()
        assert np.all(leverage >= 0.0), f"Negative leverage with staggered prices: {leverage[leverage < 0]}"

    def test_asset_a_null_in_odd_rows(self, engine: BasanosEngine) -> None:
        """Asset A is unavailable in odd rows; cash positions there must be NaN."""
        cp = engine.cash_position
        vals_a = cp["A"].to_numpy()
        odd_indices = [i for i in range(engine.prices.height) if i % 2 == 1]
        for i in odd_indices:
            assert np.isnan(vals_a[i]), f"Expected NaN for asset A at odd row {i}, got {vals_a[i]}"

    def test_asset_b_null_in_even_rows(self, engine: BasanosEngine) -> None:
        """Asset B is unavailable in even rows; cash positions there must be NaN."""
        cp = engine.cash_position
        vals_b = cp["B"].to_numpy()
        even_indices = [i for i in range(engine.prices.height) if i % 2 == 0]
        for i in even_indices:
            assert np.isnan(vals_b[i]), f"Expected NaN for asset B at even row {i}, got {vals_b[i]}"
