"""Tests for basanos.math._factor_model.FactorModel and engine integration.

Covers:
- Construction: defaults, fully-specified, all validation error paths.
- to_matrix: diagonal = 1 with default specific variances, PSD, custom params.
- solve: Woodbury result matches np.linalg.solve on the full matrix.
- Engine integration: cash_position shape, factor_model dimension mismatch,
  LargeUniverseWarning emission.
- Property-based tests (Hypothesis): solve matches full matrix for random
  well-conditioned inputs.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st

from basanos.exceptions import (
    FactorModelDimensionError,
    LargeUniverseWarning,
)
from basanos.math import BasanosConfig, BasanosEngine, FactorModel

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_engine(n: int = 2, t: int = 120) -> tuple[pl.DataFrame, pl.DataFrame, BasanosConfig]:
    """Return (prices, mu, cfg) for a minimal valid engine with *n* assets and *t* rows."""
    rng = np.random.default_rng(42)
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=t - 1), interval="1d", eager=True)
    asset_names = [chr(ord("A") + i) for i in range(n)]
    prices_data: dict = {"date": dates}
    mu_data: dict = {"date": dates}
    for name in asset_names:
        prices_data[name] = pl.Series(100.0 + np.cumsum(rng.normal(0.0, 0.5, t)), dtype=pl.Float64)
        mu_data[name] = pl.Series(np.tanh(rng.normal(0.0, 0.5, t)), dtype=pl.Float64)
    prices = pl.DataFrame(prices_data)
    mu = pl.DataFrame(mu_data)
    cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
    return prices, mu, cfg


def _random_psd_matrix(k: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random K×K symmetric positive-definite matrix."""
    A = rng.normal(0, 1, (k, k))  # noqa: N806
    return A @ A.T + np.eye(k) * 0.5


# ─── FactorModel construction ─────────────────────────────────────────────────


def test_construction_defaults() -> None:
    """FactorModel with only loadings computes default F and D."""
    B = np.array([[0.8, 0.2], [0.6, 0.5], [0.3, 0.9]])  # noqa: N806
    fm = FactorModel(loadings=B)
    assert fm.n_assets == 3
    assert fm.n_factors == 2
    # Default F is identity
    np.testing.assert_array_equal(fm._resolved_factor_covariance(), np.eye(2))
    # Default D ensures diagonal(Σ) = 1
    diag = np.diag(fm.to_matrix())
    np.testing.assert_allclose(diag, 1.0, atol=1e-12)


def test_construction_full_params() -> None:
    """FactorModel with all parameters explicitly provided."""
    rng = np.random.default_rng(0)
    n, k = 4, 2
    B = rng.normal(0, 0.5, (n, k))  # noqa: N806
    F = _random_psd_matrix(k, rng)  # noqa: N806
    d = np.ones(n) * 0.5
    fm = FactorModel(loadings=B, factor_covariance=F, specific_variances=d)
    assert fm.n_assets == n
    assert fm.n_factors == k
    np.testing.assert_array_equal(fm._resolved_factor_covariance(), F)
    np.testing.assert_array_equal(fm._resolved_specific_variances(), d)


def test_n_assets_and_n_factors() -> None:
    """n_assets and n_factors properties reflect loadings shape."""
    B = np.zeros((7, 3))  # noqa: N806
    fm = FactorModel(loadings=B)
    assert fm.n_assets == 7
    assert fm.n_factors == 3


# ─── FactorModel validation errors ───────────────────────────────────────────


def test_invalid_1d_loadings() -> None:
    """1-D loadings array must raise FactorModelDimensionError."""
    with pytest.raises(FactorModelDimensionError, match="2-D"):
        FactorModel(loadings=np.array([1.0, 2.0, 3.0]))


def test_invalid_factor_cov_nonsquare() -> None:
    """Non-square factor_covariance must raise FactorModelDimensionError."""
    B = np.ones((3, 2))  # noqa: N806
    with pytest.raises(FactorModelDimensionError, match="square"):
        FactorModel(loadings=B, factor_covariance=np.ones((2, 3)))


def test_invalid_factor_cov_wrong_k() -> None:
    """factor_covariance (K, K) must match K columns in loadings."""
    B = np.ones((3, 2))  # noqa: N806
    F_wrong = np.eye(3)  # K=3, but loadings has K=2  # noqa: N806
    with pytest.raises(FactorModelDimensionError, match="incompatible"):
        FactorModel(loadings=B, factor_covariance=F_wrong)


def test_invalid_specific_variances_wrong_length() -> None:
    """specific_variances with wrong length must raise FactorModelDimensionError."""
    B = np.ones((3, 2))  # noqa: N806
    with pytest.raises(FactorModelDimensionError, match="length"):
        FactorModel(loadings=B, specific_variances=np.ones(4))


def test_invalid_specific_variances_2d() -> None:
    """2-D specific_variances must raise FactorModelDimensionError."""
    B = np.ones((3, 2))  # noqa: N806
    with pytest.raises(FactorModelDimensionError, match="1-D"):
        FactorModel(loadings=B, specific_variances=np.ones((3, 1)))


def test_invalid_factor_cov_asymmetric() -> None:
    """Asymmetric factor_covariance must raise FactorModelDimensionError."""
    B = np.ones((3, 2))  # noqa: N806
    F_asymmetric = np.array([[1.0, 0.5], [0.0, 1.0]])  # upper != lower triangle  # noqa: N806
    with pytest.raises(FactorModelDimensionError, match="symmetric"):
        FactorModel(loadings=B, factor_covariance=F_asymmetric)


def test_invalid_specific_variances_nonpositive() -> None:
    """Zero or negative specific_variances must raise FactorModelDimensionError."""
    B = np.ones((3, 2))  # noqa: N806
    with pytest.raises(FactorModelDimensionError, match="strictly positive"):
        FactorModel(loadings=B, specific_variances=np.array([0.5, 0.0, 0.5]))

    with pytest.raises(FactorModelDimensionError, match="strictly positive"):
        FactorModel(loadings=B, specific_variances=np.array([0.5, -0.1, 0.5]))


# ─── to_matrix ────────────────────────────────────────────────────────────────


def test_to_matrix_diagonal_is_one_with_defaults() -> None:
    """Default specific_variances ensure diag(Σ) = 1."""
    rng = np.random.default_rng(1)
    B = rng.normal(0, 0.4, (5, 3))  # noqa: N806
    fm = FactorModel(loadings=B)
    M = fm.to_matrix()  # noqa: N806
    np.testing.assert_allclose(np.diag(M), 1.0, atol=1e-12)


def test_to_matrix_is_symmetric() -> None:
    """Materialised matrix must be symmetric."""
    rng = np.random.default_rng(2)
    B = rng.normal(0, 0.5, (4, 2))  # noqa: N806
    fm = FactorModel(loadings=B)
    M = fm.to_matrix()  # noqa: N806
    np.testing.assert_allclose(M, M.T, atol=1e-14)


def test_to_matrix_is_psd() -> None:
    """Materialised matrix must be positive semi-definite (all eigenvalues ≥ 0)."""
    rng = np.random.default_rng(3)
    B = rng.normal(0, 0.5, (6, 3))  # noqa: N806
    fm = FactorModel(loadings=B)
    eigvals = np.linalg.eigvalsh(fm.to_matrix())
    assert np.all(eigvals >= -1e-10), f"Negative eigenvalues: {eigvals}"


def test_to_matrix_custom_params() -> None:
    """to_matrix with explicit F and D matches the manual formula."""
    B = np.array([[1.0, 0.0], [0.0, 1.0]])  # noqa: N806
    sv = np.array([0.5, 0.5])
    fm = FactorModel(loadings=B, specific_variances=sv)
    expected = np.array([[1.5, 0.0], [0.0, 1.5]])
    np.testing.assert_allclose(fm.to_matrix(), expected, atol=1e-14)


def test_to_matrix_identity_loadings_custom_f() -> None:
    """With identity loadings and custom F, Σ = F + D."""
    n, k = 3, 3
    B = np.eye(n, k)  # noqa: N806
    F = np.array([[2.0, 0.5, 0.0], [0.5, 1.5, 0.3], [0.0, 0.3, 1.0]])  # noqa: N806
    d = np.array([0.1, 0.2, 0.3])
    fm = FactorModel(loadings=B, factor_covariance=F, specific_variances=d)
    expected = F + np.diag(d)
    np.testing.assert_allclose(fm.to_matrix(), expected, atol=1e-14)


# ─── solve ────────────────────────────────────────────────────────────────────


def test_solve_matches_full_matrix_basic() -> None:
    """Woodbury solve must match np.linalg.solve on the full matrix."""
    B = np.array([[0.8, 0.2], [0.6, 0.5], [0.3, 0.9]])  # noqa: N806
    fm = FactorModel(loadings=B)
    rhs = np.array([1.0, 0.0, -1.0])
    x_woodbury = fm.solve(rhs)
    x_full = np.linalg.solve(fm.to_matrix(), rhs)
    np.testing.assert_allclose(x_woodbury, x_full, rtol=1e-8)


def test_solve_with_custom_factor_cov() -> None:
    """Woodbury solve with custom F matches full-matrix solve."""
    rng = np.random.default_rng(7)
    n, k = 5, 2
    B = rng.normal(0, 0.5, (n, k))  # noqa: N806
    F = _random_psd_matrix(k, rng)  # noqa: N806
    d = rng.uniform(0.1, 0.9, n)
    fm = FactorModel(loadings=B, factor_covariance=F, specific_variances=d)
    rhs = rng.normal(0, 1, n)
    np.testing.assert_allclose(fm.solve(rhs), np.linalg.solve(fm.to_matrix(), rhs), rtol=1e-8)


def test_solve_zero_rhs() -> None:
    """Solving Σ·x = 0 must return the zero vector."""
    B = np.array([[0.8, 0.2], [0.3, 0.7]])  # noqa: N806
    fm = FactorModel(loadings=B)
    x = fm.solve(np.zeros(2))
    np.testing.assert_allclose(x, np.zeros(2), atol=1e-14)


def test_solve_dimension_mismatch() -> None:
    """Passing rhs with wrong length must raise DimensionMismatchError."""
    from basanos.exceptions import DimensionMismatchError

    B = np.array([[0.8, 0.2], [0.3, 0.7]])  # noqa: N806
    fm = FactorModel(loadings=B)
    with pytest.raises(DimensionMismatchError):
        fm.solve(np.ones(5))


def test_solve_single_factor() -> None:
    """Single-factor model (K=1) Woodbury solve matches full-matrix solve."""
    rng = np.random.default_rng(9)
    n = 6
    B = rng.normal(0, 0.5, (n, 1))  # noqa: N806
    d = rng.uniform(0.1, 0.5, n)
    fm = FactorModel(loadings=B, specific_variances=d)
    rhs = rng.normal(0, 1, n)
    np.testing.assert_allclose(fm.solve(rhs), np.linalg.solve(fm.to_matrix(), rhs), rtol=1e-8)


def test_solve_lu_fallback_f() -> None:
    """When cho_factor raises for F, the LU fallback produces the correct result."""
    import unittest.mock as mock

    rng = np.random.default_rng(11)
    n, k = 4, 2
    B = rng.normal(0, 0.5, (n, k))  # noqa: N806
    d = rng.uniform(0.1, 0.9, n)
    fm = FactorModel(loadings=B, specific_variances=d)
    rhs = rng.normal(0, 1, n)
    expected = np.linalg.solve(fm.to_matrix(), rhs)

    # Force every cho_factor call to fail — exercises the np.linalg.solve fallback for F
    with mock.patch("basanos.math._factor_model.cho_factor", side_effect=np.linalg.LinAlgError):
        x = fm.solve(rhs)

    np.testing.assert_allclose(x, expected, rtol=1e-8)


def test_solve_lu_fallback_m() -> None:
    """When cho_factor succeeds for F but raises for M, the LU fallback produces the correct result."""
    import unittest.mock as mock

    from scipy.linalg import cho_factor as real_cho_factor

    rng = np.random.default_rng(12)
    n, k = 4, 2
    B = rng.normal(0, 0.5, (n, k))  # noqa: N806
    d = rng.uniform(0.1, 0.9, n)
    fm = FactorModel(loadings=B, specific_variances=d)
    rhs = rng.normal(0, 1, n)
    expected = np.linalg.solve(fm.to_matrix(), rhs)

    # First call (for F) succeeds; second call (for M) fails
    call_count = {"n": 0}

    def patched_cho_factor(a, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise np.linalg.LinAlgError("forced failure for M")  # noqa: TRY003
        return real_cho_factor(a, *args, **kwargs)

    with mock.patch("basanos.math._factor_model.cho_factor", side_effect=patched_cho_factor):
        x = fm.solve(rhs)

    np.testing.assert_allclose(x, expected, rtol=1e-8)


# ─── Engine integration ───────────────────────────────────────────────────────


def test_engine_with_factor_model_cash_position_shape() -> None:
    """Engine with factor_model produces cash_position with correct shape."""
    n, t = 3, 80
    prices, mu, cfg = _make_engine(n=n, t=t)
    rng = np.random.default_rng(0)
    B = rng.normal(0, 0.5, (n, 2))  # noqa: N806
    fm = FactorModel(loadings=B)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg, factor_model=fm)
    cp = engine.cash_position
    assert cp.shape == (t, n + 1)  # date + n asset columns
    assert cp.columns[0] == "date"


def test_engine_with_factor_model_produces_finite_positions() -> None:
    """Engine with factor_model produces some finite positions for valid data."""
    n, t = 2, 80
    prices, mu, cfg = _make_engine(n=n, t=t)
    rng = np.random.default_rng(1)
    B = rng.normal(0, 0.5, (n, 2))  # noqa: N806
    fm = FactorModel(loadings=B)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg, factor_model=fm)
    cp = engine.cash_position
    # After warm-up (vola=10) there should be finite positions
    asset_cols = [c for c in cp.columns if c != "date"]
    cp_np = cp.select(asset_cols).to_numpy()
    assert np.any(np.isfinite(cp_np[cfg.vola :]))


def test_engine_without_factor_model_unchanged() -> None:
    """Engine without factor_model uses EWMA path (unchanged behaviour)."""
    prices, mu, cfg = _make_engine(n=2, t=60)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
    cp = engine.cash_position
    assert cp.shape == (60, 3)


def test_engine_factor_model_dimension_mismatch_raises() -> None:
    """Providing a factor_model with wrong n_assets must raise FactorModelDimensionError."""
    prices, mu, cfg = _make_engine(n=3, t=60)
    B_wrong = np.ones((5, 2))  # 5 assets but engine has 3  # noqa: N806
    fm = FactorModel(loadings=B_wrong)
    with pytest.raises(FactorModelDimensionError, match="n_assets"):
        BasanosEngine(prices=prices, mu=mu, cfg=cfg, factor_model=fm)


def test_engine_factor_model_portfolio_accessible() -> None:
    """Engine with factor_model supports .portfolio property."""
    n, t = 2, 80
    prices, mu, cfg = _make_engine(n=n, t=t)
    rng = np.random.default_rng(2)
    B = rng.normal(0, 0.5, (n, 2))  # noqa: N806
    fm = FactorModel(loadings=B)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg, factor_model=fm)
    portfolio = engine.portfolio
    assert portfolio is not None


# ─── LargeUniverseWarning ─────────────────────────────────────────────────────


def test_large_universe_warning_emitted() -> None:
    """LargeUniverseWarning is emitted when peak RAM would exceed ~4 GB without factor_model.

    Peak bytes = 112 * T * N². For N=200, T=1000: 112 * 1000 * 40000 = 4.48 GB > 4 GB.
    """
    # Build a prices/mu frame large enough to trigger the warning at N=200, T=1000.
    # We won't actually run the EWMA — we just need __post_init__ to emit the warning.
    n, t = 200, 1000
    rng = np.random.default_rng(42)
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=t - 1), interval="1d", eager=True)
    asset_names = [f"A{i}" for i in range(n)]
    prices_data: dict = {"date": dates}
    mu_data: dict = {"date": dates}
    for name in asset_names:
        raw = 100.0 + np.cumsum(rng.normal(0.0, 0.5, t))
        prices_data[name] = pl.Series(raw.tolist(), dtype=pl.Float64)
        mu_data[name] = pl.Series(rng.normal(0.0, 0.1, t).tolist(), dtype=pl.Float64)
    prices = pl.DataFrame(prices_data)
    mu = pl.DataFrame(mu_data)
    cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BasanosEngine(prices=prices, mu=mu, cfg=cfg)
        large_universe_warnings = [x for x in w if issubclass(x.category, LargeUniverseWarning)]
        assert len(large_universe_warnings) == 1
        assert "FactorModel" in str(large_universe_warnings[0].message)


def test_large_universe_warning_suppressed_with_factor_model() -> None:
    """LargeUniverseWarning is NOT emitted when factor_model is provided."""
    n, t = 200, 1000
    rng = np.random.default_rng(42)
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=t - 1), interval="1d", eager=True)
    asset_names = [f"A{i}" for i in range(n)]
    prices_data: dict = {"date": dates}
    mu_data: dict = {"date": dates}
    for name in asset_names:
        raw = 100.0 + np.cumsum(rng.normal(0.0, 0.5, t))
        prices_data[name] = pl.Series(raw.tolist(), dtype=pl.Float64)
        mu_data[name] = pl.Series(rng.normal(0.0, 0.1, t).tolist(), dtype=pl.Float64)
    prices = pl.DataFrame(prices_data)
    mu = pl.DataFrame(mu_data)
    cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
    B = rng.normal(0, 0.3, (n, 5))  # noqa: N806
    fm = FactorModel(loadings=B)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BasanosEngine(prices=prices, mu=mu, cfg=cfg, factor_model=fm)
        large_universe_warnings = [x for x in w if issubclass(x.category, LargeUniverseWarning)]
        assert len(large_universe_warnings) == 0


def test_no_warning_for_small_universe() -> None:
    """No LargeUniverseWarning for a small universe (N=2, T=120)."""
    prices, mu, cfg = _make_engine(n=2, t=120)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BasanosEngine(prices=prices, mu=mu, cfg=cfg)
        large_universe_warnings = [x for x in w if issubclass(x.category, LargeUniverseWarning)]
        assert len(large_universe_warnings) == 0


# ─── Property-based tests ─────────────────────────────────────────────────────


@st.composite
def _factor_model_strategy(draw: st.DrawFn) -> FactorModel:
    """Hypothesis strategy for a random well-conditioned FactorModel."""
    n = draw(st.integers(min_value=2, max_value=8))
    k = draw(st.integers(min_value=1, max_value=min(n, 4)))
    B = draw(  # noqa: N806
        np_st.arrays(
            dtype=np.float64,
            shape=(n, k),
            elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    # Use explicit specific variances (uniform positive) to ensure PD matrix
    d = draw(
        np_st.arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
        )
    )
    return FactorModel(loadings=B, specific_variances=d)


@given(fm=_factor_model_strategy())
@settings(max_examples=100, deadline=None)
def test_solve_matches_full_matrix_property(fm: FactorModel) -> None:
    """Woodbury solve matches np.linalg.solve on the full matrix for any valid FactorModel."""
    rng = np.random.default_rng(0)
    rhs = rng.normal(0, 1, fm.n_assets)
    M = fm.to_matrix()  # noqa: N806
    # Only test when the matrix is well-conditioned
    if np.linalg.cond(M) > 1e10:
        return
    x_woodbury = fm.solve(rhs)
    x_full = np.linalg.solve(M, rhs)
    np.testing.assert_allclose(x_woodbury, x_full, rtol=1e-6, atol=1e-10)


@given(fm=_factor_model_strategy())
@settings(max_examples=100, deadline=None)
def test_to_matrix_is_symmetric_property(fm: FactorModel) -> None:
    """to_matrix always produces a symmetric matrix."""
    M = fm.to_matrix()  # noqa: N806
    np.testing.assert_allclose(M, M.T, atol=1e-14)


@given(
    n=st.integers(min_value=2, max_value=8),
    k=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=60, deadline=None)
def test_to_matrix_diagonal_property(n: int, k: int, seed: int) -> None:
    """Default specific_variances ensure diag(Σ) = 1 where (B·F·B^T)_{ii} ≤ 1.

    When the factor contribution (B·F·B^T)_{ii} exceeds 1, the specific-variance
    floor (1e-12) is applied, so the diagonal will be (B·F·B^T)_{ii} + 1e-12 > 1.
    The guarantee is that the diagonal is always ≥ 1 - 1e-12 (i.e., never below 1
    unless the floor dominates), and exactly 1 when the factor contribution ≤ 1.
    """
    k = min(k, n)
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 0.4, (n, k))  # noqa: N806
    fm = FactorModel(loadings=B)
    M = fm.to_matrix()  # noqa: N806
    bfbt_diag = np.einsum("ik,kl,il->i", B, np.eye(k), B)
    # Where factor contribution ≤ 1: diag(Σ) must equal 1 (specific variance = 1 - bfbt_diag)
    unclamped = bfbt_diag <= 1.0
    if unclamped.any():
        np.testing.assert_allclose(np.diag(M)[unclamped], 1.0, atol=1e-12)
    # Where factor contribution > 1: specific variance is floored at 1e-12, diagonal > 1
    clamped = ~unclamped
    assert np.all(np.diag(M)[clamped] > 1.0)
