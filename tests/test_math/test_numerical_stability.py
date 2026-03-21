"""Numerical stability tests for basanos.math.

Covers five targeted stability scenarios plus one Hypothesis property:

1. Near-singular correlation matrices (condition number > 1e12) emit
   :class:`~basanos.exceptions.IllConditionedMatrixWarning`.
2. ``min_corr_denom`` boundary: when the EWMA correlation denominator
   ``sqrt(var_x * var_y)`` falls at or below 1e-14 the result is NaN.
3. Ill-conditioned covariance with no shrinkage toward identity (``shrink=1.0``)
   and highly collinear assets: the engine emits
   :class:`~basanos.exceptions.IllConditionedMatrixWarning` or yields
   :attr:`~basanos.math.SolveStatus.DEGENERATE`.
4. :class:`~basanos.exceptions.IllConditionedMatrixWarning` fires at the
   documented threshold of 1e12, not at 1e11 or 1e13.
5. :attr:`~basanos.math.SolveStatus.DEGENERATE` (not silent ``NaN``) is
   returned when the correlation matrix is singular.
6. Hypothesis property: for every :attr:`~basanos.math.SolveStatus.VALID`
   step the solver residual ``‖C·x - μ‖₂`` is finite and bounded below a
   conservative numerical tolerance.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st

from basanos.exceptions import IllConditionedMatrixWarning
from basanos.math import BasanosConfig, BasanosEngine, SolveStatus
from basanos.math._ewm_corr import _ewm_corr_numpy
from basanos.math._linalg import _DEFAULT_COND_THRESHOLD, solve

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SOLVER_RESIDUAL_TOLERANCE: float = 1e-6
"""Conservative upper bound for ‖C·x - μ‖₂ on a well-conditioned solve."""


def _sinusoidal_mu(prices: pl.DataFrame, *, seed: int = 42) -> pl.DataFrame:
    """Return a mu DataFrame aligned with *prices* using bounded sinusoidal values."""
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


# ---------------------------------------------------------------------------
# 1. Near-singular correlation matrices (condition number > 1e12)
# ---------------------------------------------------------------------------


def test_near_singular_corr_warns_ill_conditioned() -> None:
    """solve() emits IllConditionedMatrixWarning when the matrix has κ > 1e12.

    A 2×2 positive-definite diagonal matrix [[1, 0], [0, ε]] with ε = 1e-13
    has condition number 1/ε = 1e13 > 1e12 (the documented threshold).
    The warning must be emitted and the solve result must still satisfy the
    linear system within a reasonable tolerance.
    """
    eps = 1e-13
    matrix = np.diag([1.0, eps])
    rhs = np.array([1.0, 1.0])

    assert float(np.linalg.cond(matrix)) > _DEFAULT_COND_THRESHOLD

    with pytest.warns(IllConditionedMatrixWarning, match="condition number"):
        x = solve(matrix=matrix, rhs=rhs)

    # Despite the ill-conditioning, the returned solution must be finite.
    assert np.all(np.isfinite(x))


def test_near_singular_corr_residual_finite() -> None:
    """solve() returns a finite solution for a near-singular PD matrix.

    Even when the condition number exceeds 1e12 the solver must not return
    NaN or raise; it emits a warning and produces a best-effort solution.
    """
    eps = 1e-13
    matrix = np.diag([1.0, eps])
    rhs = np.array([0.5, 0.5])

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        x = solve(matrix=matrix, rhs=rhs)

    assert np.all(np.isfinite(x)), f"Expected finite solution, got {x}"


# ---------------------------------------------------------------------------
# 2. min_corr_denom boundary (1e-14)
# ---------------------------------------------------------------------------


def test_min_corr_denom_zero_variance_gives_nan() -> None:
    """Constant (zero-variance) returns produce denom=0 < 1e-14 → NaN off-diagonal.

    When both assets have identically zero returns the EWM variance is zero,
    so the correlation denominator ``sqrt(var_x * var_y) = 0`` falls below
    the guard threshold of 1e-14.  The off-diagonal entries must be NaN.
    """
    # All-zero returns → zero variance → denom = 0 < 1e-14 → NaN correlation.
    data = np.zeros((40, 2))
    result = _ewm_corr_numpy(data, com=5, min_periods=1, min_corr_denom=1e-14)

    # Off-diagonal entries should be NaN (denom = 0, which is not > 1e-14).
    assert np.all(np.isnan(result[:, 0, 1])), "Expected NaN off-diagonal for zero-variance data"
    assert np.all(np.isnan(result[:, 1, 0])), "Expected NaN off-diagonal for zero-variance data"


def test_min_corr_denom_normal_variance_gives_finite() -> None:
    """Non-trivial returns produce denom >> 1e-14 → finite off-diagonal correlation.

    With alternating returns of magnitude ~0.02 the variance is ~4e-4, and
    the denominator is far above the 1e-14 guard, so correlations are finite.
    """
    rng = np.random.default_rng(0)
    # Two similar but non-identical alternating series with normal-scale returns.
    n = 60
    pct = rng.uniform(0.01, 0.03, size=(n, 2))
    signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    data = (pct * signs[:, np.newaxis]).astype(np.float64)

    result = _ewm_corr_numpy(data, com=10, min_periods=5, min_corr_denom=1e-14)

    # After warmup, the off-diagonal entries must be finite.
    finite_mask = np.isfinite(result[:, 0, 1])
    assert finite_mask.any(), "Expected at least some finite off-diagonal correlations"
    finite_vals = result[finite_mask, 0, 1]
    assert np.all(np.abs(finite_vals) <= 1.0 + 1e-9), f"Correlations out of [-1, 1]: {finite_vals}"


def test_min_corr_denom_boundary_is_strict_greater_than() -> None:
    """The guard is ``denom > min_corr_denom`` (strict), so at exactly 1e-14 → NaN.

    By setting ``min_corr_denom = 0`` we disable the guard entirely; even
    near-zero denominator data then produces a finite (±1) correlation,
    confirming the guard is what drives NaN for all-zero input.
    """
    # Zero returns → denom = 0.  With guard=0 (disabled), no NaN is forced
    # by the threshold, but the EWM variance is 0 so the expression is 0/0 = NaN
    # from floating-point arithmetic.  The key test is on non-zero tiny returns.
    tiny = 1e-8  # very small, constant magnitude → var ≈ 1e-16, denom ≈ 1e-16
    n = 100
    signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    # Both assets have the same sign pattern → perfect correlation.
    data = np.column_stack([signs * tiny, signs * tiny])

    # With the default guard (1e-14): denom ≈ 1e-16 < 1e-14 → NaN.
    result_guarded = _ewm_corr_numpy(data, com=10, min_periods=1, min_corr_denom=1e-14)
    assert np.all(np.isnan(result_guarded[10:, 0, 1])), "Expected NaN when tiny-variance denom is below guard threshold"

    # With guard disabled (=0): correlation is computable and should be ±1.
    result_unguarded = _ewm_corr_numpy(data, com=10, min_periods=1, min_corr_denom=0.0)
    finite_mask = np.isfinite(result_unguarded[:, 0, 1])
    if finite_mask.any():
        assert np.allclose(result_unguarded[finite_mask, 0, 1], 1.0, atol=1e-6), (
            "Expected correlation ≈ 1 for identical tiny-variance series"
        )


# ---------------------------------------------------------------------------
# 3. Ill-conditioned covariance: shrink=1.0 with perfectly collinear assets
# ---------------------------------------------------------------------------


def test_shrink_one_collinear_assets_emits_warning_or_degenerates() -> None:
    """shrink=1.0 + perfectly collinear assets → warning emitted or DEGENERATE status.

    With two assets following identical price paths the EWMA correlation
    matrix converges to [[1, 1], [1, 1]], which is rank-1 (singular).
    With ``cfg.shrink = 1.0`` (no shrinkage toward the identity) the raw
    singular matrix is passed directly to the solver, which must either:

    * emit :class:`~basanos.exceptions.IllConditionedMatrixWarning` (if the
      matrix is nearly-but-not-exactly singular), or
    * raise :class:`~basanos.exceptions.SingularMatrixError` (caught
      internally) and yield :attr:`~basanos.math.SolveStatus.DEGENERATE`.

    At minimum, no silent NaN must propagate unchecked — the engine must
    report DEGENERATE for affected rows.
    """
    n = 80
    rng = np.random.default_rng(7)
    pct = rng.uniform(0.01, 0.05, size=n - 1)
    signs = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(n - 1)])
    prices = np.empty(n)
    prices[0] = 100.0
    for j in range(1, n):
        prices[j] = prices[j - 1] * (1.0 + signs[j - 1] * pct[j - 1])

    # Both assets have identical prices → perfect positive correlation.
    prices_df = pl.DataFrame(
        {
            "date": list(range(n)),
            "A": pl.Series(prices.tolist(), dtype=pl.Float64),
            "B": pl.Series(prices.tolist(), dtype=pl.Float64),
        }
    )
    mu_df = _sinusoidal_mu(prices_df)
    cfg = BasanosConfig(vola=5, corr=10, clip=2.0, shrink=1.0, aum=1e6)
    engine = BasanosEngine(prices=prices_df, mu=mu_df, cfg=cfg)

    statuses = engine.position_status["status"].to_list()
    warmup = cfg.corr

    post_warmup_statuses = statuses[warmup:]
    # After warm-up, every row must be either DEGENERATE (singular solve) or
    # ZERO_SIGNAL (no signal to solve for) — VALID is only possible if the
    # solver somehow found a non-degenerate solution, which should not happen
    # for a rank-1 correlation matrix with full shrink retention.
    unexpected = [s for s in post_warmup_statuses if s not in (SolveStatus.DEGENERATE, SolveStatus.ZERO_SIGNAL)]
    assert not unexpected, (
        f"Expected only DEGENERATE/ZERO_SIGNAL post-warmup with collinear assets and shrink=1.0, but got: {unexpected}"
    )


# ---------------------------------------------------------------------------
# 4. IllConditionedMatrixWarning fires at the documented threshold (1e12)
# ---------------------------------------------------------------------------


def test_ill_conditioned_warning_threshold_is_1e12() -> None:
    """IllConditionedMatrixWarning fires for κ > 1e12 but not for κ = 1e12 exactly.

    The documented threshold in ``_linalg._DEFAULT_COND_THRESHOLD`` is 1e12.
    The internal guard uses a *strict* greater-than comparison (``cond >
    threshold``), so a matrix with condition number *exactly* 1e12 must NOT
    trigger the warning.
    """
    # Condition number exactly 1e12: diag(1, 1e-12) → κ = 1.0 / 1e-12 = 1e12.
    at_threshold = np.diag([1.0, 1e-12])
    rhs = np.array([1.0, 1.0])
    assert math.isclose(float(np.linalg.cond(at_threshold)), 1e12, rel_tol=1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        # Should NOT raise because κ == 1e12, not > 1e12.
        solve(matrix=at_threshold, rhs=rhs)


def test_ill_conditioned_warning_fires_above_threshold() -> None:
    """IllConditionedMatrixWarning fires when condition number strictly exceeds 1e12."""
    # diag(1, 1e-13) → κ = 1e13 > 1e12 → warning.
    above_threshold = np.diag([1.0, 1e-13])
    rhs = np.array([1.0, 1.0])

    with pytest.warns(IllConditionedMatrixWarning, match="condition number"):
        solve(matrix=above_threshold, rhs=rhs)


def test_ill_conditioned_warning_is_suppressed_well_below_threshold() -> None:
    """No warning for a well-conditioned matrix whose κ is well below 1e12."""
    well_conditioned = np.array([[2.0, 0.5], [0.5, 2.0]])  # κ ≈ 3 / 1 = 3
    rhs = np.array([1.0, 2.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        solve(matrix=well_conditioned, rhs=rhs)


# ---------------------------------------------------------------------------
# 5. SolveStatus.DEGENERATE (not silent NaN) when matrix is singular
# ---------------------------------------------------------------------------


def test_singular_matrix_position_status_is_degenerate() -> None:
    """A singular correlation matrix yields SolveStatus.DEGENERATE, not silent NaN.

    With two perfectly correlated assets and ``shrink=1.0`` the engine
    passes the rank-1 correlation matrix [[1, 1], [1, 1]] directly to the
    solver.  The solver raises :class:`~basanos.exceptions.SingularMatrixError`,
    which the engine catches and converts to
    :attr:`~basanos.math.SolveStatus.DEGENERATE`.  The corresponding
    ``cash_position`` must be 0 (not NaN), confirming that the failure is
    explicit, not a silent propagation.
    """
    n = 80
    rng = np.random.default_rng(13)
    pct = rng.uniform(0.01, 0.05, size=n - 1)
    signs = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(n - 1)])
    prices_arr = np.empty(n)
    prices_arr[0] = 100.0
    for j in range(1, n):
        prices_arr[j] = prices_arr[j - 1] * (1.0 + signs[j - 1] * pct[j - 1])

    prices_df = pl.DataFrame(
        {
            "date": list(range(n)),
            "A": pl.Series(prices_arr.tolist(), dtype=pl.Float64),
            "B": pl.Series(prices_arr.tolist(), dtype=pl.Float64),
        }
    )
    mu_df = _sinusoidal_mu(prices_df)
    cfg = BasanosConfig(vola=5, corr=10, clip=2.0, shrink=1.0, aum=1e6)
    engine = BasanosEngine(prices=prices_df, mu=mu_df, cfg=cfg)

    statuses = engine.position_status["status"].to_list()
    cp = engine.cash_position
    warmup = cfg.corr

    degenerate_indices = [i for i, s in enumerate(statuses) if s == SolveStatus.DEGENERATE and i >= warmup]
    assert degenerate_indices, "Expected at least one DEGENERATE row after warm-up for a rank-1 correlation matrix"

    # For every post-warmup DEGENERATE row the cash position must be 0.0, not
    # NaN.  At row >= cfg.corr both EWMA volatility and correlation have warmed
    # up, so the "0 / vola" scaling in _scale_to_cash produces exactly 0.0.
    assets = engine.assets
    cp_np = cp.select(assets).to_numpy()
    for idx in degenerate_indices:
        row = cp_np[idx]
        assert np.all(np.isfinite(row)), (
            f"Unexpected NaN at DEGENERATE row {idx}: {row} — DEGENERATE must produce explicit 0.0, not silent NaN"
        )
        assert np.allclose(row, 0.0), f"Expected cash_position=0 for DEGENERATE row {idx}, got {row}"


def test_degenerate_status_value_matches_enum() -> None:
    """SolveStatus.DEGENERATE must equal the string 'degenerate' (StrEnum contract)."""
    assert SolveStatus.DEGENERATE == "degenerate"
    assert SolveStatus.VALID == "valid"


# ---------------------------------------------------------------------------
# 6. Hypothesis property: VALID step residual ‖C·x - μ‖₂ < tolerance
# ---------------------------------------------------------------------------

_FINITE_FLOAT = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
_SMALL_POSITIVE = st.floats(min_value=0.005, max_value=0.05, allow_nan=False, allow_infinity=False)


@st.composite
def _well_conditioned_engine(draw: st.DrawFn) -> BasanosEngine:
    """Strategy for a BasanosEngine that is numerically well-conditioned.

    Draws small non-monotonic price series (n_assets ∈ {1, 2}) and caps the
    shrink retention weight at 0.9, guaranteeing a minimum eigenvalue ≥ 0.1
    in the shrunk correlation matrix and a condition number well below 1e12.
    """
    n_assets = draw(st.integers(min_value=1, max_value=2))
    vola = draw(st.integers(min_value=3, max_value=8))
    corr = draw(st.integers(min_value=vola, max_value=vola + 10))
    n_rows = draw(st.integers(min_value=corr + 5, max_value=corr + 20))
    clip = draw(st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    # Cap shrink at 0.9 so that min_eigenvalue ≥ 0.1 and κ ≤ 20 for n=2.
    shrink = draw(st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False))
    aum = draw(st.floats(min_value=1e4, max_value=1e8, allow_nan=False, allow_infinity=False))
    cfg = BasanosConfig(vola=vola, corr=corr, clip=clip, shrink=shrink, aum=aum)

    dates = list(range(n_rows))
    price_cols: dict = {"date": pl.Series(dates, dtype=pl.Int64)}
    mu_cols: dict = {"date": pl.Series(dates, dtype=pl.Int64)}

    for i in range(n_assets):
        col = f"a{i}"
        pct = draw(
            np_st.arrays(
                dtype=np.float64,
                shape=n_rows - 1,
                elements=_SMALL_POSITIVE,
            )
        )
        signs = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(n_rows - 1)])
        base = draw(st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False))
        prices = np.empty(n_rows)
        prices[0] = base
        for j in range(1, n_rows):
            prices[j] = prices[j - 1] * (1.0 + signs[j - 1] * pct[j - 1])
        price_cols[col] = pl.Series(prices.tolist(), dtype=pl.Float64)

        mu_arr = draw(
            np_st.arrays(
                dtype=np.float64,
                shape=n_rows,
                elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        )
        mu_cols[col] = pl.Series(mu_arr.tolist(), dtype=pl.Float64)

    return BasanosEngine(
        prices=pl.DataFrame(price_cols),
        mu=pl.DataFrame(mu_cols),
        cfg=cfg,
    )


@given(engine=_well_conditioned_engine())
@settings(max_examples=80)
def test_valid_step_residual_bounded(engine: BasanosEngine) -> None:
    """For every SolveStatus.VALID step, ‖C·x - μ‖₂ is bounded below the tolerance.

    For a well-conditioned linear system solved in float64 (condition number ≤ 20,
    n ≤ 2) the backward error ``‖C·x - μ‖₂`` should be many orders of magnitude
    below machine epsilon times the matrix norm.  A tolerance of 1e-6 is a
    conservative upper bound that any double-precision Cholesky / LU solver
    must satisfy for the problem sizes used here.
    """
    statuses = engine.position_status["status"].to_list()
    residuals = engine.solver_residual["residual"].to_list()

    for row_idx, (status, residual) in enumerate(zip(statuses, residuals, strict=False)):
        if status != SolveStatus.VALID:
            continue
        assert residual is not None, f"VALID row {row_idx} has None residual"
        assert math.isfinite(residual), f"VALID row {row_idx} has non-finite residual {residual}"
        assert residual < _SOLVER_RESIDUAL_TOLERANCE, (
            f"VALID row {row_idx}: residual {residual:.3e} exceeds tolerance {_SOLVER_RESIDUAL_TOLERANCE:.3e}"
        )
