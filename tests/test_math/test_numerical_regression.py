"""Numerical precision and regression tests against analytical solutions.

Tests in this module verify that the optimizer produces *exact* numeric
outputs — not just structural properties — for well-understood cases:

1. **Closed-form 2-asset test** (:func:`test_solver_identity_correlation_produces_unit_direction`):
   For a 2×2 identity correlation matrix, the solver output from
   :meth:`~basanos.math._engine_solve._SolveMixin._compute_position` is
   analytically known as ``mu / ‖mu‖₂``.  This is verified to float64 epsilon
   using a known Pythagorean triple (``mu = [3, 4]``).

2. **Full-pipeline analytical test** (:func:`test_2asset_diagonal_shrink_zero_first_valid_row`):
   With ``shrink=0.0`` the effective correlation matrix is always the identity
   (after warmup).  At the first valid row the profit-variance EMA equals
   ``profit_variance_init × decay^N`` (where *N* is the warmup length and all
   prior positions were zero), giving a closed-form prediction for
   ``cash_position`` that is verified to within float64 relative error.

3. **Golden-output regression** (:func:`test_golden_output_matches_fixture`):
   Pre-computed cash-position arrays stored in ``tests/resources/`` are loaded
   and compared bit-for-bit against the current implementation.  CI fails
   whenever an algorithmic refactor shifts any value.

4. **EWMA warm-up path** (:func:`test_ewma_warmup_phase_structure`):
   Asserts the three-phase warmup structure (NaN / zero / valid) with
   pre-computed expected values for the first valid post-warmup row.
"""

from __future__ import annotations

import pathlib
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from basanos.math import BasanosConfig, BasanosEngine
from basanos.math._engine_solve import MatrixBundle, SolveStatus, _SolveMixin

# ─── Paths ────────────────────────────────────────────────────────────────────

_RESOURCES = pathlib.Path(__file__).parent.parent / "resources"

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_2asset_engine(
    *,
    n: int,
    rng_seed: int,
    vola: int,
    corr: int,
    shrink: float,
    mu_override: np.ndarray | None = None,
) -> tuple[BasanosEngine, pl.DataFrame, pl.DataFrame]:
    """Return ``(engine, prices, mu)`` for a seeded 2-asset scenario.

    Args:
        n: Number of rows.
        rng_seed: Seed for :class:`numpy.random.default_rng`.
        vola: EWMA volatility lookback.
        corr: EWMA correlation lookback.
        shrink: Shrinkage towards identity.
        mu_override: If provided, used as the (n, 2) signal array directly.
            Otherwise a sinusoidal signal is derived from *n*.
    """
    rng = np.random.default_rng(rng_seed)
    p_a = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=n))
    p_b = 200.0 + np.cumsum(rng.normal(0.0, 0.7, size=n))
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    prices = pl.DataFrame({"date": dates, "A": pl.Series(p_a), "B": pl.Series(p_b)})
    if mu_override is not None:
        mu = pl.DataFrame({"date": dates, "A": pl.Series(mu_override[:, 0]), "B": pl.Series(mu_override[:, 1])})
    else:
        theta = np.linspace(0.0, 4.0 * np.pi, num=n)
        mu = pl.DataFrame(
            {"date": dates, "A": pl.Series(np.tanh(np.sin(theta))), "B": pl.Series(np.tanh(np.cos(theta)))}
        )
    cfg = BasanosConfig(vola=vola, corr=corr, clip=3.0, shrink=shrink, aum=1e6)
    return BasanosEngine(prices=prices, mu=mu, cfg=cfg), prices, mu


# ─── 1. Closed-form 2-asset solver test ──────────────────────────────────────


def test_solver_identity_correlation_produces_unit_direction() -> None:
    """_compute_position with C=I yields ``mu / ‖mu‖₂`` to float64 precision.

    For the Pythagorean triple ``mu = [3, 4]``:

    * ``denom = inv_a_norm([3, 4], I) = sqrt(9 + 16) = 5.0``
    * ``pos   = solve(I, [3, 4]) = [3, 4]``
    * result  = ``pos / denom = [0.6, 0.8]``

    This is a closed-form solution — no numerical approximation is involved
    for this specific input — so the comparison uses ``atol=0, rtol=1e-15``.
    """
    identity = np.eye(2)
    mu_vec = np.array([3.0, 4.0])
    mask = np.array([True, True])
    bundle = MatrixBundle(matrix=identity)
    t = date(2020, 1, 1)

    _i, _t, _mask, pos, status = _SolveMixin._compute_position(
        i=0, t=t, mask=mask, expected_mu=mu_vec, bundle=bundle, denom_tol=1e-12
    )

    assert status == SolveStatus.VALID
    expected = mu_vec / np.linalg.norm(mu_vec)  # [0.6, 0.8]
    np.testing.assert_allclose(pos, expected, atol=0, rtol=1e-15)


@pytest.mark.parametrize(
    "mu_vec",
    [
        np.array([1.0, 0.0]),  # degenerate: unit vector along A
        np.array([0.0, 1.0]),  # degenerate: unit vector along B
        np.array([1.0, np.sqrt(3.0)]),  # 30-60-90 triangle
        np.array([-3.0, 4.0]),  # negative component
    ],
    ids=["unit_a", "unit_b", "30_60_90", "negative_component"],
)
def test_solver_identity_correlation_unit_direction_parametrised(mu_vec: np.ndarray) -> None:
    """_compute_position with C=I always maps mu to the unit vector along mu."""
    identity = np.eye(2)
    mask = np.array([True, True])
    bundle = MatrixBundle(matrix=identity)
    t = date(2020, 6, 1)

    _i, _t, _mask, pos, status = _SolveMixin._compute_position(
        i=0, t=t, mask=mask, expected_mu=mu_vec, bundle=bundle, denom_tol=1e-12
    )

    assert status == SolveStatus.VALID
    expected = mu_vec / np.linalg.norm(mu_vec)
    np.testing.assert_allclose(pos, expected, atol=0, rtol=1e-15)


# ─── 2. Full-pipeline analytical test ────────────────────────────────────────


def test_2asset_diagonal_shrink_zero_first_valid_row() -> None:
    r"""Full-pipeline closed-form check for 2-asset identity correlation.

    With ``shrink=0.0`` the shrunk correlation matrix is always the identity
    for any real-valued EWMA correlation matrix (``shrink2id(C, 0) = I``).

    The scenario is constructed so that only row ``corr`` (= first valid row)
    carries a non-zero signal (``mu = [3, 4]``).  All prior rows have zero
    signal, so all prior positions are zero.  At the first valid row the
    expected cash position is simply:

    .. math::

       \text{cash\_pos}_{\text{corr}} = \frac{\mu / \|\mu\|_2}{\sigma_{\text{corr}}}

    This is verified to within float64 relative error (~machine epsilon).
    """
    vola_span = 5
    corr_span = 10
    n = corr_span + 5  # a few rows beyond warmup
    first_valid = corr_span  # first row with non-NaN correlation

    mu_override = np.zeros((n, 2))
    mu_override[first_valid, :] = [3.0, 4.0]

    engine, _prices, _mu = _make_2asset_engine(
        n=n, rng_seed=17, vola=vola_span, corr=corr_span, shrink=0.0, mu_override=mu_override
    )

    cp = engine.cash_position.select(["A", "B"]).to_numpy()
    vola = engine.vola.select(["A", "B"]).to_numpy()

    # ── Verify warmup rows ─────────────────────────────────────────────────
    # Rows 0 .. vola_span-1: NaN (insufficient EWMA vola history)
    assert np.all(np.isnan(cp[:vola_span])), "Expected NaN during vola warmup"
    # Rows vola_span .. first_valid-1: zero (zero signal during corr warmup)
    assert np.all(cp[vola_span:first_valid] == 0.0), "Expected zero during corr warmup"

    # ── Verify first valid row against analytical formula ─────────────────
    mu_active = np.array([3.0, 4.0])
    pos_raw = mu_active / np.linalg.norm(mu_active)  # = [0.6, 0.8]
    expected = pos_raw / vola[first_valid]

    np.testing.assert_allclose(cp[first_valid], expected, rtol=1e-12, atol=0)


# ─── 3. Golden-output regression tests ───────────────────────────────────────

_GOLDEN_SCENARIOS = [
    pytest.param(
        {
            "fixture": "golden_2asset_cash_position.npy",
            "n": 100,
            "rng_seed": 42,
            "vola": 10,
            "corr": 20,
            "shrink": 0.5,
        },
        id="2asset_seeded_rng42",
    ),
    pytest.param(
        {
            "fixture": "golden_ewma_warmup_cash_position.npy",
            "n": 60,
            "rng_seed": 7,
            "vola": 5,
            "corr": 15,
            "shrink": 0.5,
        },
        id="ewma_warmup_rng7",
    ),
]


@pytest.mark.parametrize("scenario", _GOLDEN_SCENARIOS)
def test_golden_output_matches_fixture(scenario: dict) -> None:
    """Cash positions must match pre-computed golden fixtures to float64 precision.

    Golden files live in ``tests/resources/`` and are committed to the
    repository.  The test fails (and CI breaks) if any value drifts due to
    an algorithmic change — the contributor must regenerate the fixture and
    explicitly commit the change.

    The comparison uses :func:`numpy.testing.assert_allclose` with
    ``rtol=1e-12, atol=1e-11`` to tolerate ULP-level rounding differences
    (< 2e-13 absolute) introduced by vectorised linear-algebra paths.
    NaN values are treated as equal.
    """
    fixture_path = _RESOURCES / scenario["fixture"]
    if not fixture_path.exists():
        pytest.skip(f"Golden fixture not found: {fixture_path}")

    golden = np.load(fixture_path)

    engine, _prices, _mu = _make_2asset_engine(
        n=scenario["n"],
        rng_seed=scenario["rng_seed"],
        vola=scenario["vola"],
        corr=scenario["corr"],
        shrink=scenario["shrink"],
    )
    actual = engine.cash_position.select(["A", "B"]).to_numpy()

    np.testing.assert_allclose(actual, golden, rtol=1e-12, atol=1e-11, equal_nan=True)


# ─── 4. EWMA warm-up path with pre-computed expected values ──────────────────


def test_ewma_warmup_phase_structure() -> None:
    """Verify the three-phase warmup structure and first-valid-row values.

    For ``vola=V`` and ``corr=C`` (with ``C >= V``):

    * **Phase 1** — rows ``0 .. V-1``: cash position is ``NaN`` because the
      EWMA volatility estimator has not yet accumulated enough observations.
    * **Phase 2** — rows ``V .. C-1``: cash position is ``0`` because the
      EWMA correlation matrix is still within its own minimum-periods window.
    * **Phase 3** — rows ``C+``: cash position is non-trivially non-zero when
      the signal is non-zero.

    The fixture ``golden_ewma_warmup_cash_position.npy`` stores the reference
    outputs for all 60 rows.  Phase-boundary values are loaded from the fixture
    and cross-checked to confirm the warm-up transitions happened exactly at
    the expected row indices.
    """
    vola_span = 5
    corr_span = 15
    n = 60
    rng_seed = 7
    shrink = 0.5

    fixture_path = _RESOURCES / "golden_ewma_warmup_cash_position.npy"
    if not fixture_path.exists():
        pytest.skip(f"Golden fixture not found: {fixture_path}")
    golden = np.load(fixture_path)

    engine, _prices, _mu = _make_2asset_engine(n=n, rng_seed=rng_seed, vola=vola_span, corr=corr_span, shrink=shrink)
    actual = engine.cash_position.select(["A", "B"]).to_numpy()

    # ── Phase 1: NaN during EWMA-vola warmup ──────────────────────────────
    assert np.all(np.isnan(actual[:vola_span])), f"Rows 0..{vola_span - 1} must be NaN (EWMA-vola warmup)"

    # ── Phase 2: zero during EWMA-corr warmup ─────────────────────────────
    assert np.all(actual[vola_span:corr_span] == 0.0), f"Rows {vola_span}..{corr_span - 1} must be 0 (EWMA-corr warmup)"

    # ── Phase 3: first valid row matches golden fixture ────────────────────
    first_valid = corr_span
    np.testing.assert_allclose(actual[first_valid], golden[first_valid], rtol=1e-12, atol=1e-11)

    # ── Full array matches golden (NaN == NaN) ────────────────────────────
    np.testing.assert_allclose(actual, golden, rtol=1e-12, atol=1e-11, equal_nan=True)


# ─── 5. EwmaShrink batched vs sequential cross-path consistency ───────────────


def test_ewma_batch_and_sequential_paths_agree() -> None:
    r"""Batched EwmaShrink solve must match sequential ``_compute_position`` exactly.

    ``_iter_solve`` dispatches to two different implementations depending on the
    covariance mode:

    * :meth:`~basanos.math._engine_solve._SolveMixin._iter_solve_ewma_batched`
      — vectorised batch ``numpy.linalg.solve`` grouped by mask pattern
      (:class:`~basanos.math.EwmaShrinkConfig`).
    * A sequential :meth:`~basanos.math._engine_solve._SolveMixin._compute_position`
      loop (:class:`~basanos.math.SlidingWindowConfig`).

    This test asserts that both paths produce numerically identical
    ``(pos, status)`` tuples for every row when fed the same EwmaShrink
    covariance matrices, independently of the golden-file regression tests.
    Any future divergence between the two branches — a new edge case, a new
    :class:`~basanos.math._engine_solve.SolveStatus` value, or a change to
    denominator logic — will cause this test to fail immediately.
    """
    engine, _prices, _mu = _make_2asset_engine(n=80, rng_seed=99, vola=10, corr=20, shrink=0.5)
    mu_np = engine.mu.select(engine.assets).to_numpy()
    denom_tol = engine.cfg.denom_tol
    matrix_yields = list(engine._iter_matrices())

    # ── Batched path (the actual EwmaShrink implementation) ───────────────
    batched: dict[int, tuple[np.ndarray | None, SolveStatus]] = {
        i: (pos, status)
        for i, _t, _mask, pos, status in _SolveMixin._iter_solve_ewma_batched(mu_np, matrix_yields, denom_tol)
    }

    # ── Sequential path: replicate the SlidingWindow loop using EwmaShrink
    #    matrix bundles.  This is what _iter_solve would do if EwmaShrinkConfig
    #    used the sequential branch.
    sequential: dict[int, tuple[np.ndarray | None, SolveStatus]] = {}
    for i, t, mask, bundle in matrix_yields:
        if bundle is None:
            sequential[i] = (np.zeros(int(mask.sum())), SolveStatus.DEGENERATE)
            continue
        expected_mu, early = _SolveMixin._row_early_check(i, t, mask, mu_np[i])
        if early is not None:
            _i, _t, _mask, pos, status = early
            sequential[i] = (pos, status)
            continue
        _i, _t, _mask, pos, status = _SolveMixin._compute_position(i, t, mask, expected_mu, bundle, denom_tol)
        sequential[i] = (pos, status)

    # ── Both paths must cover the same row indices ────────────────────────
    assert set(batched.keys()) == set(sequential.keys()), (
        "Batched and sequential paths yielded different row indices: "
        f"batched={sorted(batched.keys())}, sequential={sorted(sequential.keys())}"
    )

    # ── Per-row: statuses and positions must agree ─────────────────────────
    for i in sorted(batched.keys()):
        b_pos, b_status = batched[i]
        s_pos, s_status = sequential[i]
        assert b_status == s_status, f"Row {i}: status mismatch: batched={b_status!r}, sequential={s_status!r}"
        if b_pos is not None and s_pos is not None:
            np.testing.assert_allclose(
                b_pos,
                s_pos,
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"Row {i}: position vectors differ between batched and sequential paths",
            )
