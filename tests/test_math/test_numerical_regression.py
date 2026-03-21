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
    signal, so all prior cash positions are zero, and the profit-variance EMA
    decays monotonically from ``profit_variance_init``:

    .. math::

       \text{pv}_{\text{corr}} = \text{init} \times \text{decay}^{\text{corr}}

    The expected cash position at the first valid row is then:

    .. math::

       \text{cash\_pos}_{\text{corr}} = \frac{\mu / \|\mu\|_2}{\text{pv}_{\text{corr}} \times \sigma_{\text{corr}}}

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
    decay = engine.cfg.profit_variance_decay
    pv_init = engine.cfg.profit_variance_init
    # pv decays once per row for rows 1 .. first_valid (= first_valid iterations)
    pv_analytical = pv_init * (decay**first_valid)

    mu_active = np.array([3.0, 4.0])
    pos_raw = mu_active / np.linalg.norm(mu_active)  # = [0.6, 0.8]
    expected = pos_raw / (pv_analytical * vola[first_valid])

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
    """Cash positions must match pre-computed golden fixtures bit-for-bit.

    Golden files live in ``tests/resources/`` and are committed to the
    repository.  The test fails (and CI breaks) if any value drifts due to
    an algorithmic change — the contributor must regenerate the fixture and
    explicitly commit the change.

    The comparison uses :func:`numpy.testing.assert_array_equal` (exact
    bit-for-bit equality) so even ULP-level rounding differences are caught.
    NaN values are treated as equal (``equal_nan=True``).
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

    np.testing.assert_array_equal(actual, golden, strict=True)


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
    np.testing.assert_array_equal(actual[first_valid], golden[first_valid])

    # ── Full array matches golden (bit-for-bit, NaN == NaN) ───────────────
    np.testing.assert_array_equal(actual, golden, strict=True)
