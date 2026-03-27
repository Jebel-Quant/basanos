"""Cross-mode validation tests: EWMA vs. factor-model consistency.

These integration-level tests verify that the EWMA (EwmaShrinkConfig) and
factor-model (SlidingWindowConfig) code paths are mutually consistent under
conditions where they should produce the same output.

Test categories
---------------
1. **Solver-path consistency** — given the *same* MatrixBundle, the batched
   EWMA solver (:meth:`_SolveMixin._iter_solve_ewma_batched`) and the
   sequential per-row solver (:meth:`_SolveMixin._compute_position`) produce
   bit-for-bit identical positions.  This test uses *sliding-window* derived
   matrices (not EWMA matrices) to prove path equivalence is independent of
   the covariance estimator.

2. **Status-code structural consistency** — both modes classify zero-signal
   rows as ``"zero_signal"`` and all-NaN price rows as ``"degenerate"``
   regardless of covariance estimator.

3. **Batch output schema consistency** — both modes return ``cash_position``
   DataFrames with the same columns, same ``NaN`` footprint during EWMA-vola
   warmup, and non-NaN positions where at least one asset has data.

4. **Streaming vs. batch equivalence** — for both EWMA and sliding-window
   modes, multiple consecutive :meth:`BasanosStream.step` calls must match
   :attr:`BasanosEngine.cash_position` within ``rtol=1e-8``.

5. **Streaming status-code consistency** — for the same zero-signal data,
   both EWMA and SW streaming modes return ``"zero_signal"`` status for the
   same post-warmup rows.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from basanos.math import (
    BasanosConfig,
    BasanosEngine,
    BasanosStream,
    SlidingWindowConfig,
)
from basanos.math._engine_solve import SolveStatus, _SolveMixin

# ─── Shared helpers ───────────────────────────────────────────────────────────


def _make_dataset(
    *,
    n: int,
    n_assets: int,
    seed: int = 0,
    zero_mu_from: int | None = None,
    nan_price_row: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(prices, mu)`` DataFrames for cross-mode tests.

    Args:
        n: Total number of rows.
        n_assets: Number of assets.
        seed: RNG seed for reproducibility.
        zero_mu_from: If given, all signal rows from this index onward are
            zeroed so that ``zero_signal`` status is expected there.
        nan_price_row: If given, all prices at this row are set to ``NaN``
            so that ``degenerate`` status is expected there.
    """
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 1)
    dates = pl.date_range(
        start=start,
        end=start + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    asset_names = [chr(ord("A") + i) for i in range(n_assets)]

    initial_price = 100.0  # conventional starting price level
    price_arr = np.cumprod(1.0 + rng.normal(0.001, 0.02, size=(n, n_assets)), axis=0) * initial_price
    mu_arr = rng.normal(0.0, 0.5, size=(n, n_assets))

    if zero_mu_from is not None:
        mu_arr[zero_mu_from:] = 0.0
    if nan_price_row is not None:
        price_arr[nan_price_row, :] = np.nan

    prices = pl.DataFrame({"date": dates, **{a: price_arr[:, i] for i, a in enumerate(asset_names)}})
    mu = pl.DataFrame({"date": dates, **{a: mu_arr[:, i] for i, a in enumerate(asset_names)}})
    return prices, mu


def _ewma_cfg(vola: int = 5, corr: int = 10) -> BasanosConfig:
    return BasanosConfig(vola=vola, corr=corr, clip=3.0, shrink=0.5, aum=1e6)


def _sw_cfg(vola: int = 5, corr: int = 10, window: int = 20, n_factors: int = 2) -> BasanosConfig:
    return BasanosConfig(
        vola=vola,
        corr=corr,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        covariance_config=SlidingWindowConfig(window=window, n_factors=n_factors),
    )


# ─── 1. Solver-path consistency (cross-mode matrix injection) ─────────────────


def test_sw_matrices_give_identical_results_via_batched_and_sequential_solvers() -> None:
    r"""Batched EWMA solver and sequential solver agree for sliding-window matrices.

    :meth:`_SolveMixin._iter_solve_ewma_batched` is a vectorised batch
    ``numpy.linalg.solve`` that is the hot-path for the EWMA covariance mode.
    The SW mode uses a sequential :meth:`_SolveMixin._compute_position` loop.

    Both code paths call into the same linear-algebra primitives and must
    agree to float64 precision when fed **identical** :class:`MatrixBundle`
    objects — regardless of which covariance estimator produced the matrix.

    This test feeds sliding-window derived matrices to *both* code paths and
    asserts that positions and status codes are numerically identical.  Any
    divergence between the two solver implementations will cause this test to
    fail regardless of which covariance mode is in active use.
    """
    n, n_assets = 80, 3
    prices, mu = _make_dataset(n=n, n_assets=n_assets, seed=7)
    cfg = _sw_cfg()

    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
    mu_np = engine.mu.select(engine.assets).to_numpy()
    denom_tol = engine.cfg.denom_tol

    # Materialise the SW matrix yields once; both paths consume the same list.
    matrix_yields = list(engine._iter_matrices())

    # ── Batched path (EWMA hot-path) ─────────────────────────────────────────
    batched: dict[int, tuple[np.ndarray | None, SolveStatus]] = {
        i: (pos, status)
        for i, _t, _mask, pos, status in _SolveMixin._iter_solve_ewma_batched(mu_np, matrix_yields, denom_tol)
    }

    # ── Sequential path (SW hot-path) ────────────────────────────────────────
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

    # ── Both paths must cover the same rows ──────────────────────────────────
    batched_keys = sorted(batched.keys())
    sequential_keys = sorted(sequential.keys())
    assert set(batched_keys) == set(sequential_keys), (
        "Batched and sequential paths yielded different row indices: "
        f"batched={batched_keys}, sequential={sequential_keys}"
    )

    # ── Per-row agreement ────────────────────────────────────────────────────
    for i in batched_keys:
        b_pos, b_status = batched[i]
        s_pos, s_status = sequential[i]
        assert b_status == s_status, f"Row {i}: status mismatch: batched={b_status!r}, sequential={s_status!r}"
        if b_pos is not None and s_pos is not None:
            np.testing.assert_allclose(
                b_pos,
                s_pos,
                rtol=1e-12,
                atol=1e-14,
                err_msg=(
                    f"Row {i}: positions differ between batched (EWMA path) "
                    f"and sequential (SW path) when given identical SW matrices"
                ),
            )


# ─── 2. Status-code structural consistency ────────────────────────────────────


def test_both_modes_report_zero_signal_status_for_all_zero_mu() -> None:
    """EWMA and SW batch engines both report ``zero_signal`` for all-zero signal rows.

    After each mode's respective warmup period, rows with ``mu == 0`` should be
    labelled ``"zero_signal"`` in both modes.  Any divergence in status-code
    logic between the two covariance paths will cause this test to fail.
    """
    n, n_assets = 80, 3
    vola, corr, window = 5, 10, 20

    # Signal is zero for all rows at and beyond row 40 (well past warmup in
    # both modes: EWMA warmup ends at row `corr`; SW warmup ends at row `window`).
    zero_from = 40
    prices, mu = _make_dataset(n=n, n_assets=n_assets, seed=11, zero_mu_from=zero_from)

    ewma_engine = BasanosEngine(prices=prices, mu=mu, cfg=_ewma_cfg(vola=vola, corr=corr))
    sw_engine = BasanosEngine(
        prices=prices,
        mu=mu,
        cfg=_sw_cfg(vola=vola, corr=corr, window=window, n_factors=2),
    )

    ewma_statuses = ewma_engine.position_status["status"].to_list()
    sw_statuses = sw_engine.position_status["status"].to_list()

    # Rows from zero_from onward (past both warmup periods) must be zero_signal.
    for i in range(zero_from, n):
        assert ewma_statuses[i] == SolveStatus.ZERO_SIGNAL, (
            f"EWMA mode: row {i} expected zero_signal, got {ewma_statuses[i]!r}"
        )
        assert sw_statuses[i] == SolveStatus.ZERO_SIGNAL, (
            f"SW mode: row {i} expected zero_signal, got {sw_statuses[i]!r}"
        )


def test_both_modes_report_degenerate_status_for_all_nan_price_row() -> None:
    """EWMA and SW batch engines both report ``degenerate`` for an all-NaN price row.

    When every asset price is NaN at a given timestamp, no mask entries are
    active and the optimiser has no valid sub-problem to solve.  Both modes
    must guard against this gracefully and return ``"degenerate"`` status (not
    raise an exception or silently return stale positions).
    """
    n, n_assets = 60, 3
    vola, corr, window = 5, 10, 20
    nan_row = 45  # well past both warmup periods

    prices, mu = _make_dataset(n=n, n_assets=n_assets, seed=22, nan_price_row=nan_row)

    ewma_engine = BasanosEngine(prices=prices, mu=mu, cfg=_ewma_cfg(vola=vola, corr=corr))
    sw_engine = BasanosEngine(
        prices=prices,
        mu=mu,
        cfg=_sw_cfg(vola=vola, corr=corr, window=window, n_factors=2),
    )

    ewma_status = ewma_engine.position_status["status"][nan_row]
    sw_status = sw_engine.position_status["status"][nan_row]

    assert ewma_status == SolveStatus.DEGENERATE, (
        f"EWMA mode: all-NaN price row {nan_row} expected degenerate, got {ewma_status!r}"
    )
    assert sw_status == SolveStatus.DEGENERATE, (
        f"SW mode: all-NaN price row {nan_row} expected degenerate, got {sw_status!r}"
    )


# ─── 3. Batch output schema consistency ──────────────────────────────────────


def test_both_modes_produce_same_cash_position_schema() -> None:
    """EWMA and SW ``cash_position`` DataFrames share the same schema and NaN structure.

    The schema (column names and dtypes) and the *NaN footprint* during the
    EWMA-vola warm-up window must be identical for both modes; the covariance
    estimator must not influence which rows/assets are NaN due to insufficient
    volatility history.
    """
    n, n_assets = 80, 3
    vola = 5

    prices, mu = _make_dataset(n=n, n_assets=n_assets, seed=33)
    ewma_engine = BasanosEngine(prices=prices, mu=mu, cfg=_ewma_cfg(vola=vola))
    sw_engine = BasanosEngine(prices=prices, mu=mu, cfg=_sw_cfg(vola=vola))

    ewma_cp = ewma_engine.cash_position
    sw_cp = sw_engine.cash_position

    # ── Same schema ──────────────────────────────────────────────────────────
    assert ewma_cp.columns == sw_cp.columns, f"Column mismatch: EWMA={ewma_cp.columns}, SW={sw_cp.columns}"
    assert ewma_cp.dtypes == sw_cp.dtypes, f"Dtype mismatch: EWMA={ewma_cp.dtypes}, SW={sw_cp.dtypes}"

    # ── Same NaN footprint during EWMA-vola warmup ───────────────────────────
    assets = ewma_engine.assets
    ewma_np = ewma_cp.select(assets).to_numpy()
    sw_np = sw_cp.select(assets).to_numpy()

    # Rows 0..vola-1 must be NaN in both modes (insufficient vola history).
    for row in range(vola):
        assert np.all(np.isnan(ewma_np[row])), f"EWMA vola-warmup row {row} must be all-NaN"
        assert np.all(np.isnan(sw_np[row])), f"SW vola-warmup row {row} must be all-NaN"

    # ── NaN footprint matches between modes during vola warmup ───────────────
    ewma_nan = np.isnan(ewma_np[:vola])
    sw_nan = np.isnan(sw_np[:vola])
    np.testing.assert_array_equal(
        ewma_nan,
        sw_nan,
        err_msg="EWMA and SW NaN patterns differ during EWMA-vola warmup rows",
    )


# ─── 4. Streaming vs. batch equivalence (both modes, multiple steps) ─────────


@pytest.mark.parametrize("n_steps", [3, 8])
def test_ewma_streaming_matches_batch_for_multiple_steps(n_steps: int) -> None:
    """EWMA streaming must match EWMA batch for ``n_steps`` consecutive rows.

    Extends the single-step streaming equivalence test to cover multiple
    consecutive :meth:`BasanosStream.step` calls, verifying that IIR filter
    state accumulation does not drift relative to the batch EWMA computation.
    """
    warmup_len, n_assets = 50, 3
    n_total = warmup_len + n_steps
    prices, mu = _make_dataset(n=n_total, n_assets=n_assets, seed=44)
    cfg = _ewma_cfg()
    assets = [chr(ord("A") + i) for i in range(n_assets)]

    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
    expected = engine.cash_position.select(assets).to_numpy()

    for step_i in range(n_steps):
        row_idx = warmup_len + step_i
        result = stream.step(prices_np[row_idx], mu_np[row_idx], prices["date"][row_idx])
        np.testing.assert_allclose(
            result.cash_position,
            expected[row_idx],
            rtol=1e-8,
            equal_nan=True,
            err_msg=(f"EWMA streaming/batch mismatch at step {step_i + 1} (row {row_idx})"),
        )


@pytest.mark.parametrize("n_steps", [3, 8])
def test_sw_streaming_matches_batch_for_multiple_steps(n_steps: int) -> None:
    """SW streaming must match SW batch for ``n_steps`` consecutive rows.

    Extends the single-step SW streaming equivalence test to cover multiple
    consecutive :meth:`BasanosStream.step` calls, ensuring the rolling
    return buffer stays in sync with the batch engine's sliding window.
    """
    warmup_len, n_assets = 50, 3
    n_total = warmup_len + n_steps
    prices, mu = _make_dataset(n=n_total, n_assets=n_assets, seed=55)
    cfg = _sw_cfg()
    assets = [chr(ord("A") + i) for i in range(n_assets)]

    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
    expected = engine.cash_position.select(assets).to_numpy()

    for step_i in range(n_steps):
        row_idx = warmup_len + step_i
        result = stream.step(prices_np[row_idx], mu_np[row_idx], prices["date"][row_idx])
        np.testing.assert_allclose(
            result.cash_position,
            expected[row_idx],
            rtol=1e-6,
            equal_nan=True,
            err_msg=(f"SW streaming/batch mismatch at step {step_i + 1} (row {row_idx})"),
        )


# ─── 5. Streaming status-code consistency (zero-signal) ──────────────────────


def test_both_streaming_modes_report_zero_signal_for_zero_mu_step() -> None:
    """EWMA and SW streaming modes both return ``zero_signal`` for a zero-mu step.

    After warmup is complete for both modes, a step with ``mu = 0`` must yield
    ``StepResult.status == "zero_signal"`` regardless of the covariance mode.
    This guards against either streaming implementation accidentally bypassing
    the zero-signal early-exit check.
    """
    n_assets = 3
    warmup_len = 50
    prices, mu = _make_dataset(n=warmup_len + 1, n_assets=n_assets, seed=66)
    assets = [chr(ord("A") + i) for i in range(n_assets)]
    prices_np = prices.select(assets).to_numpy()

    zero_mu = np.zeros(n_assets)
    step_date = prices["date"][warmup_len]

    ewma_stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), _ewma_cfg())
    sw_stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), _sw_cfg())

    ewma_result = ewma_stream.step(prices_np[warmup_len], zero_mu, step_date)
    sw_result = sw_stream.step(prices_np[warmup_len], zero_mu, step_date)

    assert ewma_result.status == SolveStatus.ZERO_SIGNAL, (
        f"EWMA streaming: zero-mu step expected zero_signal, got {ewma_result.status!r}"
    )
    assert sw_result.status == SolveStatus.ZERO_SIGNAL, (
        f"SW streaming: zero-mu step expected zero_signal, got {sw_result.status!r}"
    )


def test_both_streaming_modes_return_zero_positions_for_zero_mu_step() -> None:
    """EWMA and SW streaming modes both return all-zero positions for zero-mu.

    Verifies that the cash-position vector is exactly zero (not NaN or
    near-zero) when the signal is zero, in both covariance modes.
    """
    n_assets = 3
    warmup_len = 50
    prices, mu = _make_dataset(n=warmup_len + 1, n_assets=n_assets, seed=77)
    assets = [chr(ord("A") + i) for i in range(n_assets)]
    prices_np = prices.select(assets).to_numpy()
    zero_mu = np.zeros(n_assets)
    step_date = prices["date"][warmup_len]

    ewma_stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), _ewma_cfg())
    sw_stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), _sw_cfg())

    ewma_result = ewma_stream.step(prices_np[warmup_len], zero_mu, step_date)
    sw_result = sw_stream.step(prices_np[warmup_len], zero_mu, step_date)

    np.testing.assert_array_equal(
        ewma_result.cash_position,
        np.zeros(n_assets),
        err_msg="EWMA streaming: cash_position must be all-zero for zero-mu input",
    )
    np.testing.assert_array_equal(
        sw_result.cash_position,
        np.zeros(n_assets),
        err_msg="SW streaming: cash_position must be all-zero for zero-mu input",
    )
