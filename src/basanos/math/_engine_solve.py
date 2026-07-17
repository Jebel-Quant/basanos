"""Solve/position mixin for BasanosEngine.

This private module contains `_SolveMixin`, which provides the
``_iter_matrices`` and ``_iter_solve`` generator methods.  Separating them
from `optimizer` keeps the engine facade lean and makes
the per-timestamp solve logic independently readable and testable.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, cast

import numpy as np
from cvx.linalg import SingularMatrixError, inv_a_norm, solve

from ._config import EwmaShrinkConfig, SlidingWindowConfig
from ._engine_solve_base import MatrixBundle as MatrixBundle
from ._engine_solve_base import MatrixYield as MatrixYield
from ._engine_solve_base import SolveStatus as SolveStatus
from ._engine_solve_base import SolveYield as SolveYield
from ._engine_solve_base import WarmupState as WarmupState
from ._engine_solve_base import _SolvePrimitivesMixin
from ._factor_model import FactorModel
from ._signal import shrink2id

if TYPE_CHECKING:
    from ._engine_protocol import _EngineProtocol

_logger = logging.getLogger(__name__)


class _SolveMixin(_SolvePrimitivesMixin):
    """Mixin that provides ``_iter_matrices`` and ``_iter_solve`` generators.

    Inherits the stateless helpers from `_SolvePrimitivesMixin` and adds the
    per-timestamp solve orchestration.  Consumers must also inherit from (or
    satisfy the interface of) `_EngineProtocol` so that
    ``self.assets``, ``self.prices``, ``self.mu``, ``self.cfg``, ``self.cor``,
    and ``self.ret_adj`` are all available.
    """

    @staticmethod
    def _compute_position(
        i: int,
        t: datetime.date,
        mask: np.ndarray,
        expected_mu: np.ndarray,
        bundle: MatrixBundle,
        denom_tol: float,
    ) -> SolveYield:
        """Shared solve step used by both covariance branches.

        Computes the normalisation denominator via `inv_a_norm`
        and solves the linear system via `solve`, then
        delegates to `_denom_guard_yield`.  Handles
        :exc:`~basanos.exceptions.SingularMatrixError` from both calls.

        Accepting a `MatrixBundle` instead of a raw array means future
        covariance modes can attach auxiliary state to the bundle without
        changing this method's signature.

        Args:
            i: Row index.
            t: Timestamp.
            mask: Boolean asset mask of shape ``(n_assets,)``.
            expected_mu: Masked expected-return vector of shape ``(n_active,)``.
            bundle: Covariance bundle whose ``matrix`` field is an
                ``(n_active, n_active)`` covariance matrix for the active assets.
            denom_tol: Tolerance threshold for the normalisation denominator.

        Returns:
            SolveYield: A degenerate or valid ``(i, t, mask, pos, status)`` tuple.
        """
        matrix = bundle.matrix
        try:
            denom = inv_a_norm(expected_mu, matrix)
        except SingularMatrixError:
            denom = float("nan")
        try:
            pos = solve(matrix, expected_mu)
        except SingularMatrixError:
            return i, t, mask, np.zeros_like(expected_mu), SolveStatus.DEGENERATE
        return _SolveMixin._denom_guard_yield(i, t, mask, expected_mu, pos, denom, denom_tol)

    def _replay_positions(
        self: _EngineProtocol,
        risk_pos_np: np.ndarray,
        cash_pos_np: np.ndarray,
        vola_np: np.ndarray,
    ) -> None:
        """Replay positions across all rows, filling position arrays.

        Iterates `_iter_solve`, writes risk and cash positions into the
        provided pre-allocated arrays.  Both arrays are mutated **in-place**.

        When `max_turnover` is set, the L1 norm of the
        position change ``sum(|x_t - x_{t-1}|)`` is capped at that value by
        proportionally scaling the delta toward the previous position before
        writing to ``cash_pos_np``.

        Args:
            risk_pos_np: Pre-allocated ``(T, N)`` array for risk positions.
            cash_pos_np: Pre-allocated ``(T, N)`` array for cash positions.
            vola_np: ``(T, N)`` EWMA volatility array.
        """
        max_to: float | None = self.cfg.max_turnover
        for i, _t, mask, pos, _status in self._iter_solve():
            if pos is not None:
                new_cash = _SolveMixin._scale_to_cash(pos, vola_np[i, mask])
                if max_to is not None and i > 0:
                    new_cash = _SolveMixin._apply_turnover_constraint(new_cash, cash_pos_np[i - 1, mask], max_to)
                risk_pos_np[i, mask] = new_cash * vola_np[i, mask]
                cash_pos_np[i, mask] = new_cash

    def _iter_matrices(self: _EngineProtocol) -> Generator[MatrixYield, None, None]:
        r"""Yield ``(i, t, mask, bundle)`` for every timestamp.

        ``bundle`` is a `MatrixBundle` wrapping the effective
        $(n_{\text{sub}},\ n_{\text{sub}})$ correlation matrix for the
        active assets (those with finite prices at timestamp *t*).  Yields
        ``None`` when no valid matrix is available (e.g., before the warm-up
        period has elapsed or when no assets have finite prices).

        The behaviour depends on `covariance_config`:

        * `EwmaShrinkConfig`:  Applies `shrink2id` to
          the EWMA correlation matrix (same computation as
          `cash_position`).
        * `SlidingWindowConfig`: Builds a
          `FactorModel` from the last
          ``cfg.covariance_config.window`` rows of vol-adjusted returns and returns its
          `covariance`.

        Yields:
            tuple: ``(i, t, mask, bundle)`` where

            * ``i`` (*int*): Row index into ``self.prices``.
            * ``t``: Timestamp value from ``self.prices["date"]``.
            * ``mask`` (*np.ndarray[bool]*): Shape ``(n_assets,)``; ``True``
              for assets with finite prices at row *i*.
            * ``bundle`` (`MatrixBundle` | ``None``): Covariance bundle
              of shape ``(mask.sum(), mask.sum())``, or ``None``.
        """
        prices_num = self.prices.select(self.assets).to_numpy()
        dates = self.prices["date"].to_list()

        if isinstance(self.cfg.covariance_config, EwmaShrinkConfig):
            yield from _SolveMixin._iter_matrices_ewma(self, prices_num, dates)
        else:
            yield from _SolveMixin._iter_matrices_sliding(self, prices_num, dates)

    def _iter_matrices_ewma(
        self: _EngineProtocol,
        prices_num: np.ndarray,
        dates: list[datetime.date],
    ) -> Generator[MatrixYield, None, None]:
        """Yield per-timestamp `MatrixYield` for the `EwmaShrinkConfig` path.

        Applies `shrink2id` to the EWMA correlation matrix and restricts it to
        the active-asset sub-matrix; yields ``None`` when no asset has a finite
        price at that row.
        """
        cor = self.cor
        for i, t in enumerate(dates):
            mask = _SolveMixin._compute_mask(prices_num[i])
            if not mask.any():
                yield i, t, mask, None
                continue
            corr_n = cor[t]
            matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]
            yield i, t, mask, MatrixBundle(matrix=matrix)

    def _iter_matrices_sliding(
        self: _EngineProtocol,
        prices_num: np.ndarray,
        dates: list[datetime.date],
    ) -> Generator[MatrixYield, None, None]:
        """Yield per-timestamp `MatrixYield` for the `SlidingWindowConfig` path.

        Fits a `FactorModel` from the last ``window`` rows of vol-adjusted
        returns; yields ``None`` during warm-up, when no asset has a finite
        price, or when the factor-model fit fails.
        """
        sw_config = cast(SlidingWindowConfig, self.cfg.covariance_config)
        win_w: int = sw_config.window
        win_k: int = sw_config.n_factors
        ret_adj_np = self.ret_adj.select(self.assets).to_numpy()
        for i, t in enumerate(dates):
            mask = _SolveMixin._compute_mask(prices_num[i])
            if not mask.any() or i + 1 < win_w:
                yield i, t, mask, None
                continue
            window_ret = ret_adj_np[i + 1 - win_w : i + 1][:, mask]
            window_ret = np.where(np.isfinite(window_ret), window_ret, 0.0)
            n_sub = int(mask.sum())
            k_eff = min(win_k, win_w, n_sub)
            try:
                fm = FactorModel.from_returns(window_ret, k=k_eff)
                yield i, t, mask, MatrixBundle(matrix=fm.covariance)
            except (np.linalg.LinAlgError, ValueError) as exc:
                _logger.warning("Factor model fit failed at t=%s: %s", t, exc)
                yield i, t, mask, None

    @staticmethod
    def _batched_solve_group(
        group: list[tuple[int, datetime.date, np.ndarray, np.ndarray, np.ndarray]],
        denom_tol: float,
    ) -> dict[int, SolveYield]:
        """Solve a batch of linear systems sharing the same active-asset mask.

        Stacks the ``len(group)`` systems into a ``(G, n, n)`` coefficient tensor
        and a ``(G, n)`` right-hand-side matrix, then dispatches a single
        ``numpy.linalg.solve`` call (which maps to a single batched LAPACK
        routine).  Denominators are computed directly from the batch result as
        ``sqrt(mu_i · pos_i)`` — algebraically identical to the per-row
        `inv_a_norm` call.

        Falls back to row-by-row `_compute_position` when
        ``numpy.linalg.solve`` raises ``LinAlgError`` (any matrix in the batch
        is singular).

        Args:
            group: List of ``(i, t, mask, expected_mu, matrix)`` tuples; all
                entries share the same boolean mask and therefore the same
                ``n_active x n_active`` matrix shape.
            denom_tol: Passed through to `_denom_guard_yield`.

        Returns:
            dict: Mapping from row index ``i`` to its `SolveYield`.
        """
        results: dict[int, SolveYield] = {}
        a_stack = np.stack([row[4] for row in group])  # (G, n, n)
        mu_stack = np.stack([row[3] for row in group])  # (G, n)

        try:
            # numpy.linalg.solve requires the RHS to be (..., M, K) when a is (..., M, M).
            # Reshape mu_stack from (G, n) → (G, n, 1) so core dims match, then squeeze.
            pos_stack = np.linalg.solve(a_stack, mu_stack[..., np.newaxis])[..., 0]  # (G, n)
        except np.linalg.LinAlgError:
            # At least one matrix is singular — fall back to sequential per-row solve.
            return _SolveMixin._sequential_solve_group(group, denom_tol)

        # Denominators: sqrt(mu_i^T A_i^{-1} mu_i) = sqrt(mu_i · pos_i).
        dots = (mu_stack * pos_stack).sum(axis=1)  # (G,)
        denoms = np.where(dots > 0.0, np.sqrt(dots), np.nan)

        for (i, t, mask, expected_mu, _matrix), pos, denom in zip(group, pos_stack, denoms, strict=True):
            results[i] = _SolveMixin._denom_guard_yield(i, t, mask, expected_mu, pos, float(denom), denom_tol)

        return results

    @staticmethod
    def _sequential_solve_group(
        group: list[tuple[int, datetime.date, np.ndarray, np.ndarray, np.ndarray]],
        denom_tol: float,
    ) -> dict[int, SolveYield]:
        """Row-by-row fallback used when a batched solve hits a singular matrix.

        Solves each system in *group* independently via `_compute_position` so a
        single ill-conditioned matrix does not abort the whole batch.
        """
        return {
            i: _SolveMixin._compute_position(i, t, mask, expected_mu, MatrixBundle(matrix=matrix), denom_tol)
            for i, t, mask, expected_mu, matrix in group
        }

    @staticmethod
    def _iter_solve_ewma_batched(
        mu_np: np.ndarray,
        matrix_yields: list[MatrixYield],
        denom_tol: float,
    ) -> Generator[SolveYield, None, None]:
        r"""Vectorised EwmaShrink solve: batch ``numpy.linalg.solve`` across timestamps.

        Groups rows by their boolean asset mask so all systems within a group
        share the same ``(n_active, n_active)`` shape, then stacks them into a
        ``(G, n, n)`` tensor and calls ``numpy.linalg.solve`` once per unique
        mask pattern.  Results are collected in a dict and yielded in original
        row order.

        Denominators are derived from the batch solution as
        $\sqrt{\mu_i \cdot \mathbf{pos}_i} = \sqrt{\mu_i^\top \Sigma_i^{-1} \mu_i}$,
        matching the scalar `inv_a_norm` result up
        to float64 rounding.

        Any group whose batch solve raises ``LinAlgError`` (singular matrix in
        the batch) falls back to sequential `_compute_position` for that
        group only.

        Args:
            mu_np: Signal matrix, shape ``(T, n_assets)``.
            matrix_yields: Pre-collected list from `_iter_matrices`
                (the EwmaShrinkConfig branch).
            denom_tol: Denominator guard tolerance.

        Yields:
            `SolveYield` tuples in original row order.
        """
        # First pass: categorise each row as early-exit or a solve candidate.
        all_results, solve_groups = _SolveMixin._partition_ewma_rows(mu_np, matrix_yields)

        # Second pass: batch-solve each mask group.
        for group in solve_groups.values():
            all_results.update(_SolveMixin._batched_solve_group(group, denom_tol))

        # Yield in original row order.
        for i in range(len(matrix_yields)):
            if i in all_results:
                yield all_results[i]

    @staticmethod
    def _partition_ewma_rows(
        mu_np: np.ndarray,
        matrix_yields: list[MatrixYield],
    ) -> tuple[
        dict[int, SolveYield],
        dict[bytes, list[tuple[int, datetime.date, np.ndarray, np.ndarray, np.ndarray]]],
    ]:
        """Split rows into resolved early-exits and mask-grouped solve candidates.

        Returns ``(early_results, solve_groups)`` where ``early_results`` maps a
        row index to its final `SolveYield` (no-data / warmup or an early
        mask/signal exit) and ``solve_groups`` maps each ``mask.tobytes()`` key
        to the rows sharing that active-asset pattern.
        """
        early_results: dict[int, SolveYield] = {}
        # mask.tobytes() → list of (i, t, mask, expected_mu, matrix)
        solve_groups: dict[bytes, list[tuple[int, datetime.date, np.ndarray, np.ndarray, np.ndarray]]] = {}

        for i, t, mask, bundle in matrix_yields:
            if bundle is None:
                early_results[i] = (i, t, mask, np.zeros(int(mask.sum())), SolveStatus.DEGENERATE)
                continue
            expected_mu, early = _SolveMixin._row_early_check(i, t, mask, mu_np[i])
            if early is not None:
                early_results[i] = early
                continue
            solve_groups.setdefault(mask.tobytes(), []).append((i, t, mask, expected_mu, bundle.matrix))

        return early_results, solve_groups

    def _iter_solve(self: _EngineProtocol) -> Generator[SolveYield, None, None]:
        r"""Yield ``(i, t, mask, pos_or_none, status)`` for every timestamp.

        Iterates `_iter_matrices` for the per-row covariance sub-matrix,
        then applies `_row_early_check` (mask/signal guard) and
        `_compute_position` (linear solve and denominator guard).  The two
        covariance modes differ only in how ``matrix`` is built, which
        `_iter_matrices` already encapsulates.

        * ``matrix is None`` → `WARMUP` (sliding-window before
          sufficient history) or `DEGENERATE` otherwise.
        * Signal all-zero → `ZERO_SIGNAL`.
        * Singular or degenerate solve → `DEGENERATE`.
        * Success → `VALID`.

        For the `EwmaShrinkConfig` path the solve step is
        vectorised: rows are grouped by their active-asset mask pattern and each
        group is solved via a single batched ``numpy.linalg.solve`` call (see
        `_iter_solve_ewma_batched`).  The `SlidingWindowConfig`
        path retains a sequential per-row solve because the factor-model matrices
        are constructed lazily and may vary in numerical character across rows.

        .. note::

            **Dual-path maintenance obligation**: this method dispatches to two
            fundamentally different implementations.  Any change to solve
            semantics — a new edge case, a new `SolveStatus` value, or a
            change to denominator logic — **must be applied to both branches**:

            * `_iter_solve_ewma_batched` / `_batched_solve_group`
              (EwmaShrink vectorised path)
            * The sequential ``_compute_position`` loop below
              (SlidingWindow path)

            The cross-path numerical consistency test
            ``test_ewma_batch_and_sequential_paths_agree`` in
            ``tests/test_math/test_numerical_regression.py`` will fail
            whenever the two paths drift apart, surfacing the divergence
            before it reaches production.

        Yields:
            SolveYield: ``(i, t, mask, pos_or_none, status)`` — see
            `SolveYield` for detailed field descriptions.
        """
        mu_np = self.mu.select(self.assets).to_numpy()
        cov_config = self.cfg.covariance_config

        if not isinstance(cov_config, SlidingWindowConfig):
            # EwmaShrinkConfig path: vectorised batch solve grouped by mask pattern.
            yield from _SolveMixin._iter_solve_ewma_batched(mu_np, list(self._iter_matrices()), self.cfg.denom_tol)
            return

        # SlidingWindowConfig path: sequential per-row solve (lazy factor models).
        yield from _SolveMixin._iter_solve_sliding(self, mu_np, cov_config.window)

    def _iter_solve_sliding(
        self: _EngineProtocol,
        mu_np: np.ndarray,
        win_w: int,
    ) -> Generator[SolveYield, None, None]:
        """Sequential per-row solve for the `SlidingWindowConfig` path.

        Factor-model matrices are constructed lazily and may vary in numerical
        character across rows, so each row is solved individually via
        `_compute_position` rather than batched.
        """
        for i, t, mask, bundle in self._iter_matrices():
            if bundle is None:
                yield _SolveMixin._sliding_warmup_or_degenerate(i, t, mask, win_w)
                continue
            expected_mu, early = _SolveMixin._row_early_check(i, t, mask, mu_np[i])
            if early is not None:
                yield early
                continue
            yield _SolveMixin._compute_position(i, t, mask, expected_mu, bundle, self.cfg.denom_tol)

    def warmup_state(self: _EngineProtocol) -> WarmupState:
        """Return the final `WarmupState` after replaying the full batch.

        Encapsulates the position replay loop that was previously duplicated
        inside `from_warmup`.  By centralising the loop
        here, `from_warmup` no longer needs to call the
        private `_iter_solve` generator directly.

        Returns:
            WarmupState: A frozen dataclass with:

            * ``prev_cash_pos`` - cash-position vector at the last row,
              shape ``(n_assets,)``.

        Examples:
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math import BasanosConfig, BasanosEngine
            >>> rng = np.random.default_rng(0)
            >>> dates = list(range(30))
            >>> prices = pl.DataFrame({
            ...     "date": dates,
            ...     "A": np.cumprod(1 + rng.normal(0.001, 0.02, 30)) * 100.0,
            ...     "B": np.cumprod(1 + rng.normal(0.001, 0.02, 30)) * 150.0,
            ... })
            >>> mu = pl.DataFrame({
            ...     "date": dates,
            ...     "A": rng.normal(0, 0.5, 30),
            ...     "B": rng.normal(0, 0.5, 30),
            ... })
            >>> cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> ws = engine.warmup_state()
            >>> ws.prev_cash_pos.shape
            (2,)
        """
        assets = self.assets
        n_rows = self.prices.height
        vola_np = self.vola.select(assets).to_numpy()

        risk_pos_np = np.full((n_rows, len(assets)), np.nan, dtype=float)
        cash_pos_np = np.full((n_rows, len(assets)), np.nan, dtype=float)

        _SolveMixin._replay_positions(self, risk_pos_np, cash_pos_np, vola_np)
        prev_cash_pos = cash_pos_np[-1].copy()
        return WarmupState(prev_cash_pos=prev_cash_pos)
