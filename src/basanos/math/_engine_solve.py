"""Solve/position mixin for BasanosEngine.

This private module contains :class:`_SolveMixin`, which provides the
``_iter_matrices`` and ``_iter_solve`` generator methods.  Separating them
from :mod:`basanos.math.optimizer` keeps the engine facade lean and makes
the per-timestamp solve logic independently readable and testable.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from ..exceptions import SingularMatrixError
from ._config import EwmaShrinkConfig, SlidingWindowConfig
from ._ewm_corr import CorrIirState, _ewm_corr_iir_zf
from ._factor_model import FactorModel
from ._linalg import inv_a_norm, solve
from ._signal import shrink2id

if TYPE_CHECKING:
    from ._engine_protocol import _EngineProtocol

_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class WarmupState:
    """Final state produced by a full batch solve; consumed by :meth:`BasanosStream.from_warmup`.

    Returned by :meth:`BasanosEngine.warmup_state` and used by
    :meth:`BasanosStream.from_warmup` to initialise the streaming state without
    coupling to the private :meth:`~_SolveMixin._iter_solve` generator.

    Attributes:
        profit_variance: Final EWMA profit-variance scalar after replaying the
            full warmup batch.
        prev_cash_pos: Cash positions at the last warmup row, shape
            ``(n_assets,)``.  ``NaN`` for assets that were still in their
            own warmup period.
        corr_iir_state: Final IIR filter states for the four EWM correlation
            accumulators, produced by :func:`~basanos.math._ewm_corr._ewm_corr_iir_zf`.
            ``None`` when the engine uses :class:`~basanos.math.SlidingWindowConfig`
            (streaming is only supported for :class:`~basanos.math.EwmaShrinkConfig`).
    """

    profit_variance: float
    prev_cash_pos: np.ndarray
    corr_iir_state: CorrIirState | None = None


class _SolveMixin:
    """Mixin that provides ``_iter_matrices`` and ``_iter_solve`` generators.

    Consumers must also inherit from (or satisfy the interface of)
    :class:`~basanos.math._engine_protocol._EngineProtocol` so that
    ``self.assets``, ``self.prices``, ``self.mu``, ``self.cfg``, ``self.cor``,
    and ``self.ret_adj`` are all available.
    """

    def _iter_matrices(self: _EngineProtocol):
        r"""Yield ``(i, t, mask, matrix)`` for every timestamp.

        ``matrix`` is the effective :math:`(n_{\text{sub}},\ n_{\text{sub}})`
        correlation matrix for the active assets (those with finite prices at
        timestamp *t*).  Yields ``None`` when no valid matrix is available
        (e.g., before the warm-up period has elapsed or when no assets have
        finite prices).

        The behaviour depends on :attr:`BasanosConfig.covariance_config`:

        * :class:`EwmaShrinkConfig`:  Applies :func:`~basanos.math._signal.shrink2id` to
          the EWMA correlation matrix (same computation as
          :attr:`cash_position`).
        * :class:`SlidingWindowConfig`: Builds a
          :class:`~basanos.math._factor_model.FactorModel` from the last
          ``cfg.covariance_config.window`` rows of vol-adjusted returns and returns its
          :attr:`~basanos.math._factor_model.FactorModel.covariance`.

        Yields:
            tuple: ``(i, t, mask, matrix)`` where

            * ``i`` (*int*): Row index into ``self.prices``.
            * ``t``: Timestamp value from ``self.prices["date"]``.
            * ``mask`` (*np.ndarray[bool]*): Shape ``(n_assets,)``; ``True``
              for assets with finite prices at row *i*.
            * ``matrix`` (*np.ndarray | None*): Shape
              ``(mask.sum(), mask.sum())`` or ``None``.
        """
        assets = self.assets
        prices_num = self.prices.select(assets).to_numpy()
        dates = self.prices["date"].to_list()

        if isinstance(self.cfg.covariance_config, EwmaShrinkConfig):
            cor = self.cor
            for i, t in enumerate(dates):
                mask = np.isfinite(prices_num[i])
                if not mask.any():
                    yield i, t, mask, None
                    continue
                corr_n = cor[t]
                matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]
                yield i, t, mask, matrix
        else:
            sw_config = cast(SlidingWindowConfig, self.cfg.covariance_config)
            win_w: int = sw_config.window
            win_k: int = sw_config.n_factors
            ret_adj_np = self.ret_adj.select(assets).to_numpy()
            for i, t in enumerate(dates):
                mask = np.isfinite(prices_num[i])
                if not mask.any() or i + 1 < win_w:
                    yield i, t, mask, None
                    continue
                window_ret = ret_adj_np[i + 1 - win_w : i + 1][:, mask]
                window_ret = np.where(np.isfinite(window_ret), window_ret, 0.0)
                n_sub = int(mask.sum())
                k_eff = min(win_k, win_w, n_sub)
                try:
                    fm = FactorModel.from_returns(window_ret, k=k_eff)
                    yield i, t, mask, fm.covariance
                except (np.linalg.LinAlgError, ValueError) as exc:
                    _logger.warning("Factor model fit failed at t=%s: %s", t, exc)
                    yield i, t, mask, None

    def _iter_solve(self: _EngineProtocol):
        r"""Yield ``(i, t, mask, pos_or_none, status)`` for every timestamp.

        This is the single authoritative implementation of the per-timestamp
        position-determination logic shared by :attr:`cash_position` and
        :attr:`position_status`.  Extracting it here eliminates the DRY
        violation where both properties previously duplicated the mask
        computation, covariance dispatch, :math:`\mu` handling, denominator
        check, and linear-solve logic.

        Yields:
            tuple: ``(i, t, mask, pos_or_none, status)`` where

            * ``i`` (*int*): Row index into ``self.prices``.
            * ``t``: Timestamp value from ``self.prices["date"]``.
            * ``mask`` (*np.ndarray[bool]*): Shape ``(n_assets,)``; ``True``
              for assets with finite prices at row *i*.
            * ``pos_or_none`` (*np.ndarray | None*): Per-active-asset position
              vector **before** ``profit_variance`` scaling.  The value and its
              downstream effect depend on ``status`` as follows:

              .. list-table::
                 :header-rows: 1

                 * - ``status``
                   - ``pos_or_none``
                   - Downstream effect in :attr:`cash_position`
                 * - ``'warmup'``
                   - ``None``
                   - Positions stay ``NaN`` — insufficient history.
                 * - ``'zero_signal'``
                   - ``np.zeros(n_active)``
                   - Positions written as ``0`` for all active assets.
                 * - ``'degenerate'``
                   - ``np.zeros(n_active)`` (empty when no prices)
                   - Positions written as ``0`` for active assets; rows with
                     no finite prices have an empty mask so all positions
                     remain ``NaN`` as a natural consequence.
                 * - ``'valid'``
                   - ``np.ndarray`` of shape ``(n_active,)``
                   - Solved positions written for all active assets.

              ``None`` is yielded **only** for ``'warmup'`` rows.  All other
              statuses yield an ``np.ndarray`` (possibly zero-length when
              ``mask`` is all-``False``), so consumers can branch solely on
              ``pos_or_none is None`` to detect the warmup case without
              inspecting ``status``.

            * ``status`` (*str*): One of ``'warmup'``, ``'zero_signal'``,
              ``'degenerate'``, or ``'valid'``.
        """
        assets = self.assets
        prices_num = self.prices.select(assets).to_numpy()
        mu_np = self.mu.select(assets).to_numpy()
        dates = self.prices["date"].to_list()

        if isinstance(self.cfg.covariance_config, EwmaShrinkConfig):
            cor = self.cor
            for i, t in enumerate(dates):
                mask = np.isfinite(prices_num[i])
                if not mask.any():
                    yield i, t, mask, np.zeros(0), "degenerate"
                    continue
                corr_n = cor[t]
                matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]
                expected_mu = np.nan_to_num(mu_np[i][mask])
                if np.allclose(expected_mu, 0.0):
                    yield i, t, mask, np.zeros_like(expected_mu), "zero_signal"
                    continue
                try:
                    denom = inv_a_norm(expected_mu, matrix)
                except SingularMatrixError:
                    denom = float("nan")
                if not np.isfinite(denom) or denom <= self.cfg.denom_tol:
                    _logger.warning(
                        "Positions zeroed at t=%s: normalisation denominator is degenerate "
                        "(denom=%s, denom_tol=%s). Check signal magnitude and covariance matrix.",
                        t,
                        denom,
                        self.cfg.denom_tol,
                        extra={
                            "context": {
                                "t": str(t),
                                "denom": denom,
                                "denom_tol": self.cfg.denom_tol,
                            }
                        },
                    )
                    yield i, t, mask, np.zeros_like(expected_mu), "degenerate"
                    continue
                try:
                    pos = solve(matrix, expected_mu) / denom
                except SingularMatrixError:
                    yield i, t, mask, np.zeros_like(expected_mu), "degenerate"
                    continue
                yield i, t, mask, pos, "valid"
        else:
            sw_config = cast(SlidingWindowConfig, self.cfg.covariance_config)
            win_w: int = sw_config.window
            win_k: int = sw_config.n_factors
            ret_adj_np = self.ret_adj.select(assets).to_numpy()
            for i, t in enumerate(dates):
                mask = np.isfinite(prices_num[i])
                if not mask.any():
                    yield i, t, mask, np.zeros(0), "degenerate"
                    continue
                if i + 1 < win_w:
                    yield i, t, mask, None, "warmup"
                    continue
                window_ret = ret_adj_np[i + 1 - win_w : i + 1][:, mask]
                window_ret = np.where(np.isfinite(window_ret), window_ret, 0.0)
                n_sub = int(mask.sum())
                k_eff = min(win_k, win_w, n_sub)
                try:
                    fm = FactorModel.from_returns(window_ret, k=k_eff)
                except (np.linalg.LinAlgError, ValueError) as exc:
                    _logger.debug("Sliding window SVD failed at t=%s: %s", t, exc)
                    yield i, t, mask, np.zeros(n_sub), "degenerate"
                    continue
                expected_mu = np.nan_to_num(mu_np[i][mask])
                if np.allclose(expected_mu, 0.0):
                    yield i, t, mask, np.zeros(n_sub), "zero_signal"
                    continue
                try:
                    x = fm.solve(expected_mu)
                    denom = float(np.sqrt(max(0.0, float(np.dot(expected_mu, x)))))
                except (np.linalg.LinAlgError, ValueError) as exc:
                    _logger.warning("Woodbury solve failed at t=%s: %s", t, exc)
                    yield i, t, mask, np.zeros(n_sub), "degenerate"
                    continue
                if not np.isfinite(denom) or denom <= self.cfg.denom_tol:
                    _logger.warning(
                        "Positions zeroed at t=%s (sliding_window): normalisation "
                        "denominator is degenerate (denom=%s, denom_tol=%s).",
                        t,
                        denom,
                        self.cfg.denom_tol,
                    )
                    yield i, t, mask, np.zeros(n_sub), "degenerate"
                    continue
                yield i, t, mask, x / denom, "valid"

    def warmup_state(self: _EngineProtocol) -> WarmupState:
        """Return the final :class:`WarmupState` after replaying the full batch.

        Encapsulates the profit-variance EMA replay loop that was previously
        duplicated inside :meth:`BasanosStream.from_warmup`.  By centralising
        the loop here, :meth:`~BasanosStream.from_warmup` no longer needs to
        call the private :meth:`_iter_solve` generator directly.

        Returns:
            WarmupState: A frozen dataclass with:

            * ``profit_variance`` - final EWMA profit-variance scalar.
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
            >>> isinstance(ws.profit_variance, float)
            True
            >>> ws.prev_cash_pos.shape
            (2,)
        """
        assets = self.assets
        n_assets = len(assets)
        n_rows = self.prices.height
        prices_np = self.prices.select(assets).to_numpy()
        vola_np = self.vola.select(assets).to_numpy()

        returns_num = np.zeros((n_rows, n_assets), dtype=float)
        if n_rows > 1:
            returns_num[1:] = prices_np[1:] / prices_np[:-1] - 1.0

        profit_variance: float = self.cfg.profit_variance_init
        lamb: float = self.cfg.profit_variance_decay

        risk_pos_np = np.full((n_rows, n_assets), np.nan, dtype=float)
        cash_pos_np = np.full((n_rows, n_assets), np.nan, dtype=float)

        for i, _t, mask, pos, _status in self._iter_solve():
            if i > 0:
                ret_mask = np.isfinite(returns_num[i]) & mask
                if ret_mask.any():
                    with np.errstate(invalid="ignore"):
                        cash_pos_np[i - 1] = risk_pos_np[i - 1] / vola_np[i - 1]
                    lhs = np.nan_to_num(cash_pos_np[i - 1, ret_mask], nan=0.0)
                    rhs = np.nan_to_num(returns_num[i, ret_mask], nan=0.0)
                    profit = float(lhs @ rhs)
                    profit_variance = lamb * profit_variance + (1 - lamb) * profit**2
            if pos is not None:
                risk_pos_np[i, mask] = pos / profit_variance
                with np.errstate(invalid="ignore"):
                    cash_pos_np[i, mask] = risk_pos_np[i, mask] / vola_np[i, mask]

        prev_cash_pos = cash_pos_np[-1].copy()
        return WarmupState(profit_variance=profit_variance, prev_cash_pos=prev_cash_pos)
