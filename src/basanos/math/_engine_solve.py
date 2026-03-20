"""Solve/position mixin for BasanosEngine.

This private module contains :class:`_SolveMixin`, which provides the
``_iter_matrices`` and ``_iter_solve`` generator methods.  Separating them
from :mod:`basanos.math.optimizer` keeps the engine facade lean and makes
the per-timestamp solve logic independently readable and testable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from ..exceptions import SingularMatrixError
from ._config import EwmaShrinkConfig, SlidingWindowConfig
from ._factor_model import FactorModel
from ._linalg import inv_a_norm, solve
from ._signal import shrink2id

if TYPE_CHECKING:
    from ._engine_protocol import _EngineProtocol

_logger = logging.getLogger(__name__)


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
