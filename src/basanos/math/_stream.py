"""Incremental (streaming) API for BasanosEngine.

This private module defines the `BasanosStream` façade: an incremental
optimiser with a `from_warmup` classmethod and a `step` method.  After warming
up on a historical batch, each `step` call advances the internal state by
exactly one row in O(N^2) time without revisiting the warmup history.

The supporting pieces live in sibling modules so this file stays focused on the
streaming loop itself:

* :mod:`basanos.math._stream_state` — the mutable `_StreamState` carrier and the
  frozen `StepResult` output.
* :mod:`basanos.math._stream_math` — the pure EWMA recurrences and input
  validation helpers.
* :mod:`basanos.math._stream_solve` — the per-step position solvers for the
  EWM and sliding-window covariance modes.
* :mod:`basanos.math._stream_io` — save/load of the full stream state.

EWM correlation state model
----------------------------
In EWM mode the correlation at each step is recomputed by calling
``ewm_covariance`` from ``cvx.linalg`` over the full growing history of
vol-adjusted returns stored in ``corr_ret_buf``.  This keeps the incremental
and batch paths numerically identical at the cost of O(T·N²) time per step
(acceptable for small N or short warmup histories).

The volatility accumulators (``vola_*``, ``pct_*``) use a simpler scalar
recurrence and store the running sums directly as ``(N,)`` arrays.

Memory
------
Total incremental state is O(T·N) for the growing history buffer plus
8x(N,) + O(1) scalars.  For the SlidingWindowConfig the buffer is a fixed
(W, N) array independent of T.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, cast

import numpy as np
import polars as pl

from ..exceptions import MissingDateColumnError
from ._config import BasanosConfig, EwmaShrinkConfig, SlidingWindowConfig
from ._engine_solve import SolveStatus, _SolveMixin
from ._stream_io import load_stream_archive, save_stream_archive
from ._stream_math import _ewm_std_from_state, _ewm_vol_accumulators_from_batch, _resolve_step_vector
from ._stream_solve import solve_ewma_position, solve_sliding_window_position
from ._stream_state import StepResult as StepResult
from ._stream_state import _StreamState
from .optimizer import BasanosEngine


class BasanosStream:
    """Incremental (streaming) optimiser backed by a single `_StreamState`.

    After warming up on a historical batch via `from_warmup`, each call
    to `step` advances the internal state by exactly one row in
    O(N^2) time — without revisiting the full warmup history.

    Attributes:
        assets: Ordered list of asset column names (read-only).

    Examples:
        >>> import numpy as np
        >>> import polars as pl
        >>> from datetime import date, timedelta
        >>> from basanos.math import BasanosConfig, BasanosStream
        >>> rng = np.random.default_rng(0)
        >>> warmup_len = 60
        >>> dates = pl.date_range(
        ...     start=date(2024, 1, 1),
        ...     end=date(2024, 1, 1) + timedelta(days=warmup_len),
        ...     interval="1d",
        ...     eager=True,
        ... )
        >>> prices = pl.DataFrame({
        ...     "date": dates,
        ...     "A": np.cumprod(1 + rng.normal(0.001, 0.02, warmup_len + 1)) * 100.0,
        ...     "B": np.cumprod(1 + rng.normal(0.001, 0.02, warmup_len + 1)) * 150.0,
        ... })
        >>> mu = pl.DataFrame({
        ...     "date": dates,
        ...     "A": rng.normal(0, 0.5, warmup_len + 1),
        ...     "B": rng.normal(0, 0.5, warmup_len + 1),
        ... })
        >>> cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
        >>> stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
        >>> result = stream.step(
        ...     prices.select(["A", "B"]).to_numpy()[warmup_len],
        ...     mu.select(["A", "B"]).to_numpy()[warmup_len],
        ...     prices["date"][warmup_len],
        ... )
        >>> isinstance(result, StepResult)
        True
        >>> result.cash_position.shape
        (2,)
    """

    _cfg: BasanosConfig
    _assets: list[str]
    _state: _StreamState

    def __init__(self, cfg: BasanosConfig, assets: list[str], state: _StreamState) -> None:
        """Initialise from an explicit config, asset list, and state container."""
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_assets", assets)
        object.__setattr__(self, "_state", state)

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent accidental attribute mutation — BasanosStream is immutable."""
        raise dataclasses.FrozenInstanceError(f"{type(self).__name__}.{name}")

    @property
    def assets(self) -> list[str]:
        """Ordered list of asset column names."""
        return self._assets

    # ------------------------------------------------------------------
    # from_warmup
    # ------------------------------------------------------------------

    @classmethod
    def from_warmup(
        cls,
        prices: pl.DataFrame,
        mu: pl.DataFrame,
        cfg: BasanosConfig,
    ) -> BasanosStream:
        """Build a `BasanosStream` from a historical warmup batch.

        Runs `BasanosEngine` on the full warmup batch
        exactly once and extracts the minimal IIR-filter state required for
        subsequent `step` calls.  After this call, each `step`
        advances the optimiser in O(N^2) time without touching the warmup
        data again.

        Parameters
        ----------
        prices:
            Historical price DataFrame.  Must contain a ``'date'`` column and
            at least one numeric asset column with strictly positive,
            non-monotonic values.
        mu:
            Expected-return signal DataFrame aligned row-by-row with
            ``prices``.
        cfg:
            Engine configuration.  Both `EwmaShrinkConfig`
            and `SlidingWindowConfig` are supported.

        Returns:
        -------
        BasanosStream
            A stream instance whose `step` method is ready to accept the
            row immediately following the last warmup row.

        Notes:
        ------
        **Short-warmup behaviour with** ``SlidingWindowConfig``: when
        ``len(prices) < cfg.covariance_config.window``, the internal rolling
        buffer (``sw_ret_buf``) is NaN-padded for the missing prefix rows.
        `step` returns ``StepResult(status="warmup")`` for each of the
        first ``window - len(prices)`` calls, exactly matching the EWM warmup
        semantics.  By the time `step` returns the first non-warmup
        result the buffer contains only real data — no NaN-padded rows remain.

        Raises:
        ------
        MissingDateColumnError
            If ``'date'`` is absent from ``prices``.
        """
        # 1. Validate -------------------------------------------------------
        if "date" not in prices.columns:
            raise MissingDateColumnError("prices")

        # 2. Build the engine on the full warmup batch ----------------------
        engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
        assets = engine.assets
        n_assets = len(assets)
        n_rows = prices.height
        prices_np = prices.select(assets).to_numpy()  # (n_rows, n_assets)

        # 3. Extract mode-specific state from WarmupState --------------------
        ws = engine.warmup_state()
        if isinstance(cfg.covariance_config, EwmaShrinkConfig):
            # EWM: seed the growing history buffer from engine.ret_adj so that
            # each subsequent step() can call ewm_covariance over the full history.
            ret_adj_np = engine.ret_adj.select(assets).to_numpy()
            corr_ret_buf: np.ndarray | None = ret_adj_np
            sw_ret_buf: np.ndarray | None = None
        else:
            # SW: carry the last W vol-adjusted returns as a rolling buffer.
            sw_config = cfg.covariance_config
            win_w = sw_config.window
            ret_adj_np = engine.ret_adj.select(assets).to_numpy()  # (n_rows, N)
            if n_rows >= win_w:
                sw_ret_buf = ret_adj_np[-win_w:].copy()
            else:
                sw_ret_buf = np.full((win_w, n_assets), np.nan)
                sw_ret_buf[-n_rows:] = ret_adj_np
            corr_ret_buf = None

        # 4. Derive EWMA volatility accumulators (vectorised) ---------------
        # Both log-return (for vol_adj) and pct-return (for vola) use the
        # same beta = (vola-1)/vola.  NaN observations (leading NaN at row 0
        # from diff/pct_change) are skipped — the filter input is 0 for NaN
        # rows and the weight accumulator (s_w) only increments for finite
        # observations, matching Polars' effective behaviour for a
        # leading-NaN series.
        #
        # Delegate to the shared helper _ewm_vol_accumulators_from_batch so
        # that the batch and incremental recurrences share a single definition.
        beta_vola: float = (cfg.vola - 1) / cfg.vola
        beta_vola_sq: float = beta_vola**2

        log_ret = np.full((n_rows, n_assets), np.nan, dtype=float)
        pct_ret = np.full((n_rows, n_assets), np.nan, dtype=float)
        if n_rows > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ret[1:] = np.log(prices_np[1:] / prices_np[:-1])
                pct_ret[1:] = prices_np[1:] / prices_np[:-1] - 1.0

        vola_s_x, vola_s_x2, vola_s_w, vola_s_w2, vola_count = _ewm_vol_accumulators_from_batch(
            log_ret, beta_vola, beta_vola_sq
        )
        pct_s_x, pct_s_x2, pct_s_w, pct_s_w2, pct_count = _ewm_vol_accumulators_from_batch(
            pct_ret, beta_vola, beta_vola_sq
        )

        # 5. Extract prev_cash_pos from WarmupState --------------------------
        prev_cash_pos: np.ndarray = ws.prev_cash_pos
        prev_price: np.ndarray = prices_np[-1].copy()

        # 6. Construct _StreamState and return ------------------------------
        state = _StreamState(
            corr_ret_buf=corr_ret_buf,
            vola_s_x=vola_s_x,
            vola_s_x2=vola_s_x2,
            vola_s_w=vola_s_w,
            vola_s_w2=vola_s_w2,
            vola_count=vola_count,
            pct_s_x=pct_s_x,
            pct_s_x2=pct_s_x2,
            pct_s_w=pct_s_w,
            pct_s_w2=pct_s_w2,
            pct_count=pct_count,
            prev_price=prev_price,
            prev_cash_pos=prev_cash_pos,
            step_count=n_rows,
            sw_ret_buf=sw_ret_buf,
        )
        return cls(cfg=cfg, assets=assets, state=state)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    @staticmethod
    def _warmup_threshold(cfg: BasanosConfig) -> int:
        """Return the step count at which warmup ends for the configured mode."""
        if isinstance(cfg.covariance_config, SlidingWindowConfig):
            return cfg.covariance_config.window
        return cfg.corr

    @staticmethod
    def _warmup_result(n_assets: int, date: Any) -> StepResult:
        """Build a standard warmup ``StepResult`` payload."""
        return StepResult(
            date=date,
            cash_position=np.full(n_assets, np.nan),
            status=SolveStatus.WARMUP,
            vola=np.full(n_assets, np.nan),
        )

    def step(
        self,
        new_prices: np.ndarray | dict[str, float],
        new_mu: np.ndarray | dict[str, float],
        date: Any = None,
    ) -> StepResult:
        """Advance the stream by one row and return the new optimised position.

        Parameters
        ----------
        new_prices:
            Per-asset prices for the new timestep.  Either a numpy array of
            shape ``(N,)`` (assets ordered as in `assets`) or a dict
            mapping asset names to price values.
        new_mu:
            Per-asset expected-return signals, same format as ``new_prices``.
        date:
            Timestamp for this step (stored in `date`
            verbatim; not used in any computation).

        Returns:
        -------
        StepResult
            Frozen dataclass with ``cash_position``, ``vola``, ``status``, and
            ``date`` for this timestep.
        """
        cfg = self._cfg
        assets = self._assets
        state = self._state
        n_assets = len(assets)

        # ── Check if still in the warmup period ──────────────────────────────
        # step_count is initialised to n_rows in from_warmup.
        #
        # EwmaShrinkConfig: in_warmup is True for the first (cfg.corr - n_rows)
        # calls when the warmup batch was shorter than cfg.corr (not enough rows
        # to populate the EWM correlation matrix).
        #
        # SlidingWindowConfig: in_warmup is True for the first (window - n_rows)
        # calls when the warmup batch was shorter than the window.  During this
        # period sw_ret_buf still contains NaN-padded prefix rows; each step
        # shifts one NaN out and appends a real row, so the buffer is fully
        # populated with real data exactly when in_warmup becomes False.
        #
        # In both modes all accumulators are still updated during warmup so that
        # the state is ready the moment the warmup period ends.
        _warmup_thresh = self._warmup_threshold(cfg)
        in_warmup: bool = state.step_count < _warmup_thresh

        # ── Resolve inputs to (N,) float64 arrays ──────────────────────────
        new_p = _resolve_step_vector(new_prices, assets, n_assets, "new_prices")
        new_m = _resolve_step_vector(new_mu, assets, n_assets, "new_mu")

        prev_p = state.prev_price
        beta_vola: float = (cfg.vola - 1) / cfg.vola
        beta_vola_sq: float = beta_vola**2

        # ── Compute new log-returns and pct-returns ─────────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                np.isfinite(new_p) & np.isfinite(prev_p) & (prev_p > 0),
                new_p / prev_p,
                np.nan,
            )
            log_ret = np.log(ratio)
            pct_ret = ratio - 1.0

        # ── Update log-return EWMA accumulators ────────────────────────────
        fin_log = np.isfinite(log_ret)
        vola_s_x = beta_vola * state.vola_s_x + np.where(fin_log, log_ret, 0.0)
        vola_s_x2 = beta_vola * state.vola_s_x2 + np.where(fin_log, log_ret**2, 0.0)
        vola_s_w = beta_vola * state.vola_s_w + fin_log.astype(float)
        vola_s_w2 = beta_vola_sq * state.vola_s_w2 + fin_log.astype(float)
        vola_count = state.vola_count + fin_log.astype(int)

        # ── Update pct-return EWMA accumulators ────────────────────────────
        fin_pct = np.isfinite(pct_ret)
        pct_s_x = beta_vola * state.pct_s_x + np.where(fin_pct, pct_ret, 0.0)
        pct_s_x2 = beta_vola * state.pct_s_x2 + np.where(fin_pct, pct_ret**2, 0.0)
        pct_s_w = beta_vola * state.pct_s_w + fin_pct.astype(float)
        pct_s_w2 = beta_vola_sq * state.pct_s_w2 + fin_pct.astype(float)
        pct_count = state.pct_count + fin_pct.astype(int)

        # ── Compute vol-adjusted return (for the correlation IIR input) ─────
        log_vol = _ewm_std_from_state(vola_s_x, vola_s_x2, vola_s_w, vola_s_w2, vola_count, min_samples=1)
        # Divide; std == 0 yields ±inf → clipped to ±cfg.clip (matches Polars)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_adj_val = np.where(
                fin_log,
                np.clip(log_ret / log_vol, -cfg.clip, cfg.clip),
                np.nan,
            )

        # ── Mode-specific correlation state update ───────────────────────────
        if isinstance(cfg.covariance_config, SlidingWindowConfig):
            # SW: shift the rolling window buffer in-place and append this row.
            buf = cast(np.ndarray, state.sw_ret_buf)  # (W, N), already owned by state
            buf[:-1] = buf[1:]
            buf[-1] = vol_adj_val
            corr_ret_buf = state.corr_ret_buf  # None for SW; pass through
        else:
            # EWM: append new vol-adjusted return to the growing history buffer.
            new_row = vol_adj_val[np.newaxis]  # (1, N)
            corr_ret_buf = np.vstack([cast(np.ndarray, state.corr_ret_buf), new_row])

        # ── Early return during EWM warmup period ───────────────────────────
        # All accumulators are already updated above; skip the O(N²) matrix
        # reconstruction and O(N³) Cholesky solve which are wasteful during
        # warmup — the computed positions would be discarded anyway.
        if in_warmup:
            state.persist(
                corr_ret_buf=corr_ret_buf,
                vola_s_x=vola_s_x,
                vola_s_x2=vola_s_x2,
                vola_s_w=vola_s_w,
                vola_s_w2=vola_s_w2,
                vola_count=vola_count,
                pct_s_x=pct_s_x,
                pct_s_x2=pct_s_x2,
                pct_s_w=pct_s_w,
                pct_s_w2=pct_s_w2,
                pct_count=pct_count,
                new_price=new_p,
            )
            return self._warmup_result(n_assets, date)

        # ── Compute EWMA volatility (pct-return std) — shared ───────────────
        vola_vec = _ewm_std_from_state(pct_s_x, pct_s_x2, pct_s_w, pct_s_w2, pct_count, min_samples=cfg.vola)

        # ── Solve for position ───────────────────────────────────────────────
        mask = np.isfinite(new_p)
        if isinstance(cfg.covariance_config, SlidingWindowConfig):
            new_cash_pos, status = solve_sliding_window_position(
                cfg=cfg,
                state=state,
                mask=mask,
                new_m=new_m,
                vola_vec=vola_vec,
                n_assets=n_assets,
                date=date,
            )
        else:
            new_cash_pos, status = solve_ewma_position(
                cfg=cfg,
                state=state,
                corr_ret_buf=cast(np.ndarray, corr_ret_buf),
                mask=mask,
                new_m=new_m,
                vola_vec=vola_vec,
                assets=list(self._assets),
                n_assets=n_assets,
                date=date,
            )

        # ── Apply turnover constraint ─────────────────────────────────────────
        if cfg.max_turnover is not None and status == SolveStatus.VALID:
            new_cash_pos[mask] = _SolveMixin._apply_turnover_constraint(
                new_cash_pos[mask],
                state.prev_cash_pos[mask],
                cfg.max_turnover,
            )

        # ── Persist updated state ───────────────────────────────────────────
        state.persist(
            corr_ret_buf=corr_ret_buf,
            vola_s_x=vola_s_x,
            vola_s_x2=vola_s_x2,
            vola_s_w=vola_s_w,
            vola_s_w2=vola_s_w2,
            vola_count=vola_count,
            pct_s_x=pct_s_x,
            pct_s_x2=pct_s_x2,
            pct_s_w=pct_s_w,
            pct_s_w2=pct_s_w2,
            pct_count=pct_count,
            new_price=new_p,
            new_cash_pos=new_cash_pos,
        )

        return StepResult(
            date=date,
            cash_position=new_cash_pos,
            status=status,
            vola=vola_vec,
        )

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike[str]) -> None:
        """Serialise the stream to a ``.npz`` archive at *path*.

        Delegates to `basanos.math._stream_io.save_stream_archive`.  A stream
        restored via `load` produces bit-for-bit identical `step` output.

        Args:
            path: Destination file path.  ``np.savez`` appends ``.npz``
                automatically when the suffix is absent.
        """
        save_stream_archive(self._cfg, self._assets, self._state, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> BasanosStream:
        """Restore a stream previously saved with `save`.

        Delegates to `basanos.math._stream_io.load_stream_archive`.

        Args:
            path: Path to a ``.npz`` archive written by `save`.

        Returns:
            A `BasanosStream` whose `step` output is bit-for-bit identical to
            the original stream at the time `save` was called.
        """
        cfg, assets, state = load_stream_archive(path)
        return cls(cfg=cfg, assets=assets, state=state)
