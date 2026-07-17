"""Foundational types and stateless solve primitives for `_SolveMixin`.

Holds the `SolveStatus` enum, the `MatrixBundle` / `WarmupState` carriers, the
`MatrixYield` / `SolveYield` aliases, and `_SolvePrimitivesMixin` — the pure
(``@staticmethod``) helpers that carry no dependency on the linear-algebra
backend.  Splitting them out of ``_engine_solve`` keeps the solve-orchestration
module focused on the generators that drive `BasanosEngine`.

All public names are re-exported from ``_engine_solve`` so existing imports
(``from basanos.math._engine_solve import MatrixBundle, SolveStatus, ...``) and
the ``solve`` / ``inv_a_norm`` patch targets that live there are unchanged.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
from enum import StrEnum
from typing import TypeAlias, cast

import numpy as np

# Solve-step warnings are emitted under the ``_engine_solve`` logger name (not
# this module's ``__name__``) so the logging contract — log filters and the JSON
# round-trip in the diagnostics — stays stable regardless of how the solve
# helpers are split across private modules.
_logger = logging.getLogger("basanos.math._engine_solve")


class SolveStatus(StrEnum):
    """Solver outcome labels for each timestamp.

    Since `SolveStatus` inherits from `str` via ``StrEnum``,
    values compare equal to their string equivalents (e.g.
    ``SolveStatus.VALID == "valid"``), preserving backward compatibility
    with code that matches on string literals.

    Attributes:
        WARMUP: Insufficient history for the sliding-window covariance mode.
        ZERO_SIGNAL: The expected-return vector was all-zero; positions zeroed.
        DEGENERATE: Normalisation denominator was non-finite, solve failed, or
            no asset had a finite price; positions zeroed for safety.
        VALID: Linear system solved successfully; positions are non-trivially
            non-zero.
    """

    WARMUP = "warmup"
    ZERO_SIGNAL = "zero_signal"
    DEGENERATE = "degenerate"
    VALID = "valid"


@dataclasses.dataclass(frozen=True)
class MatrixBundle:
    """Container for the covariance matrix and any mode-specific auxiliary state.

    Wrapping the covariance matrix in a dataclass decouples
    `_compute_position` from the raw array so that future
    covariance modes (e.g. DCC-GARCH, RMT-cleaned) can carry additional fields
    through the same interface without changing the method signature.

    Attributes:
        matrix: The ``(n_active, n_active)`` covariance sub-matrix for the
            active assets at a given timestamp.
    """

    matrix: np.ndarray


#: Yield type for `_iter_matrices`:
#: ``(i, t, mask, bundle)`` where ``bundle`` is ``None`` during warmup/no-data.
MatrixYield: TypeAlias = tuple[int, datetime.date, np.ndarray, MatrixBundle | None]

#: Yield type for `_iter_solve`:
#: ``(i, t, mask, pos_or_none, status)`` where ``pos_or_none`` is ``None`` only for warmup rows.
SolveYield: TypeAlias = tuple[int, datetime.date, np.ndarray, np.ndarray | None, SolveStatus]


@dataclasses.dataclass(frozen=True)
class WarmupState:
    """Final state produced by a full batch solve; consumed by `from_warmup`.

    Returned by `warmup_state` and used by
    `from_warmup` to initialise the streaming state without
    coupling to the private `_iter_solve` generator.

    Attributes:
        prev_cash_pos: Cash positions at the last warmup row, shape
            ``(n_assets,)``.  ``NaN`` for assets that were still in their
            own warmup period.
    """

    prev_cash_pos: np.ndarray


class _SolvePrimitivesMixin:
    """Stateless solve helpers shared by `_SolveMixin`.

    Every method here is a ``@staticmethod`` that carries no dependency on the
    linear-algebra backend, so it can be reasoned about (and reused) in
    isolation from the per-timestamp solve generators.
    """

    @staticmethod
    def _compute_mask(prices_row: np.ndarray) -> np.ndarray:
        """Return boolean mask indicating which assets have finite prices in the given row."""
        mask: np.ndarray = np.isfinite(prices_row)
        return mask

    @staticmethod
    def _check_signal(mu: np.ndarray, mask: np.ndarray) -> SolveStatus | None:
        """Return ``ZERO_SIGNAL`` when the masked expected-return vector is all-zero.

        Returns ``None`` when the signal is non-trivially non-zero, indicating
        that the caller should proceed to the linear solve.
        """
        if np.allclose(np.nan_to_num(mu[mask]), 0.0):
            return SolveStatus.ZERO_SIGNAL
        return None

    @staticmethod
    def _scale_to_cash(pos: np.ndarray, vola_active: np.ndarray) -> np.ndarray:
        """Convert raw solver positions to cash-adjusted positions.

        Divides *pos* by *vola_active* (volatility for the active asset subset)
        to get cash positions.  ``np.errstate(invalid="ignore")`` is applied
        internally so NaN volatility values propagate quietly.
        """
        with np.errstate(invalid="ignore"):
            return cast("np.ndarray", pos / vola_active)

    @staticmethod
    def _row_early_check(
        i: int,
        t: datetime.date,
        mask: np.ndarray,
        mu_row: np.ndarray,
    ) -> tuple[np.ndarray, SolveYield | None]:
        """Validate the price mask and expected-return signal for a single row.

        Returns an ``(expected_mu, early_yield)`` pair.  When ``early_yield``
        is not ``None``, the caller should ``yield early_yield; continue``
        immediately — the row is either degenerate (empty mask) or has an
        all-zero signal.  When ``early_yield`` is ``None`` the row is ready
        for the mode-specific solve step.

        Args:
            i: Row index.
            t: Timestamp.
            mask: Boolean array of shape ``(n_assets,)`` indicating finite prices.
            mu_row: Expected-return row of shape ``(n_assets,)``.

        Returns:
            tuple: ``(expected_mu, early_yield)`` where ``expected_mu`` is
            ``np.nan_to_num(mu_row[mask])`` and ``early_yield`` is either a
            complete `SolveYield` tuple (when the caller should yield
            and continue) or ``None`` (when the caller should proceed to solve).
        """
        if not mask.any():
            return np.zeros(0), (i, t, mask, np.zeros(0), SolveStatus.DEGENERATE)
        expected_mu = np.nan_to_num(mu_row[mask])
        sig_status = _SolvePrimitivesMixin._check_signal(mu_row, mask)
        if sig_status is not None:
            return expected_mu, (i, t, mask, np.zeros_like(expected_mu), sig_status)
        return expected_mu, None

    @staticmethod
    def _denom_guard_yield(
        i: int,
        t: datetime.date,
        mask: np.ndarray,
        expected_mu: np.ndarray,
        pos_raw: np.ndarray,
        denom: float,
        denom_tol: float,
    ) -> SolveYield:
        """Apply the normalisation-denominator guard and return the appropriate yield tuple.

        Emits a `WARNING` and returns a
        `DEGENERATE` yield when *denom* is non-finite or at
        or below *denom_tol*; otherwise returns a `VALID`
        yield with normalised positions ``pos_raw / denom``.

        Args:
            i: Row index.
            t: Timestamp.
            mask: Boolean asset mask of shape ``(n_assets,)``.
            expected_mu: Masked expected-return vector of shape ``(n_active,)``.
            pos_raw: Raw (pre-normalisation) position vector of shape ``(n_active,)``.
            denom: Computed normalisation denominator.
            denom_tol: Tolerance threshold below which *denom* is treated as degenerate.

        Returns:
            SolveYield: Either a degenerate or valid ``(i, t, mask, pos, status)`` tuple.
        """
        n_active = len(expected_mu)
        if not np.isfinite(denom) or denom <= denom_tol:
            _logger.warning(
                "Positions zeroed at t=%s: normalisation denominator is degenerate "
                "(denom=%s, denom_tol=%s). Check signal magnitude and covariance matrix.",
                t,
                denom,
                denom_tol,
                extra={
                    "context": {
                        "t": str(t),
                        "denom": denom,
                        "denom_tol": denom_tol,
                    }
                },
            )
            return i, t, mask, np.zeros(n_active), SolveStatus.DEGENERATE
        return i, t, mask, pos_raw / denom, SolveStatus.VALID

    @staticmethod
    def _apply_turnover_constraint(
        new_cash: np.ndarray,
        prev_cash: np.ndarray,
        max_turnover: float,
    ) -> np.ndarray:
        """Cap the L1 norm of the position change to *max_turnover*.

        When ``sum(|new_cash - prev_cash|) > max_turnover``, the delta is
        scaled back proportionally toward *prev_cash* so that the constraint
        is exactly met.  When the constraint is already satisfied the input is
        returned unchanged.

        Args:
            new_cash: Proposed cash positions after the solve step, shape
                ``(n_active,)`` — ``NaN`` values treated as zero.
            prev_cash: Cash positions at the previous step, shape
                ``(n_active,)`` — ``NaN`` values treated as zero.
            max_turnover: Maximum allowed L1 norm of the position change.

        Returns:
            np.ndarray: The (possibly scaled) new cash positions.
        """
        curr = np.nan_to_num(new_cash, nan=0.0)
        prev = np.nan_to_num(prev_cash, nan=0.0)
        delta = curr - prev
        total_delta = float(np.sum(np.abs(delta)))
        if total_delta > max_turnover:
            scale = max_turnover / total_delta
            return cast("np.ndarray", prev + delta * scale)
        return new_cash

    @staticmethod
    def _sliding_warmup_or_degenerate(
        i: int,
        t: datetime.date,
        mask: np.ndarray,
        win_w: int,
    ) -> SolveYield:
        """Classify a no-matrix sliding-window row as WARMUP or DEGENERATE.

        Distinguishes an insufficient-history warm-up row (mask non-empty but
        fewer than ``win_w`` rows seen) from a genuine no-data / model-failure
        row, which is zeroed and marked degenerate.
        """
        if mask.any() and i + 1 < win_w:
            return i, t, mask, None, SolveStatus.WARMUP
        return i, t, mask, np.zeros(int(mask.sum())), SolveStatus.DEGENERATE
