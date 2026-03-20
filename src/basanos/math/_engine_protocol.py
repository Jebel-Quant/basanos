"""Structural protocol defining the attribute contract for BasanosEngine mixins.

:class:`_EngineProtocol` is the single source of truth for the attributes and
methods that :class:`_DiagnosticsMixin` and :class:`_SignalEvaluatorMixin`
expect to be provided by the concrete consuming class
(:class:`~basanos.math.optimizer.BasanosEngine`).

Using a :class:`~typing.Protocol` instead of annotation-only class variables
makes the contract formally verifiable by type checkers (``ty``, ``mypy``,
``pyright``) and removes the need for runtime stubs that would only be
discovered when an attribute is first accessed.

Classes in this module are **private implementation details**.  The public API
is :class:`~basanos.math.optimizer.BasanosEngine`.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, Protocol

import numpy as np
import polars as pl

# (i, t, mask, matrix) — see BasanosEngine._iter_matrices for details.
_MatrixRow = tuple[int, Any, np.ndarray, np.ndarray | None]


class _EngineProtocol(Protocol):
    """Structural contract for classes that consume the engine mixins.

    Any class that satisfies this protocol by providing the listed attributes
    and methods can safely inherit from :class:`_DiagnosticsMixin` and
    :class:`_SignalEvaluatorMixin`.
    """

    assets: list[str]
    prices: pl.DataFrame
    mu: pl.DataFrame

    def _iter_matrices(self) -> Generator[_MatrixRow, None, None]:
        """Yield ``(i, t, mask, matrix)`` tuples over all timestamps."""
        ...

    def _ic_series(self, use_rank: bool) -> pl.DataFrame:
        """Compute the cross-sectional IC time series."""
        ...
