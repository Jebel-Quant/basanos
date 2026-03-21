"""Structural protocol defining the attribute contract for BasanosEngine helpers.

:class:`_EngineProtocol` is the single source of truth for the attributes and
methods that the private helper modules (:mod:`basanos.math._engine_diagnostics`,
:mod:`basanos.math._engine_ic`, and :mod:`basanos.math._engine_solve`) expect to
be provided by the concrete consuming class
(:class:`~basanos.math.optimizer.BasanosEngine`).

Using a :class:`~typing.Protocol` instead of annotation-only class variables
makes the contract formally verifiable by type checkers (``ty``, ``mypy``,
``pyright``) and removes the need for runtime stubs that would only be
discovered when an attribute is first accessed.

Classes in this module are **private implementation details**.  The public API
is :class:`~basanos.math.optimizer.BasanosEngine`.

----

**Contributor guide: the ``self: _EngineProtocol`` helper pattern**

Methods in the private helper modules use an explicit ``self`` type annotation
instead of the plain ``self`` convention:

.. code-block:: python

    class _MyHelpers:
        def my_method(self: _EngineProtocol) -> ...:
            # self.assets, self.cfg, etc. are now fully typed
            ...

Why this pattern?

1. **Type-checker verification.**  Annotating ``self`` with
   :class:`_EngineProtocol` tells ``ty``/``mypy``/``pyright`` that the
   method may only be called on objects that satisfy the protocol.  Any
   attribute access on ``self`` inside the method is validated against the
   protocol definition — missing attributes become type errors rather than
   silent runtime failures.

2. **Why annotation-only class variables are not enough.**
   A pattern like ``class _MyHelper: assets: list[str]`` creates a *class-level
   annotation* but no actual attribute.  The annotation is invisible to the
   type checker on ``self`` inside methods, and the attribute does not exist
   at runtime unless the concrete subclass initialises it.  Using
   :class:`_EngineProtocol` as the ``self`` type avoids both problems:
   the protocol is the single, authoritative declaration of what the host
   class must provide.

3. **How to add new engine methods.**
   a. Check whether all attributes your method needs are already declared on
      :class:`_EngineProtocol`.  If not, add them there.
   b. Write the method in the appropriate helper module with
      ``self: _EngineProtocol`` (imported under ``TYPE_CHECKING`` to avoid a
      circular import at runtime).
   c. Assign the descriptor directly in :class:`~basanos.math.optimizer.BasanosEngine`
      under the matching section — see ``CONTRIBUTING.md`` for details.

   .. code-block:: python

       # your_helper_module.py
       from __future__ import annotations
       from typing import TYPE_CHECKING

       if TYPE_CHECKING:
           from ._engine_protocol import _EngineProtocol


       class _MyHelpers:
           def compute_something(self: _EngineProtocol) -> float:
               return float(self.cfg.shrink) * len(self.assets)

   The ``TYPE_CHECKING`` guard keeps the import free of circular-import issues
   at runtime while still giving the type checker full visibility.
"""

from __future__ import annotations

import datetime
from collections.abc import Generator
from typing import TYPE_CHECKING, Protocol

import numpy as np
import polars as pl

from ._config import BasanosConfig

if TYPE_CHECKING:
    from ._engine_solve import MatrixYield, SolveYield


class _EngineProtocol(Protocol):
    """Structural contract for classes that consume the engine helper modules.

    Any class that satisfies this protocol by providing the listed attributes
    and methods can safely use methods from :class:`_DiagnosticsMixin`,
    :class:`_SignalEvaluatorMixin`, and :class:`_SolveMixin`.
    """

    assets: list[str]
    prices: pl.DataFrame
    mu: pl.DataFrame
    cfg: BasanosConfig
    cor: dict[datetime.date, np.ndarray]
    ret_adj: pl.DataFrame
    vola: pl.DataFrame

    def _iter_matrices(self) -> Generator[MatrixYield, None, None]:
        """Yield ``(i, t, mask, matrix)`` tuples over all timestamps."""
        ...

    def _iter_solve(self) -> Generator[SolveYield, None, None]:
        """Yield ``(i, t, mask, pos_or_none, status)`` tuples over all timestamps."""
        ...

    def _ic_series(self, use_rank: bool) -> pl.DataFrame:
        """Compute the cross-sectional IC time series."""
        ...
