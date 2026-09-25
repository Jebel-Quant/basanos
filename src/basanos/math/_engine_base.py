"""Validated batch core shared by `BasanosEngine` and `BasanosStream`.

`_BatchCore` holds the three engine inputs (``prices``, ``mu``, ``cfg``),
validates them on construction, and composes `_CoreDataMixin` (``assets``,
``ret_adj``, ``vola``, ``cor``) and `_SolveMixin` (the per-timestamp solve and
``warmup_state``).

It also composes `_SignalEvaluatorMixin`. The stream never calls the IC
methods, but the solve mixin's methods are typed ``self: _EngineProtocol``
(enforced by ``tests/test_math/test_engine_protocol.py``), and that protocol
includes ``_ic_series``. The IC mixin is stateless, so carrying it costs nothing
and keeps the single-protocol convention intact.

`BasanosEngine` subclasses it and adds the diagnostics and performance
mixins. `BasanosStream.from_warmup` builds a `_BatchCore`
directly, so the streaming path runs the same validation and solve code as the
batch engine without depending on the ``optimizer`` facade.
"""

import dataclasses

import polars as pl

from ._config import BasanosConfig
from ._engine_core import _CoreDataMixin
from ._engine_ic import _SignalEvaluatorMixin
from ._engine_solve import _SolveMixin
from ._engine_validation import _validate_inputs


@dataclasses.dataclass(frozen=True)
class _BatchCore(_CoreDataMixin, _SolveMixin, _SignalEvaluatorMixin):
    """Validated ``prices`` / ``mu`` / ``cfg`` with core data access and solve logic.

    Attributes:
        prices: Polars DataFrame of price levels per asset over time.
        mu: Polars DataFrame of expected-return signals aligned with *prices*.
        cfg: Immutable `BasanosConfig`.
    """

    prices: pl.DataFrame
    mu: pl.DataFrame
    cfg: BasanosConfig

    def __post_init__(self) -> None:
        """Validate inputs by delegating to `_validate_inputs`."""
        _validate_inputs(self.prices, self.mu, self.cfg)
