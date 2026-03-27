"""Tests for the _EngineProtocol mixin-contract pattern.

This module verifies that:

1. Every mixin method that accesses engine attributes annotates its ``self``
   parameter as ``_EngineProtocol``.  This is the convention described in
   ``_engine_protocol.py`` and ``docs/ARCHITECTURE.md`` that prevents
   spurious ``attribute-not-found`` errors from static type checkers and IDEs.

2. ``_EngineProtocol`` declares all attributes accessed by the mixin methods
   (attribute completeness).

3. ``BasanosEngine`` exposes every attribute that ``_EngineProtocol`` requires
   (runtime conformance).

The goal is to give contributors a failing test — rather than a silent
runtime error or spurious type-checker warning — whenever the mixin contract
is broken.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import polars as pl
import pytest

from basanos.math import BasanosConfig, BasanosEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_self_annotation(cls: type, name: str) -> str | None:
    """Return the raw string annotation for ``self`` on *cls.name*.

    Works for plain methods and properties (via ``fget``).  Returns ``None``
    when no annotation is present on the first parameter.  With
    ``from __future__ import annotations`` all annotations are stored as
    string literals; this function returns that raw string so tests do not
    need to import ``_EngineProtocol`` at runtime (it lives inside an
    ``if TYPE_CHECKING:`` guard).
    """
    attr = getattr(cls, name)
    func = attr.fget if isinstance(attr, property) else attr
    raw = inspect.get_annotations(func)
    return raw.get("self")


# ---------------------------------------------------------------------------
# Table of methods that MUST carry ``self: _EngineProtocol``
# ---------------------------------------------------------------------------

_MUST_USE_PROTOCOL: list[tuple[str, str]] = [
    # _DiagnosticsMixin
    ("_DiagnosticsMixin", "condition_number"),
    ("_DiagnosticsMixin", "effective_rank"),
    ("_DiagnosticsMixin", "solver_residual"),
    ("_DiagnosticsMixin", "signal_utilisation"),
    # _SignalEvaluatorMixin
    ("_SignalEvaluatorMixin", "_ic_series"),
    ("_SignalEvaluatorMixin", "ic"),
    ("_SignalEvaluatorMixin", "rank_ic"),
    # _SolveMixin
    ("_SolveMixin", "_replay_positions"),
    ("_SolveMixin", "_iter_matrices"),
    ("_SolveMixin", "_iter_solve"),
    ("_SolveMixin", "warmup_state"),
]


def _resolve_class(name: str) -> type:
    if name == "_DiagnosticsMixin":
        from basanos.math._engine_diagnostics import _DiagnosticsMixin

        return _DiagnosticsMixin
    if name == "_SignalEvaluatorMixin":
        from basanos.math._engine_ic import _SignalEvaluatorMixin

        return _SignalEvaluatorMixin
    if name == "_SolveMixin":
        from basanos.math._engine_solve import _SolveMixin

        return _SolveMixin
    raise ValueError(name)  # only called with known class names from _MUST_USE_PROTOCOL


@pytest.mark.parametrize(("cls_name", "method_name"), _MUST_USE_PROTOCOL)
def test_mixin_self_annotation_is_engine_protocol(cls_name: str, method_name: str) -> None:
    """Each listed mixin method must annotate ``self`` as ``_EngineProtocol``.

    This test guards against contributors accidentally omitting the annotation
    when adding new methods (which would silently allow type checkers to miss
    attribute-not-found errors on ``self``).
    """
    cls = _resolve_class(cls_name)
    ann = _get_self_annotation(cls, method_name)
    assert ann == "_EngineProtocol", (
        f"{cls_name}.{method_name}: expected ``self: _EngineProtocol`` annotation, "
        f"got {ann!r}.  "
        f"See docs/ARCHITECTURE.md §'Engine mixin architecture' for the convention."
    )


# ---------------------------------------------------------------------------
# Protocol attribute completeness
# ---------------------------------------------------------------------------


def test_engine_protocol_declares_all_required_attributes() -> None:
    """_EngineProtocol must declare every attribute used by the mixin methods.

    This test reads the ``__annotations__`` of ``_EngineProtocol`` at import
    time (safe because pyproject.toml excludes the file from coverage, but the
    module itself is importable) and checks that the documented engine
    attributes are all present.
    """
    from basanos.math._engine_protocol import _EngineProtocol

    protocol_annotations = _EngineProtocol.__annotations__
    required_attrs = {"assets", "prices", "mu", "cfg", "cor", "ret_adj", "vola"}
    missing = required_attrs - set(protocol_annotations)
    assert not missing, (
        f"_EngineProtocol is missing annotations for: {missing}.  "
        f"Add them to _engine_protocol.py so that mixin self-typing is complete."
    )


# ---------------------------------------------------------------------------
# Runtime conformance: BasanosEngine satisfies _EngineProtocol
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_engine() -> BasanosEngine:
    """Minimal BasanosEngine used for structural conformance checks."""
    rng = np.random.default_rng(0)
    n = 40
    prices = pl.DataFrame(
        {
            "date": list(range(n)),
            "A": np.cumprod(1 + rng.normal(0.001, 0.02, n)) * 100.0,
            "B": np.cumprod(1 + rng.normal(0.001, 0.02, n)) * 120.0,
        }
    )
    mu = pl.DataFrame(
        {
            "date": list(range(n)),
            "A": rng.normal(0.0, 0.3, n),
            "B": rng.normal(0.0, 0.3, n),
        }
    )
    cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1_000_000)
    return BasanosEngine(prices=prices, mu=mu, cfg=cfg)


@pytest.mark.parametrize(
    "attr_name",
    ["assets", "prices", "mu", "cfg", "cor", "ret_adj", "vola"],
)
def test_basanos_engine_provides_protocol_attributes(small_engine: BasanosEngine, attr_name: str) -> None:
    """BasanosEngine must provide every attribute declared in _EngineProtocol.

    A failing test here means ``_EngineProtocol`` has drifted from
    ``BasanosEngine`` — either an attribute was renamed in the engine without
    updating the protocol, or a new protocol attribute was added without a
    corresponding engine implementation.
    """
    assert hasattr(small_engine, attr_name), (
        f"BasanosEngine is missing attribute {attr_name!r} declared in _EngineProtocol."
    )


def test_basanos_engine_is_frozen_dataclass() -> None:
    """BasanosEngine must remain a frozen dataclass.

    Mixin inheritance relies on ``BasanosEngine`` being a frozen dataclass so
    that the shared ``__post_init__`` validation hook runs, and so that
    instances remain hashable and immutable after construction.
    """
    assert dataclasses.is_dataclass(BasanosEngine)
    assert BasanosEngine.__dataclass_params__.frozen  # type: ignore[attr-defined]
