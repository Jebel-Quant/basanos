#!/usr/bin/env python3
"""Check that public source modules are tested and tests trace back to code.

basanos organises its tests by *behaviour* rather than mirroring the source
tree one-to-one: a single source module is often exercised by several suites
(e.g. ``optimizer.py`` is covered by ``test_optimizer.py``,
``test_optimizer_property.py`` and ``test_optimizer_edge_cases.py``), and
whole-pipeline suites (the ``*_notebook`` end-to-end tests, numerical
regression/stability suites, the paper-example walkthrough) span many modules
and map to no single source file. A strict ``tests/<pkg>/test_<name>.py`` ↔
``src/<pkg>/<name>.py`` mirror is therefore the wrong shape for this repo.

This checker enforces the two properties that actually matter here, while
recording the intentional behaviour-grouping as configuration:

  * **Every public source module is tested.** For each non-underscore module
    under ``src/`` (private ``_*`` modules and ``__init__`` are skipped), a
    ``test_<name>.py`` file must exist *somewhere* under ``tests/`` — the
    behaviour grouping decides where.
  * **Every test traces back to real code.** Each ``tests/**/test_*.py`` file
    must either name a real source module (``<name>.py`` or ``_<name>.py``
    anywhere under ``src/``) or be a declared cross-cutting suite (see
    :data:`ALLOWED_ORPHANS` / :data:`ORPHAN_SUFFIXES`).

``__init__.py``/``conftest.py`` and the ``tests/benchmarks/`` and
``tests/stress/`` trees are exempt. Test *functions* and *classes* are
unconstrained.

Usage:
  python3 scripts/check_test_layout.py

Exits 0 when the layout is clean, 1 (listing every violation) otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC_ROOT = ROOT / "src"
TEST_ROOT = ROOT / "tests"

_IGNORED = {"__init__.py", "conftest.py"}

# Top-level directories under tests/ that need not trace back to a source
# module (benchmarks and load/stress suites).
_EXEMPT_DIRS = {"benchmarks", "stress"}

# Cross-cutting test suites that intentionally map to no single source module.
# ``test_<stem>.py`` files whose stem is listed here (or matches one of
# ``ORPHAN_SUFFIXES``) are recognised as behaviour suites, not orphans.
ALLOWED_ORPHANS: frozenset[str] = frozenset(
    {
        "cross_mode_consistency",  # EWMA vs sliding-window path agreement
        "engine_integration",  # end-to-end engine wiring
        "ewm_covariance",  # EWMA covariance behaviour
        "numerical_regression",  # pinned numerical outputs
        "numerical_stability",  # degenerate-input robustness
        "paper_example",  # reproduces the reference paper example
        "shim",  # analytics compatibility shim
    }
)

# Suffixes marking a whole family of behaviour suites (kept as patterns so new
# suites in these families need no extra configuration).
ORPHAN_SUFFIXES: tuple[str, ...] = ("_notebook", "_property", "_edge_cases")


def _public_source_modules() -> list[Path]:
    """Return public (non-underscore, non-dunder) ``.py`` modules under src/."""
    return sorted(p for p in SRC_ROOT.rglob("*.py") if p.name not in _IGNORED and not p.name.startswith("_"))


def _source_basenames() -> set[str]:
    """Return every source module basename, with any leading underscore stripped.

    ``src/basanos/math/_signal.py`` and ``src/basanos/math/optimizer.py`` both
    contribute (``signal``, ``optimizer``), so a ``test_signal.py`` covering a
    private module is not reported as an orphan.
    """
    names: set[str] = set()
    for p in SRC_ROOT.rglob("*.py"):
        if p.name in _IGNORED:
            continue
        names.add(p.stem.lstrip("_"))
    return names


def _test_files() -> list[Path]:
    """Return ``test_*.py`` files under tests/ (ignoring conftest/exempt dirs)."""
    return sorted(
        p
        for p in TEST_ROOT.rglob("test_*.py")
        if p.name not in _IGNORED and p.relative_to(TEST_ROOT).parts[0] not in _EXEMPT_DIRS
    )


def _test_stems() -> set[str]:
    """Return the set of ``<stem>`` for every ``test_<stem>.py`` under tests/."""
    return {p.stem[len("test_") :] for p in _test_files()}


def _is_cross_cutting(stem: str) -> bool:
    """Return whether *stem* names a declared cross-cutting behaviour suite."""
    return stem in ALLOWED_ORPHANS or stem.endswith(ORPHAN_SUFFIXES)


def check() -> list[str]:
    """Return a list of layout violations (empty when the layout is clean)."""
    errors: list[str] = []

    # Forward: every public source module must be tested somewhere.
    test_stems = _test_stems()
    for module in _public_source_modules():
        if module.stem not in test_stems:
            errors.append(
                f"missing test file test_{module.stem}.py (anywhere under tests/) "
                f"for public source module {module.relative_to(ROOT)}"
            )

    # Reverse: every test file must name real code or a declared behaviour suite.
    source_basenames = _source_basenames()
    for test_file in _test_files():
        stem = test_file.stem[len("test_") :]
        if _is_cross_cutting(stem) or stem in source_basenames:
            continue
        errors.append(
            f"orphan test file {test_file.relative_to(ROOT)} "
            f"(no source module {stem}.py/_{stem}.py, and not a declared cross-cutting suite)"
        )

    return errors


def main() -> int:
    """Run the checks and return an exit code."""
    errors = check()
    if errors:
        print("Test-layout check failed:", file=sys.stderr)
        for err in errors:
            print(f"  ✗ {err}", file=sys.stderr)
        return 1
    print("Test-layout OK: every public module is tested; every test traces back to code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
