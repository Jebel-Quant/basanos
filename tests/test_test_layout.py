"""Run ``scripts/check_test_layout.py`` as part of the test gate.

``[tool.check_test_layout]`` in ``pyproject.toml`` opts this repo out of the
generic ``src``/``tests`` mirror check that tooling (``/rhiza:quality``) applies,
on the recorded grounds that parity is enforced by ``scripts/check_test_layout.py``
instead. That opt-out is only honest while something actually runs the script.

It used to be ``make test-layout``, a prerequisite of ``make test``. rhiza v1.4
retired the synced make layer, so that target is gone and CI invokes the gates
through ``rhiza-task`` rather than through make — a repo-local make target would
no longer run in CI. This test is the replacement: the ``test`` gate is one of the
four jobs the required CI gate aggregates, so a layout drift fails the build again.
"""

from __future__ import annotations

import subprocess  # nosec B404 - fixed argv, no shell, repo-local script
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHECKER = ROOT / "scripts" / "check_test_layout.py"


def test_test_layout_is_clean() -> None:
    """Every public module is tested, and every test traces back to code."""
    result = subprocess.run(  # nosec B603 - fixed argv, shell=False
        [sys.executable, str(CHECKER)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
