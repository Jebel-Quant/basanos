"""Direct-execution CI gate for all Marimo notebooks.

Each test runs ``python -m marimo export html --no-sandbox <notebook>`` and
verifies that:

1. The process exits with code 0.
2. No cell-failure keywords appear in the combined stdout/stderr.

This complements the API-mirror tests in ``tests/test_math/test_*_notebook.py``
by executing the actual notebook cells so that changes to notebook code are
caught — not only changes to the underlying API.
"""

from __future__ import annotations

import subprocess  # nosec
import sys
import tempfile
from pathlib import Path

import pytest

# Resolve the notebooks directory relative to this file's repository root.
_REPO_ROOT = Path(__file__).parent.parent.parent
_NOTEBOOKS_DIR = _REPO_ROOT / "book" / "marimo" / "notebooks"

# Collect notebook paths at import/collection time so pytest.parametrize can
# use them.  Returns an empty list when the directory does not exist so
# parametrize degrades gracefully (the discovery test below will skip).
_NOTEBOOK_PATHS: list[Path] = sorted(_NOTEBOOKS_DIR.glob("*.py")) if _NOTEBOOKS_DIR.exists() else []

# Keywords that indicate a cell failed to execute even when the process exits 0.
_FAILURE_KEYWORDS: list[str] = [
    "cells failed to execute",
    "marimoexceptionraisederror",
]


def test_notebooks_discovered() -> None:
    """At least one notebook must be present for the parametrized tests to run."""
    if not _NOTEBOOK_PATHS:
        pytest.skip(f"No Marimo notebooks found in {_NOTEBOOKS_DIR}")


@pytest.mark.slow
@pytest.mark.parametrize("notebook_path", _NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_executes_without_errors(notebook_path: Path) -> None:
    """Execute a Marimo notebook and assert no cells fail.

    Uses ``python -m marimo export html --no-sandbox`` so that the notebook
    runs inside the project's existing virtual environment (basanos already
    installed) without spawning an isolated sandbox.

    The exported HTML is written to a temporary file and discarded; only the
    process exit-code and failure keywords in the output are checked.
    """
    with tempfile.NamedTemporaryFile(suffix=".html", delete=True) as tmp:
        cmd = [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            "--no-sandbox",
            str(notebook_path),
            "-o",
            tmp.name,
            "-f",  # overwrite the temp file without prompting
        ]

        result = subprocess.run(  # nosec
            cmd,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=300,
        )

    assert result.returncode == 0, (
        f"marimo export returned non-zero for {notebook_path.name}:\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    lower = combined.lower()
    for kw in _FAILURE_KEYWORDS:
        assert kw.lower() not in lower, (
            f"Notebook {notebook_path.name} reported cell failures "
            f"(found keyword '{kw}'):\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
