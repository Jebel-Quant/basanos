"""Fuzz the basanos matrix-shrinkage helper against arbitrary matrices.

``shrink2id`` shrinks a square numpy matrix toward the identity by a weight
factor and must never crash with an unexpected exception on adversarial input
(degenerate shapes, non-finite values). This harness exercises that contract
with coverage-guided input.

Run locally:
    pip install atheris numpy scipy polars pydantic
    python tests/fuzz/fuzz_signal.py -atheris_runs=20000

Run in ClusterFuzzLite: this file is built by .clusterfuzzlite/build.sh.
"""

from __future__ import annotations

import contextlib
import sys

import atheris

# Pre-import the heavy native dependencies OUTSIDE the instrumentation block.
# Atheris's import hook miscompiles parts of polars' (and other Rust/C
# extensions') Python machinery, so we let them load uninstrumented and
# instrument only the first-party package under test. Importing
# basanos.math._signal pulls in the whole basanos.math package, whose __init__
# imports these libraries, so they must be cached uninstrumented beforehand.
import numpy as np
import plotly.graph_objects  # pre-imported uninstrumented
import plotly.io  # noqa: F401  # pre-imported uninstrumented
import polars as pl  # noqa: F401  # pre-imported uninstrumented
import pydantic  # noqa: F401  # pre-imported uninstrumented
import scipy.signal  # pre-imported uninstrumented
import scipy.stats  # noqa: F401  # pre-imported uninstrumented

with atheris.instrument_imports():
    from basanos.math._signal import shrink2id

_ALLOWED = (ValueError, TypeError, ZeroDivisionError, np.linalg.LinAlgError)


def test_one_input(data: bytes) -> None:
    """Shrink a fuzzed square matrix toward the identity."""
    fdp = atheris.FuzzedDataProvider(data)
    n = fdp.ConsumeIntInRange(0, 8)
    matrix = np.array([fdp.ConsumeFloat() for _ in range(n * n)], dtype=np.float64).reshape(n, n)

    with contextlib.suppress(_ALLOWED):
        shrink2id(matrix, fdp.ConsumeProbability())


def main() -> None:
    """Run the Atheris fuzz loop."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
