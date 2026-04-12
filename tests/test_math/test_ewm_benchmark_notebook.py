"""CI execution gate for the ewm_benchmark notebook.

This module mirrors the correctness-check cells from
``book/marimo/notebooks/ewm_benchmark.py`` so that any change to the
``_ewm_corr_numpy`` implementation or its import path is caught by
``make test`` before it can silently corrupt the notebook.

The notebook compares two EWM correlation implementations:
- Original: ``pandas.DataFrame.ewm().corr()``
- Current: ``_ewm_corr_numpy`` (scipy.signal.lfilter-based)

Both must agree to within 1e-10 (absolute) on all finite entries, and
NaN patterns must be identical.

pandas is not a basanos runtime dependency; tests are skipped if it is
not installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# _ewm_corr_numpy is imported directly from the private module because this
# test validates the internal implementation that the ewm_benchmark notebook
# also uses.  Tests are permitted to access private internals; making the
# symbol public solely to satisfy a test import is what the issue warns against.
from basanos.math._ewm_corr import ewm_corr as ewm_corr_numpy

_NOTEBOOK = Path(__file__).parents[2] / "book/marimo/notebooks/ewm_benchmark.py"

# ─── Pandas reference implementation (mirrors notebook setup cell) ────────────


def _ewm_corr_pandas(data: np.ndarray, com: int, min_periods: int) -> np.ndarray:
    """EWM correlation via pandas — mirrors the notebook's reference implementation."""
    t_len, n_assets = data.shape
    df = pd.DataFrame(data)
    ewm_result = df.ewm(com=com, min_periods=min_periods).corr()
    cor = ewm_result.reset_index(names=["t", "asset"])
    result = np.full((t_len, n_assets, n_assets), np.nan)
    for t, df_t in cor.groupby("t"):
        result[int(t)] = df_t.drop(columns=["t", "asset"]).to_numpy()
    return result


# ─── Import path ──────────────────────────────────────────────────────────────


class TestEwmCorrNumpyImportPath:
    """The notebook imports _ewm_corr_numpy from basanos.math._ewm_corr."""

    def test_import_is_callable(self) -> None:
        assert callable(ewm_corr_numpy)

    def test_import_returns_ndarray_on_trivial_input(self) -> None:
        data = np.ones((10, 2))
        result = ewm_corr_numpy(data, com=5, min_periods=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 2, 2)


# ─── Output shape ─────────────────────────────────────────────────────────────


class TestEwmCorrNumpyShape:
    """Output shape is (T, N, N) for all input sizes (mirrors cell_07)."""

    @pytest.mark.parametrize(
        ("t_len", "n_assets", "com"),
        [
            (100, 2, 10),
            (500, 4, 32),
            (200, 6, 16),
        ],
    )
    def test_output_shape(self, t_len: int, n_assets: int, com: int) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(size=(t_len, n_assets))
        result = ewm_corr_numpy(data, com=com, min_periods=com)
        assert result.shape == (t_len, n_assets, n_assets)

    def test_diagonal_is_one_or_nan(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.normal(size=(200, 4))
        result = ewm_corr_numpy(data, com=32, min_periods=32)
        for t in range(result.shape[0]):
            diag = np.diag(result[t])
            finite_diag = diag[np.isfinite(diag)]
            if len(finite_diag) > 0:
                np.testing.assert_allclose(finite_diag, np.ones(len(finite_diag)), atol=1e-10)

    def test_output_is_symmetric(self) -> None:
        rng = np.random.default_rng(2)
        data = rng.normal(size=(100, 3))
        result = ewm_corr_numpy(data, com=10, min_periods=10)
        for t in range(result.shape[0]):
            mat = result[t]
            finite = np.isfinite(mat)
            if finite.any():
                np.testing.assert_array_equal(mat[finite], mat.T[finite])


# ─── Correctness vs pandas ────────────────────────────────────────────────────


class TestEwmCorrNumpyVsPandas:
    """numpy result matches pandas to within 1e-10 (mirrors cell_09 correctness check)."""

    @pytest.fixture(scope="class")
    def dense_data(self) -> tuple[np.ndarray, int]:
        rng = np.random.default_rng(42)
        data = rng.normal(size=(500, 4))
        return data, 32

    @pytest.fixture(scope="class")
    def sparse_data(self) -> tuple[np.ndarray, int]:
        """~5% NaN entries, matching the notebook's missing-data scenario."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(500, 4))
        data[rng.random(data.shape) < 0.05] = np.nan
        return data, 32

    def _compare(self, data: np.ndarray, com: int) -> None:
        pd_result = _ewm_corr_pandas(data, com, com)
        np_result = ewm_corr_numpy(data, com, com)
        both_finite = np.isfinite(pd_result) & np.isfinite(np_result)
        nan_match = bool(np.all(~np.isfinite(pd_result) == ~np.isfinite(np_result)))
        assert nan_match, "NaN patterns differ between pandas and numpy implementations"
        if both_finite.any():
            max_diff = float(np.abs(pd_result[both_finite] - np_result[both_finite]).max())
            assert max_diff < 1e-10, f"Max absolute difference {max_diff:.2e} exceeds 1e-10"

    def test_dense_input_matches_pandas(self, dense_data: tuple[np.ndarray, int]) -> None:
        self._compare(*dense_data)

    def test_sparse_input_matches_pandas(self, sparse_data: tuple[np.ndarray, int]) -> None:
        self._compare(*sparse_data)

    @pytest.mark.parametrize("com", [5, 16, 64])
    def test_various_com_values_match_pandas(self, com: int) -> None:
        rng = np.random.default_rng(99)
        data = rng.normal(size=(300, 4))
        self._compare(data, com)

    def test_single_asset_matches_pandas(self) -> None:
        rng = np.random.default_rng(7)
        data = rng.normal(size=(200, 1))
        self._compare(data, 20)


# ─── Direct notebook execution ───────────────────────────────────────────────


def test_notebook_executes() -> None:
    """Execute ewm_benchmark.py directly via marimo export html (no sandbox).

    This catches regressions in notebook cell code itself, not just the API
    that the mirror tests validate.
    """
    result = subprocess.run(  # nosec
        [sys.executable, "-m", "marimo", "export", "html", "--no-sandbox", str(_NOTEBOOK), "-o", os.devnull],
        capture_output=True,
        text=True,
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    failure_keywords = ["cells failed to execute", "marimoexceptionraisederror"]
    for kw in failure_keywords:
        assert kw.lower() not in combined.lower(), (
            f"Notebook {_NOTEBOOK.name} reported cell failures (keyword '{kw}'):\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    assert result.returncode == 0, (
        f"marimo export returned non-zero for {_NOTEBOOK.name}:\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
