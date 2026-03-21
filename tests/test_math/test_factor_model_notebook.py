"""CI execution gate for the factor_model_guide notebook.

This module mirrors the data setup and cell logic from
``book/marimo/notebooks/factor_model_guide.py`` so that any drift in the
``FactorModel`` public API is caught by ``make test`` before it can silently
corrupt the notebook.

Covered API surface:

- Direct construction — ``FactorModel(factor_loadings, factor_covariance, idiosyncratic_var)``
- Properties — ``n_assets``, ``n_factors``, ``covariance``
- Fitting — ``FactorModel.from_returns(returns, k)``
- Woodbury solve — ``FactorModel.solve(b)`` vs ``np.linalg.solve(fm.covariance, b)``
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from basanos.math import FactorModel

_NOTEBOOK = Path(__file__).parents[2] / "book/marimo/notebooks/factor_model_guide.py"

# ─── Manual-example constants (mirror cell_04) ───────────────────────────────

_LOADINGS = np.array(
    [
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
    ]
)
_FACTOR_COV = np.array([[0.04, 0.01], [0.01, 0.03]])
_IDIO_VAR = np.array([0.02, 0.015, 0.025])

# ─── Returns-example constants (mirror cell_08) ──────────────────────────────

_SEED = 42
_N_ASSETS = 8
_T_LEN = 200
_K = 3


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fm_manual() -> FactorModel:
    """FactorModel built directly from arrays (mirrors cell_04)."""
    return FactorModel(
        factor_loadings=_LOADINGS,
        factor_covariance=_FACTOR_COV,
        idiosyncratic_var=_IDIO_VAR,
    )


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    """Synthetic unit-variance return matrix (mirrors cell_08)."""
    rng = np.random.default_rng(_SEED)
    true_loadings = rng.standard_normal((_N_ASSETS, _K))
    true_cov = true_loadings @ true_loadings.T + np.diag(np.ones(_N_ASSETS) * 0.5)
    chol = np.linalg.cholesky(true_cov)
    raw = rng.standard_normal((_T_LEN, _N_ASSETS)) @ chol.T
    return raw / raw.std(axis=0, keepdims=True)


@pytest.fixture(scope="module")
def fm_fitted(returns: np.ndarray) -> FactorModel:
    """FactorModel fitted from the synthetic return matrix (mirrors cell_13)."""
    return FactorModel.from_returns(returns, k=_K)


# ─── Direct construction ──────────────────────────────────────────────────────


class TestFactorModelDirectConstruction:
    """FactorModel constructed from explicit arrays (cell_04)."""

    def test_n_assets(self, fm_manual: FactorModel) -> None:
        assert fm_manual.n_assets == 3

    def test_n_factors(self, fm_manual: FactorModel) -> None:
        assert fm_manual.n_factors == 2

    def test_factor_loadings_shape(self, fm_manual: FactorModel) -> None:
        assert fm_manual.factor_loadings.shape == (3, 2)

    def test_factor_covariance_shape(self, fm_manual: FactorModel) -> None:
        assert fm_manual.factor_covariance.shape == (2, 2)

    def test_idiosyncratic_var_shape(self, fm_manual: FactorModel) -> None:
        assert fm_manual.idiosyncratic_var.shape == (3,)

    def test_covariance_shape(self, fm_manual: FactorModel) -> None:
        assert fm_manual.covariance.shape == (3, 3)

    def test_covariance_is_symmetric(self, fm_manual: FactorModel) -> None:
        cov = fm_manual.covariance
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_covariance_is_positive_definite(self, fm_manual: FactorModel) -> None:
        eigvals = np.linalg.eigvalsh(fm_manual.covariance)
        assert float(eigvals.min()) > 0

    def test_covariance_diagonal_equals_systematic_plus_idiosyncratic(self, fm_manual: FactorModel) -> None:
        systematic_diag = np.diag(_LOADINGS @ _FACTOR_COV @ _LOADINGS.T)
        expected_diag = systematic_diag + _IDIO_VAR
        np.testing.assert_allclose(np.diag(fm_manual.covariance), expected_diag, rtol=1e-10)

    def test_frozen_dataclass_rejects_attribute_assignment(self, fm_manual: FactorModel) -> None:
        with pytest.raises((AttributeError, TypeError)):
            fm_manual.n_assets = 99  # type: ignore[misc]


# ─── from_returns ─────────────────────────────────────────────────────────────


class TestFactorModelFromReturns:
    """FactorModel.from_returns() mirrors cell_08 / cell_13 of the notebook."""

    def test_n_assets(self, fm_fitted: FactorModel) -> None:
        assert fm_fitted.n_assets == _N_ASSETS

    def test_n_factors(self, fm_fitted: FactorModel) -> None:
        assert fm_fitted.n_factors == _K

    def test_factor_loadings_shape(self, fm_fitted: FactorModel) -> None:
        assert fm_fitted.factor_loadings.shape == (_N_ASSETS, _K)

    def test_factor_covariance_shape(self, fm_fitted: FactorModel) -> None:
        assert fm_fitted.factor_covariance.shape == (_K, _K)

    def test_idiosyncratic_var_shape(self, fm_fitted: FactorModel) -> None:
        assert fm_fitted.idiosyncratic_var.shape == (_N_ASSETS,)

    def test_idiosyncratic_var_is_positive(self, fm_fitted: FactorModel) -> None:
        assert float(fm_fitted.idiosyncratic_var.min()) > 0

    def test_covariance_shape(self, fm_fitted: FactorModel) -> None:
        assert fm_fitted.covariance.shape == (_N_ASSETS, _N_ASSETS)

    def test_covariance_is_symmetric(self, fm_fitted: FactorModel) -> None:
        cov = fm_fitted.covariance
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_covariance_is_positive_definite(self, fm_fitted: FactorModel) -> None:
        eigvals = np.linalg.eigvalsh(fm_fitted.covariance)
        assert float(eigvals.min()) > 0

    def test_explained_variance_increases_with_k(self, returns: np.ndarray) -> None:
        _, sv, _ = np.linalg.svd(returns, full_matrices=False)
        total = float((sv**2).sum())
        prev = 0.0
        for k in range(1, _N_ASSETS + 1):
            explained = float((sv[:k] ** 2).sum()) / total
            assert explained > prev
            prev = explained


# ─── Woodbury solve ───────────────────────────────────────────────────────────


class TestFactorModelWoodburySolve:
    """FactorModel.solve(b) matches np.linalg.solve(fm.covariance, b) (cell_22)."""

    def test_solve_matches_direct_within_tolerance(self, fm_fitted: FactorModel) -> None:
        rng = np.random.default_rng(99)
        b = rng.standard_normal(fm_fitted.n_assets)
        x_woodbury = fm_fitted.solve(b)
        x_direct = np.linalg.solve(fm_fitted.covariance, b)
        np.testing.assert_allclose(x_woodbury, x_direct, rtol=1e-8, atol=1e-10)

    def test_solve_satisfies_linear_system(self, fm_fitted: FactorModel) -> None:
        rng = np.random.default_rng(99)
        b = rng.standard_normal(fm_fitted.n_assets)
        x = fm_fitted.solve(b)
        residual = float(np.abs(fm_fitted.covariance @ x - b).max())
        assert residual < 1e-10

    def test_solve_output_shape(self, fm_fitted: FactorModel) -> None:
        b = np.ones(fm_fitted.n_assets)
        x = fm_fitted.solve(b)
        assert x.shape == (fm_fitted.n_assets,)

    def test_manual_model_woodbury_matches_direct(self, fm_manual: FactorModel) -> None:
        rng = np.random.default_rng(7)
        b = rng.standard_normal(fm_manual.n_assets)
        x_woodbury = fm_manual.solve(b)
        x_direct = np.linalg.solve(fm_manual.covariance, b)
        np.testing.assert_allclose(x_woodbury, x_direct, rtol=1e-8, atol=1e-10)


# ─── Direct notebook execution ───────────────────────────────────────────────


def test_notebook_executes() -> None:
    """Execute factor_model_guide.py directly via marimo export html (no sandbox).

    This catches regressions in notebook cell code itself, not just the API
    that the mirror tests validate.
    """
    result = subprocess.run(  # nosec
        [sys.executable, "-m", "marimo", "export", "html", str(_NOTEBOOK), "-o", "/dev/null"],
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
