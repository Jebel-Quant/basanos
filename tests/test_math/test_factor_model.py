"""Tests for basanos.math._factor_model.FactorModel (Section 4.1).

Tests cover:
- Direct construction and the frozen (immutable) contract
- __post_init__ shape and positivity validation
- Properties: n_assets, n_factors, covariance
- from_returns classmethod: shapes, unit diagonal, SVD identity, edge cases
- solve method: Woodbury identity correctness, edge cases, error handling,
  and IllConditionedMatrixWarning for ill-conditioned factor_covariance
"""

import dataclasses
import warnings
from unittest.mock import patch

import numpy as np
import pytest

from basanos.exceptions import (
    DimensionMismatchError,
    FactorModelError,
    IllConditionedMatrixWarning,
    SingularMatrixError,
)
from basanos.math._factor_model import FactorModel
from basanos.math._linalg import _DEFAULT_COND_THRESHOLD

# ─── helpers ─────────────────────────────────────────────────────────────────


def _make_fm(n: int = 4, k: int = 2) -> FactorModel:
    """Construct a minimal valid FactorModel with *n* assets and *k* factors."""
    rng = np.random.default_rng(42)
    b_mat = rng.standard_normal((n, k))
    f_mat = np.eye(k)
    d = np.ones(n)
    return FactorModel(factor_loadings=b_mat, factor_covariance=f_mat, idiosyncratic_var=d)


# ─── construction & frozen semantics ─────────────────────────────────────────


def test_construction_stores_fields():
    """Stored fields must equal the arrays passed at construction."""
    b_mat = np.eye(3, 2)
    f_mat = np.eye(2)
    d = np.ones(3)
    fm = FactorModel(factor_loadings=b_mat, factor_covariance=f_mat, idiosyncratic_var=d)
    np.testing.assert_array_equal(fm.factor_loadings, b_mat)
    np.testing.assert_array_equal(fm.factor_covariance, f_mat)
    np.testing.assert_array_equal(fm.idiosyncratic_var, d)


def test_frozen_raises_on_attribute_assignment():
    """Assigning to any field of a frozen dataclass must raise FrozenInstanceError."""
    fm = _make_fm()
    with pytest.raises(dataclasses.FrozenInstanceError):
        fm.factor_loadings = np.eye(4, 2)  # type: ignore[misc]


# ─── __post_init__ validation ─────────────────────────────────────────────────


def test_post_init_rejects_1d_factor_loadings():
    """factor_loadings must be 2-D; a 1-D array must raise FactorModelError."""
    with pytest.raises(FactorModelError, match="factor_loadings must be 2-D"):
        FactorModel(
            factor_loadings=np.ones(4),
            factor_covariance=np.eye(1),
            idiosyncratic_var=np.ones(4),
        )


def test_post_init_rejects_wrong_factor_covariance_shape():
    """factor_covariance shape must match k from factor_loadings."""
    with pytest.raises(FactorModelError, match="factor_covariance must have shape"):
        FactorModel(
            factor_loadings=np.eye(4, 2),
            factor_covariance=np.eye(3),  # wrong: should be (2, 2)
            idiosyncratic_var=np.ones(4),
        )


def test_post_init_rejects_wrong_idiosyncratic_var_shape():
    """idiosyncratic_var length must match n from factor_loadings."""
    with pytest.raises(FactorModelError, match="idiosyncratic_var must have shape"):
        FactorModel(
            factor_loadings=np.eye(4, 2),
            factor_covariance=np.eye(2),
            idiosyncratic_var=np.ones(5),  # wrong: should be (4,)
        )


def test_post_init_rejects_non_positive_idiosyncratic_var():
    """All entries of idiosyncratic_var must be strictly positive."""
    with pytest.raises(FactorModelError, match="strictly positive"):
        FactorModel(
            factor_loadings=np.eye(3, 2),
            factor_covariance=np.eye(2),
            idiosyncratic_var=np.array([1.0, 0.0, 1.0]),  # zero entry
        )


def test_post_init_rejects_negative_idiosyncratic_var():
    """Negative entries in idiosyncratic_var must also raise FactorModelError."""
    with pytest.raises(FactorModelError, match="strictly positive"):
        FactorModel(
            factor_loadings=np.eye(3, 2),
            factor_covariance=np.eye(2),
            idiosyncratic_var=np.array([1.0, -0.5, 1.0]),
        )


# ─── n_assets and n_factors properties ───────────────────────────────────────


def test_n_assets_matches_loadings_rows():
    """n_assets must equal the number of rows in factor_loadings."""
    for n in (2, 5, 10):
        fm = _make_fm(n=n, k=2)
        assert fm.n_assets == n


def test_n_factors_matches_loadings_columns():
    """n_factors must equal the number of columns in factor_loadings."""
    for k in (1, 2, 4):
        fm = _make_fm(n=6, k=k)
        assert fm.n_factors == k


# ─── covariance property ─────────────────────────────────────────────────────


def test_covariance_shape():
    """Covariance must return an (n, n) array."""
    fm = _make_fm(n=5, k=2)
    assert fm.covariance.shape == (5, 5)


def test_covariance_matches_formula():
    """Covariance must equal B*F*B^T + diag(d) exactly."""
    b_mat = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    f_mat = np.array([[2.0, 0.0], [0.0, 3.0]])
    d = np.array([0.5, 0.5, 1.0])
    fm = FactorModel(factor_loadings=b_mat, factor_covariance=f_mat, idiosyncratic_var=d)
    expected = b_mat @ f_mat @ b_mat.T + np.diag(d)
    np.testing.assert_allclose(fm.covariance, expected)


def test_covariance_is_symmetric():
    """The reconstructed covariance matrix must be symmetric."""
    fm = _make_fm(n=6, k=3)
    cov = fm.covariance
    np.testing.assert_allclose(cov, cov.T, atol=1e-14)


def test_covariance_diagonal_equals_factor_plus_idio():
    """Each diagonal entry of covariance must equal (B*F*B^T)_ii + d_i."""
    fm = _make_fm(n=5, k=2)
    systematic_diag = np.einsum("ij,jj,ij->i", fm.factor_loadings, fm.factor_covariance, fm.factor_loadings)
    expected_diag = systematic_diag + fm.idiosyncratic_var
    np.testing.assert_allclose(fm.covariance.diagonal(), expected_diag, atol=1e-14)


# ─── from_returns classmethod ─────────────────────────────────────────────────


@pytest.fixture
def return_matrix() -> np.ndarray:
    """50 x 5 standard-normal return matrix."""
    return np.random.default_rng(7).standard_normal((50, 5))


def test_from_returns_shapes(return_matrix):
    """from_returns must produce a FactorModel with n=5, k=2."""
    fm = FactorModel.from_returns(return_matrix, k=2)
    assert fm.n_assets == 5
    assert fm.n_factors == 2
    assert fm.factor_loadings.shape == (5, 2)
    assert fm.factor_covariance.shape == (2, 2)
    assert fm.idiosyncratic_var.shape == (5,)


def test_from_returns_covariance_unit_diagonal(return_matrix):
    """Covariance from from_returns must have strictly positive diagonal entries."""
    fm = FactorModel.from_returns(return_matrix, k=2)
    diag = fm.covariance.diagonal()
    assert np.all(diag > 0)


def test_from_returns_idiosyncratic_var_all_positive(return_matrix):
    """All idiosyncratic_var entries must be strictly positive."""
    fm = FactorModel.from_returns(return_matrix, k=3)
    assert np.all(fm.idiosyncratic_var > 0)


def test_from_returns_factor_covariance_is_diagonal(return_matrix):
    """Factor covariance from SVD must be diagonal (off-diagonals are zero)."""
    fm = FactorModel.from_returns(return_matrix, k=2)
    off_diag = fm.factor_covariance - np.diag(np.diag(fm.factor_covariance))
    np.testing.assert_allclose(off_diag, np.zeros_like(off_diag), atol=1e-14)


def test_from_returns_factor_covariance_entries_positive(return_matrix):
    """Diagonal entries of factor_covariance must all be strictly positive."""
    fm = FactorModel.from_returns(return_matrix, k=2)
    assert np.all(np.diag(fm.factor_covariance) > 0)


def test_from_returns_k_equals_one(return_matrix):
    """from_returns must work for k=1 (single-factor / market model)."""
    fm = FactorModel.from_returns(return_matrix, k=1)
    assert fm.n_factors == 1
    assert fm.factor_loadings.shape == (5, 1)


def test_from_returns_k_equals_min_t_n():
    """from_returns must work when k equals min(T, n) (full rank)."""
    ret_mat = np.random.default_rng(0).standard_normal((10, 4))
    fm = FactorModel.from_returns(ret_mat, k=4)
    assert fm.n_factors == 4


def test_from_returns_raises_on_1d_input():
    """from_returns must raise FactorModelError when input is not 2-D."""
    with pytest.raises(FactorModelError, match="Return matrix must be 2-D"):
        FactorModel.from_returns(np.ones(10), k=1)


def test_from_returns_raises_on_k_zero():
    """from_returns must raise FactorModelError when k < 1."""
    ret_mat = np.random.default_rng(0).standard_normal((20, 4))
    with pytest.raises(FactorModelError, match="k must satisfy"):
        FactorModel.from_returns(ret_mat, k=0)


def test_from_returns_raises_on_k_too_large():
    """from_returns must raise FactorModelError when k > min(T, n)."""
    ret_mat = np.random.default_rng(0).standard_normal((20, 4))
    with pytest.raises(FactorModelError, match="k must satisfy"):
        FactorModel.from_returns(ret_mat, k=5)


def test_from_returns_public_api_import():
    """FactorModel must be importable from the basanos.math public namespace."""
    from basanos.math import FactorModel as PublicFactorModel

    assert PublicFactorModel is FactorModel


# ─── solve method ─────────────────────────────────────────────────────────────


def test_solve_matches_explicit_inversion():
    """Solve must return the same result as explicitly forming and inverting Cov."""
    fm = _make_fm(n=6, k=2)
    rng = np.random.default_rng(99)
    rhs = rng.standard_normal(6)

    x_woodbury = fm.solve(rhs)
    x_direct = np.linalg.solve(fm.covariance, rhs)

    np.testing.assert_allclose(x_woodbury, x_direct, atol=1e-10)


def test_solve_satisfies_linear_system():
    """Cov @ solve(rhs) must recover rhs to near machine precision."""
    fm = _make_fm(n=5, k=3)
    rng = np.random.default_rng(7)
    rhs = rng.standard_normal(5)

    x = fm.solve(rhs)
    residual = fm.covariance @ x - rhs

    np.testing.assert_allclose(residual, np.zeros(5), atol=1e-10)


def test_solve_output_shape():
    """Solve must return a 1-D array of length n."""
    fm = _make_fm(n=8, k=3)
    rhs = np.ones(8)
    x = fm.solve(rhs)
    assert x.shape == (8,)


def test_solve_identity_factor_model():
    """With B=0, Cov=D and solve must equal rhs / idiosyncratic_var element-wise."""
    n = 4
    d = np.array([2.0, 3.0, 4.0, 5.0])
    fm = FactorModel(
        factor_loadings=np.zeros((n, 1)),
        factor_covariance=np.eye(1),
        idiosyncratic_var=d,
    )
    rhs = np.array([1.0, 1.0, 1.0, 1.0])
    x = fm.solve(rhs)
    np.testing.assert_allclose(x, rhs / d, atol=1e-14)


def test_solve_k1_sherman_morrison():
    """For k=1 the Woodbury identity specialises to Sherman-Morrison; verify correctness."""
    rng = np.random.default_rng(13)
    n = 10
    b = rng.standard_normal((n, 1))
    f = np.array([[2.0]])
    d = np.abs(rng.standard_normal(n)) + 1.0
    fm = FactorModel(factor_loadings=b, factor_covariance=f, idiosyncratic_var=d)

    rhs = rng.standard_normal(n)
    x_woodbury = fm.solve(rhs)
    x_direct = np.linalg.solve(fm.covariance, rhs)

    np.testing.assert_allclose(x_woodbury, x_direct, atol=1e-10)


def test_solve_multiple_rhs_vectors():
    """Solve must be consistent when called independently with different rhs vectors."""
    fm = _make_fm(n=5, k=2)
    rng = np.random.default_rng(21)
    cov = fm.covariance

    for _ in range(5):
        rhs = rng.standard_normal(5)
        x = fm.solve(rhs)
        np.testing.assert_allclose(cov @ x, rhs, atol=1e-10)


@pytest.mark.parametrize("wrong_size", [3, 5])
def test_solve_raises_on_wrong_rhs_length(wrong_size: int):
    """Solve must raise DimensionMismatchError when rhs length != n_assets."""
    fm = _make_fm(n=4, k=2)
    with pytest.raises(DimensionMismatchError):
        fm.solve(np.ones(wrong_size))


def test_solve_from_returns_matches_explicit():
    """Solve on a from_returns model must match the explicit covariance solve."""
    rng = np.random.default_rng(55)
    ret = rng.standard_normal((100, 6))
    fm = FactorModel.from_returns(ret, k=2)
    rhs = rng.standard_normal(6)

    x_woodbury = fm.solve(rhs)
    x_direct = np.linalg.solve(fm.covariance, rhs)

    np.testing.assert_allclose(x_woodbury, x_direct, atol=1e-10)


def test_solve_raises_singular_matrix_error_when_cholesky_fails():
    """Solve must raise SingularMatrixError when _cholesky_solve raises LinAlgError."""
    fm = _make_fm(n=4, k=2)
    rhs = np.ones(4)
    with (
        patch(
            "basanos.math._factor_model._cholesky_solve",
            side_effect=np.linalg.LinAlgError("singular matrix"),
        ),
        pytest.raises(SingularMatrixError),
    ):
        fm.solve(rhs)


# ─── condition-number checks on factor_covariance ────────────────────────────


def test_solve_warns_ill_conditioned_factor_covariance():
    """IllConditionedMatrixWarning must fire when factor_covariance is ill-conditioned.

    A diagonal factor_covariance with entries (1.0, 1e-13) has condition
    number 1e13 > _DEFAULT_COND_THRESHOLD (1e12).  The warning must be
    emitted before the Woodbury solve proceeds.
    """
    n, k = 4, 2
    rng = np.random.default_rng(0)
    b_mat = rng.standard_normal((n, k))
    # Diagonal F with κ(F) ≈ 1e13 > 1e12.
    f_mat = np.diag([1.0, 1e-13])
    d = np.ones(n)
    fm = FactorModel(factor_loadings=b_mat, factor_covariance=f_mat, idiosyncratic_var=d)

    assert float(np.linalg.cond(f_mat)) > _DEFAULT_COND_THRESHOLD

    rhs = rng.standard_normal(n)
    with pytest.warns(IllConditionedMatrixWarning, match="condition number"):
        fm.solve(rhs)


def test_solve_does_not_warn_well_conditioned_factor_covariance():
    """No IllConditionedMatrixWarning for a well-conditioned factor_covariance."""
    fm = _make_fm(n=4, k=2)  # factor_covariance = np.eye(2), κ = 1
    rhs = np.ones(4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        fm.solve(rhs)  # must not raise


def test_solve_suppresses_warning_with_high_cond_threshold():
    """Setting cond_threshold=inf must suppress the IllConditionedMatrixWarning."""
    n, k = 4, 2
    rng = np.random.default_rng(1)
    b_mat = rng.standard_normal((n, k))
    f_mat = np.diag([1.0, 1e-13])  # κ ≈ 1e13, normally would trigger warning
    d = np.ones(n)
    fm = FactorModel(factor_loadings=b_mat, factor_covariance=f_mat, idiosyncratic_var=d)

    rhs = rng.standard_normal(n)
    with warnings.catch_warnings():
        warnings.simplefilter("error", IllConditionedMatrixWarning)
        # cond_threshold=np.inf disables the condition-number guard entirely
        fm.solve(rhs, cond_threshold=np.inf)  # must not raise


# ─── woodbury_condition_number diagnostic property ───────────────────────────


def test_woodbury_condition_number_is_positive():
    """woodbury_condition_number must return a strictly positive float."""
    fm = _make_fm(n=4, k=2)
    kappa = fm.woodbury_condition_number
    assert isinstance(kappa, float)
    assert kappa > 0.0


def test_woodbury_condition_number_identity_factor_model():
    """With F=I and B=0, M = I + 0 = I so condition number must be 1.0."""
    n = 4
    fm = FactorModel(
        factor_loadings=np.zeros((n, 1)),
        factor_covariance=np.eye(1),
        idiosyncratic_var=np.ones(n),
    )
    np.testing.assert_allclose(fm.woodbury_condition_number, 1.0, rtol=1e-10)


def test_woodbury_condition_number_increases_with_ill_conditioned_factor_covariance():
    """Ill-conditioned F must produce a large woodbury_condition_number."""
    rng = np.random.default_rng(7)
    n, k = 6, 2
    b_mat = rng.standard_normal((n, k))
    d = np.ones(n)

    # Well-conditioned F
    fm_good = FactorModel(
        factor_loadings=b_mat,
        factor_covariance=np.eye(k),
        idiosyncratic_var=d,
    )

    # Ill-conditioned F (κ(F) ≈ 1e12)
    fm_bad = FactorModel(
        factor_loadings=b_mat,
        factor_covariance=np.diag([1.0, 1e-12]),
        idiosyncratic_var=d,
    )

    assert fm_bad.woodbury_condition_number > fm_good.woodbury_condition_number


def test_woodbury_condition_number_consistent_with_solve_warning():
    """woodbury_condition_number > _DEFAULT_COND_THRESHOLD must coincide with IllConditionedMatrixWarning.

    When the inner Woodbury matrix is ill-conditioned (condition number exceeds
    ``_DEFAULT_COND_THRESHOLD``), both the warning from ``solve()`` and a large
    ``woodbury_condition_number`` must be observed.
    """
    rng = np.random.default_rng(3)
    n, k = 4, 2
    b_mat = rng.standard_normal((n, k))
    # F with κ(F) ≈ 1e13 — this will make the inner matrix ill-conditioned too.
    f_mat = np.diag([1.0, 1e-13])
    d = np.ones(n)
    fm = FactorModel(factor_loadings=b_mat, factor_covariance=f_mat, idiosyncratic_var=d)

    assert fm.woodbury_condition_number > _DEFAULT_COND_THRESHOLD

    rhs = rng.standard_normal(n)
    with pytest.warns(IllConditionedMatrixWarning, match="condition number"):
        fm.solve(rhs)


def test_woodbury_condition_number_returns_inf_for_singular_factor_covariance():
    """woodbury_condition_number must return inf when factor_covariance is singular."""
    n, k = 4, 2
    b_mat = np.eye(n, k)
    # A singular factor_covariance (rank 1 but shape 2x2 — zero second row/col)
    f_singular = np.array([[1.0, 0.0], [0.0, 0.0]])
    d = np.ones(n)
    fm = FactorModel(
        factor_loadings=b_mat,
        factor_covariance=f_singular,
        idiosyncratic_var=d,
    )

    kappa = fm.woodbury_condition_number
    assert not np.isfinite(kappa)
