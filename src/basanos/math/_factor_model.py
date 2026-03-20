r"""Factor risk model decomposition (Section 4.1 of basanos.pdf).

This private module provides the :class:`FactorModel` frozen dataclass, which
encapsulates the three-component factor model

.. math::

    \\bm{\\Sigma} = \\mathbf{B}\\mathbf{F}\\mathbf{B}^\\top + \\mathbf{D}

and a class method for fitting the model from a return matrix via the
Singular Value Decomposition (Section 4.2).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from basanos.exceptions import (
    DimensionMismatchError,
    FactorModelError,
    SingularMatrixError,
)
from basanos.math._linalg import (
    _DEFAULT_COND_THRESHOLD,
    _check_and_warn_condition,
    _cholesky_solve,
)


@dataclasses.dataclass(frozen=True)
class FactorModel:
    r"""Frozen dataclass for a factor risk model decomposition (Section 4.1).

    Encapsulates the three components of the factor model

    .. math::

        \bm{\Sigma} = \mathbf{B}\mathbf{F}\mathbf{B}^\top + \mathbf{D}

    where

    - :math:`\mathbf{B} \in \mathbb{R}^{n \times k}` is the *factor loading
      matrix*: column :math:`j` gives the sensitivity of each asset to
      factor :math:`j`.
    - :math:`\mathbf{F} \in \mathbb{R}^{k \times k}` is the *factor covariance
      matrix* (positive definite), capturing how the :math:`k` factors
      co-vary.
    - :math:`\mathbf{D} = \operatorname{diag}(d_1, \dots, d_n)` with
      :math:`d_i > 0` is the *idiosyncratic variance* diagonal, capturing
      the asset-specific variance unexplained by the common factors.

    The central assumption is :math:`k \ll n`: the dominant systematic sources
    of risk are captured by a handful of factors while the idiosyncratic
    component is, by construction, uncorrelated across assets.

    Attributes:
        factor_loadings: Factor loading matrix :math:`\mathbf{B}`,
            shape ``(n, k)``.
        factor_covariance: Factor covariance matrix :math:`\mathbf{F}`,
            shape ``(k, k)``.
        idiosyncratic_var: Idiosyncratic variance vector
            :math:`(d_1, \dots, d_n)`, shape ``(n,)``.  All entries must be
            strictly positive.

    Examples:
        >>> import numpy as np
        >>> loadings = np.eye(3, 2)
        >>> cov = np.eye(2) * 0.5
        >>> idio = np.array([0.5, 0.5, 1.0])
        >>> fm = FactorModel(factor_loadings=loadings, factor_covariance=cov, idiosyncratic_var=idio)
        >>> fm.n_assets
        3
        >>> fm.n_factors
        2
        >>> fm.covariance.shape
        (3, 3)
    """

    factor_loadings: np.ndarray
    factor_covariance: np.ndarray
    idiosyncratic_var: np.ndarray

    def __post_init__(self) -> None:
        """Validate shape consistency and strict positivity after initialization.

        Raises:
            FactorModelError: If ``factor_loadings`` is not 2-D.
            FactorModelError: If ``factor_covariance`` shape does not
                match the number of factors inferred from ``factor_loadings``.
            FactorModelError: If ``idiosyncratic_var`` length does
                not match the number of assets inferred from ``factor_loadings``.
            FactorModelError: If any element of
                ``idiosyncratic_var`` is not strictly positive.
        """
        if self.factor_loadings.ndim != 2:
            raise FactorModelError(f"factor_loadings must be 2-D, got ndim={self.factor_loadings.ndim}.")  # noqa: TRY003
        n, k = self.factor_loadings.shape
        if self.factor_covariance.shape != (k, k):
            raise FactorModelError(  # noqa: TRY003
                f"factor_covariance must have shape ({k}, {k}) to match "
                f"factor_loadings columns, got {self.factor_covariance.shape}."
            )
        if self.idiosyncratic_var.shape != (n,):
            raise FactorModelError(  # noqa: TRY003
                f"idiosyncratic_var must have shape ({n},) to match factor_loadings rows, "
                f"got {self.idiosyncratic_var.shape}."
            )
        if not np.all(self.idiosyncratic_var > 0):
            raise FactorModelError("All entries of idiosyncratic_var must be strictly positive.")  # noqa: TRY003

    @property
    def n_assets(self) -> int:
        """Number of assets *n* (rows of ``factor_loadings``)."""
        return self.factor_loadings.shape[0]

    @property
    def n_factors(self) -> int:
        """Number of factors *k* (columns of ``factor_loadings``)."""
        return self.factor_loadings.shape[1]

    @property
    def covariance(self) -> np.ndarray:
        r"""Reconstruct the full :math:`n \times n` covariance matrix.

        Computes :math:`\bm{\Sigma} = \mathbf{B}\mathbf{F}\mathbf{B}^\top +
        \mathbf{D}` by combining the low-rank systematic component with the
        diagonal idiosyncratic component.

        Returns:
            np.ndarray: Shape ``(n, n)`` symmetric covariance matrix.

        Examples:
            >>> import numpy as np
            >>> loadings = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
            >>> cov = np.eye(2)
            >>> idio = np.ones(3)
            >>> fm = FactorModel(factor_loadings=loadings, factor_covariance=cov, idiosyncratic_var=idio)
            >>> fm.covariance.diagonal().tolist()
            [2.0, 2.0, 1.0]
        """
        return self.factor_loadings @ self.factor_covariance @ self.factor_loadings.T + np.diag(self.idiosyncratic_var)

    def solve(
        self,
        rhs: np.ndarray,
        cond_threshold: float = _DEFAULT_COND_THRESHOLD,
    ) -> np.ndarray:
        r"""Solve :math:`\bm{\Sigma}\,\mathbf{x} = \mathbf{b}` via the Woodbury identity.

        Applies the Sherman--Morrison--Woodbury formula (Section 4.3 of
        basanos.pdf) to avoid forming or factorising the full
        :math:`n \times n` covariance matrix:

        .. math::

            (\mathbf{D} + \mathbf{B}\mathbf{F}\mathbf{B}^\top)^{-1}
            = \mathbf{D}^{-1}
              - \mathbf{D}^{-1}\mathbf{B}
                \bigl(\mathbf{F}^{-1} + \mathbf{B}^\top\mathbf{D}^{-1}\mathbf{B}\bigr)^{-1}
                \mathbf{B}^\top\mathbf{D}^{-1}.

        Because :math:`\mathbf{D}` is diagonal, :math:`\mathbf{D}^{-1}` is
        free.  The inner matrix is :math:`k \times k` with cost
        :math:`O(k^3)`, and the surrounding multiplications cost
        :math:`O(kn)`.  Total cost is :math:`O(k^3 + kn)` rather than
        :math:`O(n^3)`.

        Args:
            rhs: Right-hand side vector :math:`\mathbf{b}`, shape ``(n,)``.
            cond_threshold: Condition-number threshold above which an
                :class:`~basanos.exceptions.IllConditionedMatrixWarning` is
                emitted for the inner :math:`k \times k` system.  Defaults to
                ``1e12``.

        Returns:
            np.ndarray: Solution vector :math:`\mathbf{x}`, shape ``(n,)``.

        Raises:
            DimensionMismatchError: If ``rhs`` length does not match
                ``n_assets``.
            SingularMatrixError: If the inner :math:`k \times k` matrix is
                singular.

        Examples:
            >>> import numpy as np
            >>> loadings = np.eye(3, 1)
            >>> cov = np.eye(1)
            >>> idio = np.ones(3)
            >>> fm = FactorModel(factor_loadings=loadings, factor_covariance=cov, idiosyncratic_var=idio)
            >>> rhs = np.array([1.0, 2.0, 3.0])
            >>> x = fm.solve(rhs)
            >>> np.allclose(fm.covariance @ x, rhs)
            True
        """
        n = self.n_assets
        if rhs.shape != (n,):
            raise DimensionMismatchError(rhs.size, n)

        # D^{-1} is free because D is diagonal
        d_inv = 1.0 / self.idiosyncratic_var  # (n,)
        d_inv_rhs = d_inv * rhs  # D^{-1} b, shape (n,)
        d_inv_b_mat = d_inv[:, None] * self.factor_loadings  # D^{-1} B, shape (n, k)

        # Solve mid * w = B^T D^{-1} b, where mid = F^{-1} + B^T D^{-1} B.
        # F^{-1} is obtained via a Cholesky solve rather than an explicit
        # inversion, consistent with the Cholesky-first discipline in _linalg.py.
        rhs_k = self.factor_loadings.T @ d_inv_rhs  # (k,)
        try:
            mid = (
                _cholesky_solve(self.factor_covariance, np.eye(self.n_factors)) + self.factor_loadings.T @ d_inv_b_mat
            )  # (k, k)
            _check_and_warn_condition(mid, cond_threshold)
            w = _cholesky_solve(mid, rhs_k)  # (k,)
        except np.linalg.LinAlgError as exc:
            raise SingularMatrixError(str(exc)) from exc

        # x = D^{-1} b - D^{-1} B w
        return d_inv_rhs - d_inv_b_mat @ w

    @classmethod
    def from_returns(cls, returns: np.ndarray, k: int) -> FactorModel:
        r"""Fit a rank-*k* factor model from a return matrix via truncated SVD.

        Extracts latent factors from the return matrix
        :math:`\mathbf{R} \in \mathbb{R}^{T \times n}` using the Singular
        Value Decomposition (SVD).  The top-*k* singular triplets define the
        factor model components:

        .. math::

            \mathbf{B} = \mathbf{V}_k, \quad
            \mathbf{F} = \bm{\Sigma}_k^2 / T, \quad
            \hat{d}_i = 1 - \bigl(\mathbf{B}\mathbf{F}\mathbf{B}^\top\bigr)_{ii}

        where :math:`\mathbf{V}_k` and :math:`\bm{\Sigma}_k` are the top-*k*
        right singular vectors and singular values of :math:`\mathbf{R}`
        respectively.  When *returns* contains unit-variance columns (as
        produced by :func:`~basanos.math._signal.vol_adj`), the sample
        covariance has unit diagonal; the idiosyncratic term
        :math:`\hat{d}_i = 1 - (\mathbf{B}\mathbf{F}\mathbf{B}^\top)_{ii}`
        absorbs the residual so the full covariance :math:`\hat{\mathbf{C}}^{(k)}`
        also has unit diagonal.  Each :math:`\hat{d}_i` is clamped from below
        at machine epsilon to guarantee strict positivity.

        Args:
            returns: Return matrix of shape ``(T, n)``, typically
                volatility-adjusted log returns with rows as timestamps and
                columns as assets.
            k: Number of factors to retain.  Must satisfy
                ``1 <= k <= min(T, n)``.

        Returns:
            FactorModel: Fitted factor model with ``n_assets = n`` and
                ``n_factors = k``.

        Raises:
            FactorModelError: If *returns* is not 2-D.
            FactorModelError: If *k* is outside the range ``[1, min(T, n)]``.

        Examples:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> ret = rng.standard_normal((50, 5))
            >>> fm = FactorModel.from_returns(ret, k=2)
            >>> fm.n_factors
            2
            >>> fm.n_assets
            5
            >>> fm.covariance.shape
            (5, 5)
        """
        if returns.ndim != 2:
            raise FactorModelError(f"Return matrix must be 2-D, got ndim={returns.ndim}.")  # noqa: TRY003
        t_len, n = returns.shape
        if not (1 <= k <= min(t_len, n)):
            raise FactorModelError(f"k must satisfy 1 <= k <= min(T, n) = {min(t_len, n)}, got k={k}.")  # noqa: TRY003

        _, s, vt = np.linalg.svd(returns, full_matrices=False)

        # Top-k right singular vectors as columns: shape (n, k)
        v_k = vt[:k].T
        s_k = s[:k]

        # Factor covariance: diagonal matrix with entries s_j**2 / T
        factor_cov = np.diag(s_k**2 / t_len)

        # Diagonal of B*F*B^T = sum_j (s_j**2/T) * B[:,j]**2
        factor_diag = (v_k**2) @ (s_k**2 / t_len)

        # Idiosyncratic variance: target diagonal is 1.0 (unit-variance columns
        # assumed); residual = 1.0 - systematic contribution, clamped to (0, inf)
        _unit_variance = 1.0
        d = np.maximum(_unit_variance - factor_diag, np.finfo(float).eps)

        return cls(factor_loadings=v_k, factor_covariance=factor_cov, idiosyncratic_var=d)
