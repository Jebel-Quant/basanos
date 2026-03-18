r"""Factor model for structured covariance estimation (private to basanos.math).

This module provides :class:`FactorModel`, a frozen dataclass that represents
the covariance structure of an N-asset portfolio using K latent factors:

.. math::

    \\Sigma = B \\cdot F \\cdot B^T + D

where

* :math:`B` is the ``(N, K)`` factor loading matrix,
* :math:`F` is the ``(K, K)`` factor covariance matrix,
* :math:`D` is a diagonal matrix of asset-specific (idiosyncratic) variances.

The key computational benefit is the :meth:`FactorModel.solve` method, which
uses the **Woodbury matrix identity** to solve :math:`\\Sigma x = v` in
:math:`O(N K + K^3)` time rather than :math:`O(N^3)`.  For K << N this
yields dramatic speedups: at N = 1000, K = 10 the solve is ~10 000x cheaper
than a full Cholesky decomposition.

This module is considered an internal implementation detail of
``basanos.math``.  Import the public symbol :class:`~basanos.math.FactorModel`
from ``basanos.math`` instead.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from basanos.exceptions import (
    DimensionMismatchError,
    FactorModelDimensionError,
    IllConditionedMatrixWarning,
)

_ILL_CONDITIONED_THRESHOLD: float = 1e10
"""Condition number above which IllConditionedMatrixWarning is emitted for the Woodbury middle matrix M."""

_MIN_SPECIFIC_VARIANCE: float = 1e-12
"""Floor applied to auto-computed specific variances to keep Σ positive-definite."""


def _woodbury_solve(
    B: np.ndarray,  # noqa: N803
    F: np.ndarray,  # noqa: N803
    d: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    r"""Core Woodbury solve: :math:`(D + B F B^T)^{-1}` rhs, no validation.

    This is the computation kernel shared by :meth:`FactorModel.solve` and the
    optimizer's per-timestamp loop.  Callers are responsible for ensuring that
    shapes are consistent and ``d`` contains only strictly positive values.

    Args:
        B: ``(N, K)`` factor loading matrix.
        F: ``(K, K)`` factor covariance matrix.
        d: ``(N,)`` specific variances (all strictly positive).
        rhs: ``(N,)`` right-hand side vector.

    Returns:
        np.ndarray: Solution vector of length N.
    """
    k = B.shape[1]

    d_inv = 1.0 / d  # (N,)
    d_inv_v = d_inv * rhs  # (N,)

    # F^{-1} via Cholesky (LU fallback for non-PD inputs)
    try:
        f_cho = cho_factor(F, check_finite=False)
        f_inv = cho_solve(f_cho, np.eye(k), check_finite=False)
    except np.linalg.LinAlgError:
        f_inv = np.linalg.solve(F, np.eye(k))

    # M = F^{-1} + B^T D^{-1} B  (K, K)
    bt_dinv_b = (B.T * d_inv) @ B
    m_mid = f_inv + bt_dinv_b

    # Warn if M is ill-conditioned before attempting the solve
    cond_m = float(np.linalg.cond(m_mid))
    if cond_m > _ILL_CONDITIONED_THRESHOLD:
        warnings.warn(
            f"Woodbury middle matrix M is ill-conditioned (cond={cond_m:.2e}); "
            "solve result may be inaccurate. Check factor model loadings and specific variances.",
            IllConditionedMatrixWarning,
            stacklevel=2,
        )

    # M^{-1} (B^T D^{-1} v)  (K,)
    bt_dinv_v = B.T @ d_inv_v
    try:
        m_cho = cho_factor(m_mid, check_finite=False)
        m_inv_bt_dinv_v = cho_solve(m_cho, bt_dinv_v, check_finite=False)
    except np.linalg.LinAlgError:
        m_inv_bt_dinv_v = np.linalg.solve(m_mid, bt_dinv_v)

    correction = d_inv * (B @ m_inv_bt_dinv_v)
    return d_inv_v - correction


@dataclasses.dataclass(frozen=True)
class FactorModel:
    r"""Structured covariance model: :math:`\Sigma = B \cdot F \cdot B^T + D`.

    Represents the covariance structure of an N-asset portfolio using K < N
    latent factors.  The N-by-N covariance matrix is never formed explicitly
    during optimization; the :meth:`solve` method applies the Woodbury matrix
    identity to solve the linear system in :math:`O(N K + K^3)` time.

    **Factor model decomposition**

    .. math::

        \\Sigma = B F B^T + D

    where:

    * :math:`B` (``(N, K)``) — factor loading matrix; row *i* contains asset
      *i*'s exposures to each of the *K* common factors.
    * :math:`F` (``(K, K)``) — factor covariance matrix.  Defaults to
      :math:`I_K`, implying orthogonal unit-variance factors.
    * :math:`D` (diagonal, ``(N,)`` stored as a 1-D array) — specific
      (idiosyncratic) variances.  When ``None`` the default is chosen so that
      the diagonal of :math:`\\Sigma` is exactly 1:

      .. math::

          d_i = \\max\\!\\left(\\varepsilon,\\; 1 - (B F B^T)_{ii}\\right)

      where :math:`\\varepsilon = 10^{-12}` prevents a singular diagonal.

    **Woodbury solve**

    :meth:`solve` computes :math:`\\Sigma^{-1} v` via

    .. math::

        (D + B F B^T)^{-1} v
        = D^{-1} v
          - D^{-1} B \\underbrace{(F^{-1} + B^T D^{-1} B)}_{M}^{-1} B^T D^{-1} v

    The :math:`K \\times K` matrix :math:`M` is solved with Cholesky
    decomposition; :math:`D^{-1}` is trivially the element-wise reciprocal of
    the diagonal.  Total cost: :math:`O(N K + K^3)`.

    **Memory efficiency**

    Unlike the EWMA correlation path (which allocates roughly
    :math:`112 \\cdot T \\cdot N^2` bytes for a T-by-N-by-N tensor), a factor
    model only stores :math:`O(N K + K^2)` parameters.  For N = 500 and K = 20
    this is approximately 80 000 float64 values (640 KB) versus 28 GB for the
    full tensor.

    Args:
        loadings: ``(N, K)`` factor loading matrix.  Must be 2-dimensional.
        factor_covariance: ``(K, K)`` symmetric positive-definite factor
            covariance matrix.  When ``None``, defaults to :math:`I_K`.
        specific_variances: ``(N,)`` vector of idiosyncratic variances
            (diagonal of D).  All values must be strictly positive.
            When ``None``, defaults to
            :math:`\\max(10^{-12},\\; 1 - \\operatorname{diag}(B F B^T))`.

    Raises:
        FactorModelDimensionError: If ``loadings`` is not 2-dimensional.
        FactorModelDimensionError: If ``factor_covariance`` is not a square
            ``(K, K)`` matrix consistent with ``loadings``, or is not symmetric
            (max element-wise asymmetry exceeds ``1e-10``).
        FactorModelDimensionError: If ``specific_variances`` length does not
            equal ``n_assets``, or any value is not strictly positive.

    Examples:
        Construct with default specific variances (diagonal of Σ will be 1):

        >>> import numpy as np
        >>> B = np.array([[0.8, 0.2], [0.6, 0.5], [0.3, 0.9]])
        >>> fm = FactorModel(loadings=B)
        >>> fm.n_assets
        3
        >>> fm.n_factors
        2
        >>> diag = np.diag(fm.to_matrix())
        >>> np.allclose(diag, 1.0)
        True

        Solve a linear system using the Woodbury identity:

        >>> rhs = np.array([1.0, 0.0, -1.0])
        >>> x = fm.solve(rhs)
        >>> x_ref = np.linalg.solve(fm.to_matrix(), rhs)
        >>> np.allclose(x, x_ref, rtol=1e-10)
        True
    """

    loadings: np.ndarray
    factor_covariance: np.ndarray | None = None
    specific_variances: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate array shapes and specific-variance positivity."""
        if self.loadings.ndim != 2:
            raise FactorModelDimensionError(  # noqa: TRY003
                f"loadings must be a 2-D array of shape (N, K), "
                f"got {self.loadings.ndim}-D array with shape {self.loadings.shape}."
            )

        n, k = self.loadings.shape

        if self.factor_covariance is not None:
            fc = self.factor_covariance
            if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
                raise FactorModelDimensionError(  # noqa: TRY003
                    f"factor_covariance must be a square (K, K) matrix, got shape {fc.shape}."
                )
            if fc.shape[0] != k:
                raise FactorModelDimensionError(  # noqa: TRY003
                    f"factor_covariance shape {fc.shape} is incompatible with "
                    f"loadings shape {self.loadings.shape}: expected ({k}, {k})."
                )
            asymmetry = float(np.max(np.abs(fc - fc.T)))
            if asymmetry > 1e-10:
                raise FactorModelDimensionError(  # noqa: TRY003
                    f"factor_covariance must be symmetric; got max asymmetry {asymmetry:.3e}."
                )

        if self.specific_variances is not None:
            sv = self.specific_variances
            if sv.ndim != 1:
                raise FactorModelDimensionError(  # noqa: TRY003
                    f"specific_variances must be a 1-D array of length N={n}, "
                    f"got {sv.ndim}-D array with shape {sv.shape}."
                )
            if sv.shape[0] != n:
                raise FactorModelDimensionError(  # noqa: TRY003
                    f"specific_variances length {sv.shape[0]} does not match "
                    f"n_assets={n} (inferred from loadings shape)."
                )
            if np.any(sv <= 0.0):
                raise FactorModelDimensionError(  # noqa: TRY003
                    f"All specific_variances must be strictly positive; got minimum value {float(np.min(sv)):.6g}."
                )

    @property
    def n_assets(self) -> int:
        """Number of assets N (rows in the loadings matrix).

        Examples:
            >>> import numpy as np
            >>> FactorModel(loadings=np.zeros((5, 2))).n_assets
            5
        """
        return int(self.loadings.shape[0])

    @property
    def n_factors(self) -> int:
        """Number of factors K (columns in the loadings matrix).

        Examples:
            >>> import numpy as np
            >>> FactorModel(loadings=np.zeros((5, 2))).n_factors
            2
        """
        return int(self.loadings.shape[1])

    def __repr__(self) -> str:
        """Return a compact, human-readable summary of the model dimensions.

        Examples:
            >>> import numpy as np
            >>> FactorModel(loadings=np.zeros((5, 2)))
            FactorModel(n_assets=5, n_factors=2, factor_covariance=identity, specific_variances=auto)
            >>> FactorModel(loadings=np.zeros((5, 2)), factor_covariance=np.eye(2), specific_variances=np.ones(5))
            FactorModel(n_assets=5, n_factors=2, factor_covariance=custom, specific_variances=custom)
        """
        fc = "custom" if self.factor_covariance is not None else "identity"
        sv = "custom" if self.specific_variances is not None else "auto"
        return (
            f"FactorModel(n_assets={self.n_assets}, n_factors={self.n_factors}, "
            f"factor_covariance={fc}, specific_variances={sv})"
        )

    def _resolved_factor_covariance(self) -> np.ndarray:
        """Return factor covariance matrix, falling back to the identity."""
        if self.factor_covariance is None:
            return np.eye(self.n_factors)
        return self.factor_covariance

    def _resolved_specific_variances(self) -> np.ndarray:
        """Return specific variances, computing defaults so diag(Σ) = 1."""
        if self.specific_variances is not None:
            return self.specific_variances
        F = self._resolved_factor_covariance()  # noqa: N806
        # diag(B·F·B^T)_i = sum_{k,l} B_{ik} F_{kl} B_{il}
        bfbt_diag = np.einsum("ik,kl,il->i", self.loadings, F, self.loadings)
        return np.maximum(1.0 - bfbt_diag, _MIN_SPECIFIC_VARIANCE)

    def to_matrix(self) -> np.ndarray:
        """Materialise the full N-by-N covariance matrix Σ = B·F·B^T + D.

        For small universes, debugging, or visualisation.  Avoid calling this
        for large N — the benefit of the factor model lies precisely in *not*
        forming the full matrix.

        Returns:
            np.ndarray: Symmetric positive-definite ``(N, N)`` matrix.

        Examples:
            >>> import numpy as np
            >>> B = np.array([[1.0, 0.0], [0.0, 1.0]])
            >>> sv = np.array([0.5, 0.5])
            >>> FactorModel(loadings=B, specific_variances=sv).to_matrix().tolist()
            [[1.5, 0.0], [0.0, 1.5]]
        """
        F = self._resolved_factor_covariance()  # noqa: N806
        d = self._resolved_specific_variances()
        bfbt = self.loadings @ F @ self.loadings.T
        return bfbt + np.diag(d)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        r"""Solve Σ·x = rhs via the Woodbury matrix identity.

        Computes :math:`\Sigma^{-1} v` without forming the full N-by-N matrix:

        .. math::

            (D + B F B^T)^{-1} v
            = D^{-1} v
              - D^{-1} B M^{-1} B^T D^{-1} v, \quad
              M = F^{-1} + B^T D^{-1} B

        Both the :math:`K \times K` matrix :math:`M` and the factor covariance
        :math:`F` are solved with Cholesky decomposition (with LU fallback for
        non-positive-definite inputs).

        Args:
            rhs: 1-D array of length N.

        Returns:
            np.ndarray: Solution vector of length N.

        Raises:
            DimensionMismatchError: If ``rhs`` length does not equal ``n_assets``.

        Warns:
            IllConditionedMatrixWarning: If the Woodbury middle matrix
                :math:`M = F^{-1} + B^T D^{-1} B` has a condition number
                exceeding ``1e10``, indicating the solve result may be
                numerically unreliable.

        Examples:
            >>> import numpy as np
            >>> B = np.array([[0.8, 0.2], [0.3, 0.7]])
            >>> fm = FactorModel(loadings=B)
            >>> rhs = np.array([1.0, 0.0])
            >>> x = fm.solve(rhs)
            >>> np.allclose(x, np.linalg.solve(fm.to_matrix(), rhs), rtol=1e-10)
            True
        """
        if rhs.shape[0] != self.n_assets:
            raise DimensionMismatchError(rhs.shape[0], self.n_assets)

        return _woodbury_solve(
            self.loadings,
            self._resolved_factor_covariance(),
            self._resolved_specific_variances(),
            rhs,
        )
