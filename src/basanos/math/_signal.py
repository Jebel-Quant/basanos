"""Internal signal utilities (private to basanos.math).

This module contains low-level helpers for building signals and
transformations. It is considered an internal implementation detail of
``basanos.math``. Do not import this module directly from outside the
package; instead import the public symbols from ``basanos.math``.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def factor_extract(ret_mat: np.ndarray, k: int) -> np.ndarray:
    r"""Extract *k* latent factors from the returns matrix via truncated SVD.

    Performs a full (economy-size) SVD on *ret_mat* and returns the first *k* left
    singular vectors as the factor matrix :math:`C \in \mathbb{R}^{n \times k}`.
    These columns are the *k* principal directions that capture the most variance
    in the return history.

    .. math::

        R = U \, S \, V^{\top}, \quad C = U_{:, \, 0:k}

    Args:
        ret_mat: Returns matrix of shape ``(n, m)`` — *n* lookback rows, *m* assets.
        k: Number of factors to extract.  Must satisfy ``1 <= k <= min(n, m)``.

    Returns:
        np.ndarray: Factor matrix of shape ``(n, k)``.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> ret_mat = rng.normal(size=(20, 5))
        >>> factor_mat = factor_extract(ret_mat, k=3)
        >>> factor_mat.shape
        (20, 3)
    """
    left_sv, _, _ = np.linalg.svd(ret_mat, full_matrices=False)
    return left_sv[:, :k]


def factor_project(factor_mat: np.ndarray, ret_mat: np.ndarray) -> np.ndarray:
    r"""Project asset returns onto the factor space.

    Computes :math:`W = C^{\top} \cdot R \in \mathbb{R}^{k \times m}`, the
    projection of all *m* assets onto the *k*-dimensional factor space defined
    by the columns of the factor matrix.  Each row of *W* is one factor's
    exposure across the asset universe; each column is one asset's exposures
    across all factors.

    Args:
        factor_mat: Factor matrix of shape ``(n, k)`` — typically the output of
            :func:`factor_extract`.
        ret_mat: Returns matrix of shape ``(n, m)`` — must share the same *n*.

    Returns:
        np.ndarray: Projection matrix of shape ``(k, m)``.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> ret_mat = rng.normal(size=(20, 5))
        >>> factor_mat = factor_extract(ret_mat, k=3)
        >>> projection = factor_project(factor_mat, ret_mat)
        >>> projection.shape
        (3, 5)
    """
    return factor_mat.T @ ret_mat


def shrink2id(matrix: np.ndarray, lamb: float = 1.0) -> np.ndarray:
    r"""Shrink a square matrix linearly towards the identity matrix.

    This implements the **convex linear shrinkage** estimator

    .. math::

        \\hat{\\Sigma}(\\lambda) = \\lambda \\cdot M + (1 - \\lambda) \\cdot I_n

    where :math:`M` is the sample matrix, :math:`I_n` is the :math:`n \\times n`
    identity matrix, and :math:`\\lambda \\in [0, 1]` is the *retention weight*
    (equivalently, ``1 - lambda`` is the *shrinkage intensity*).

    **Why shrink toward the identity?**

    Sample covariance/correlation matrices estimated from a finite number of
    observations :math:`T` are poorly conditioned when the number of assets
    :math:`n` is large relative to :math:`T`.  This is the classical
    *curse of dimensionality*: extreme eigenvalues of the sample matrix are
    biased away from their population counterparts (the Marchenko-Pastur law
    describes the bias as a function of the concentration ratio :math:`n / T`).
    Shrinkage pulls eigenvalues toward a common target — here the unit sphere —
    reducing estimation error at the cost of a small bias [1]_.

    **Relationship to Ledoit-Wolf shrinkage**

    Ledoit and Wolf (2004) [2]_ derive the *optimal* scalar shrinkage
    intensity :math:`\\alpha^*` by minimizing the expected Frobenius loss
    :math:`\\mathbb{E}[\\|\\hat{\\Sigma}(\\alpha) - \\Sigma\\|_F^2]` under a
    general factor model.  Their closed-form estimator is a special case of
    this function where ``lamb = 1 - alpha*``.  The Oracle Approximating
    Shrinkage (OAS) estimator [3]_ improves on Ledoit-Wolf by accounting for
    the bias in the analytic formula, often yielding better finite-sample
    performance.

    **Basanos usage**

    In Basanos the target matrix is always the *correlation* identity (diagonal
    ones, off-diagonal zeros), and ``lamb`` is supplied via
    :attr:`~basanos.math.BasanosConfig.shrink` as a user-controlled
    hyperparameter rather than an analytically chosen optimal value.  This is
    appropriate in the context of *regularising a solver* (the system
    :math:`C x = \\mu` must be well-posed at every timestamp) rather than
    *estimating a covariance matrix* — here practical stability often matters
    more than minimum Frobenius loss.

    **Empirical guidance for choosing** ``lamb`` **(= cfg.shrink)**

    The table below offers practical starting points for daily financial return
    data.  All recommendations should be validated on out-of-sample data.

    +--------------------------+---------------------------+------------------+
    | Regime                   | Suggested ``lamb``        | Rationale        |
    +==========================+===========================+==================+
    | Many assets, short       | 0.3 - 0.6                 | High             |
    | lookback (n/T > 0.5)     |                           | concentration    |
    |                          |                           | ratio; sample    |
    |                          |                           | matrix is noisy. |
    +--------------------------+---------------------------+------------------+
    | Moderate assets,         | 0.5 - 0.8                 | Balanced         |
    | moderate lookback        |                           | regularisation.  |
    | (n/T ~ 0.1 - 0.5)        |                           |                  |
    +--------------------------+---------------------------+------------------+
    | Few assets, long         | 0.7 - 1.0                 | Sample matrix    |
    | lookback (n/T < 0.1)     |                           | is reliable;     |
    |                          |                           | light shrinkage  |
    |                          |                           | for robustness.  |
    +--------------------------+---------------------------+------------------+

    A simple heuristic: start with ``lamb = 1 - n / (2 * T)`` where
    ``n`` is the number of assets and ``T`` is the EWMA correlation lookback
    (``cfg.corr``) — a rough approximation of the Ledoit-Wolf formula —
    then tune on held-out data.

    **Sensitivity note**

    Shrinkage is most sensitive in the range :math:`\\lambda \\in [0.3, 0.8]`.
    Below ~0.3 the matrix can become nearly singular for small portfolios
    (``n > 10`` with ``corr < 50``); above ~0.8 the off-diagonal correlations
    are so heavily damped that the optimizer behaves almost as if assets were
    uncorrelated.

    Args:
        matrix: Square matrix to shrink (typically a correlation matrix).
        lamb: Retention weight :math:`\\lambda \\in [0, 1]`.  ``1.0`` returns
            the original matrix unchanged; ``0.0`` returns the identity.

    Returns:
        The shrunk matrix with the same shape as ``matrix``.

    References:
        .. [1] Stein, C. (1956). *Inadmissibility of the usual estimator for
               the mean of a multivariate normal distribution.*  Proceedings
               of the Third Berkeley Symposium, 1, 197-206.
        .. [2] Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for
               large-dimensional covariance matrices.*  Journal of Multivariate
               Analysis, 88(2), 365-411.
               https://doi.org/10.1016/S0047-259X(03)00096-4
        .. [3] Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010).
               *Shrinkage algorithms for MMSE covariance estimation.*  IEEE
               Transactions on Signal Processing, 58(10), 5016-5029.
               https://doi.org/10.1109/TSP.2010.2053029

    Examples:
        >>> import numpy as np
        >>> # Full retention: original matrix unchanged
        >>> shrink2id(np.array([[2.0, 1.0], [1.0, 3.0]]), lamb=1.0).tolist()
        [[2.0, 1.0], [1.0, 3.0]]
        >>> # Full shrinkage: identity matrix
        >>> shrink2id(np.array([[2.0, 0.0], [0.0, 2.0]]), lamb=0.0).tolist()
        [[1.0, 0.0], [0.0, 1.0]]
        >>> # Half-way: average of matrix and identity
        >>> m = np.array([[2.0, 1.0], [1.0, 3.0]])
        >>> shrink2id(m, lamb=0.5).tolist()
        [[1.5, 0.5], [0.5, 2.0]]
    """
    return matrix * lamb + (1 - lamb) * np.eye(N=matrix.shape[0])


def vol_adj(x: pl.Expr, vola: int, clip: float, min_samples: int = 1) -> pl.Expr:
    """Compute clipped, volatility-adjusted log returns per column.

    - ``vola`` controls the EWM std smoothing (converted to alpha internally).
    - ``clip`` applies symmetric clipping to the standardized returns.

    Args:
        x: Polars expression (price series) to transform.
        vola: EWMA lookback (span-equivalent) for std.
        clip: Symmetric clipping threshold applied after standardization.
        min_samples: Minimum samples required by EWM to yield non-null values.

    Returns:
        A Polars expression with standardized and clipped log returns.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"p": [1.0, 1.1, 1.05, 1.15, 1.2]})
        >>> result = df.select(vol_adj(pl.col("p"), vola=2, clip=3.0))
        >>> result.shape
        (5, 1)
    """
    # compute the log returns
    log_returns = x.log().diff()

    # compute the volatility of the log returns
    vol = log_returns.ewm_std(com=vola - 1, adjust=True, min_samples=min_samples)

    # compute the volatility-adjusted returns
    vol_adj_returns = (log_returns / vol).clip(-clip, clip)

    return vol_adj_returns
