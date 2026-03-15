"""Internal signal utilities (private to basanos.math).

This module contains low-level helpers for building signals and
transformations. It is considered an internal implementation detail of
``basanos.math``. Do not import this module directly from outside the
package; instead import the public symbols from ``basanos.math``.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def shrink2id(matrix: np.ndarray, lamb: float = 1.0) -> np.ndarray:
    """Shrink a square matrix towards identity by weight ``lamb``.

    Args:
        matrix: Square matrix to shrink.
        lamb: Mixing ratio in [0, 1]. ``1.0`` keeps the original matrix;
            ``0.0`` returns the identity.

    Returns:
        The shrunk matrix with the same shape as ``matrix``.

    Examples:
        >>> import numpy as np
        >>> shrink2id(np.array([[2.0, 0.0], [0.0, 2.0]]), lamb=0.0).tolist()
        [[1.0, 0.0], [0.0, 1.0]]
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
