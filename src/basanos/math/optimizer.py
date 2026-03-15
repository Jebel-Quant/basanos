"""Correlation-aware risk position optimizer (Basanos).

This module provides utilities to compute correlation-adjusted risk positions
from price data and expected-return signals. It relies on volatility-adjusted
returns to estimate a dynamic correlation matrix (via EWM), applies shrinkage
towards identity, and solves a normalized linear system per timestamp to
obtain stable positions.
"""

import dataclasses

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from ..analytics import Portfolio
from ._linalg import inv_a_norm, solve
from ._signal import shrink2id, vol_adj

_MIN_CORR_DENOM: float = 1e-14  # guard against near-zero variance in correlation computation


def _ewm_corr_numpy(data: np.ndarray, com: int, min_periods: int) -> np.ndarray:
    """Compute per-row EWM correlation matrices without pandas.

    Matches ``pandas.DataFrame.ewm(com=com, min_periods=min_periods).corr()``
    with the default ``adjust=True, ignore_na=False`` settings to within
    floating-point rounding error.

    All five EWM components used to compute ``corr(i, j)`` — namely
    ``ewm(x_i)``, ``ewm(x_j)``, ``ewm(x_i²)``, ``ewm(x_j²)``, and
    ``ewm(x_i·x_j)`` — share the **same joint weight structure**: weights
    decay at every timestep (``ignore_na=False``) but a new observation is
    only added at timesteps where *both* ``x_i`` and ``x_j`` are finite.  As
    a result the correlation for a pair is frozen once either asset goes
    missing, exactly mirroring pandas behaviour.

    Args:
        data: Float array of shape ``(T, N)`` - typically volatility-adjusted
            log returns.
        com: EWM centre-of-mass (``alpha = 1 / (1 + com)``).
        min_periods: Minimum number of joint finite observations required
            before a correlation value is reported; earlier rows are NaN.

    Returns:
        np.ndarray of shape ``(T, N, N)`` containing the per-row correlation
        matrices.  Each matrix is symmetric with diagonal 1.0 (or NaN during
        warm-up).
    """
    t_len, n_assets = data.shape
    beta = com / (1.0 + com)

    result = np.full((t_len, n_assets, n_assets), np.nan)

    # Per-pair running sums.  All decay at every timestep; only update when
    # BOTH assets are finite (joint observations).
    #
    # sx[i, j]  = weighted sum of x_i for pair (i, j)
    # sx2[i, j] = weighted sum of x_i² for pair (i, j)
    # sxy[i, j] = weighted sum of x_i·x_j for pair (i, j)
    # w_sum[i, j]   = sum of weights for pair (i, j)
    #
    # By symmetry: sx.T[i, j] = sx[j, i] = weighted sum of x_j for pair (i, j)
    #              sx2.T[i, j]             = weighted sum of x_j² for pair (i, j)
    sx = np.zeros((n_assets, n_assets))
    sx2 = np.zeros((n_assets, n_assets))
    sxy = np.zeros((n_assets, n_assets))
    w_sum = np.zeros((n_assets, n_assets))
    count = np.zeros((n_assets, n_assets), dtype=np.intp)

    for t in range(t_len):
        xt = data[t]
        fin = np.isfinite(xt)

        # Decay all sums at every timestep (ignore_na=False)
        sx *= beta
        sx2 *= beta
        sxy *= beta
        w_sum *= beta

        # Add jointly-finite observations
        xt_finite = np.where(fin, xt, 0.0)
        pair_fin = np.outer(fin, fin)  # (N, N) bool

        # xi_row[i, j] = xt[i] for all j (row-wise broadcast)
        xi_row = np.outer(xt_finite, np.ones(n_assets))

        sx[pair_fin] += xi_row[pair_fin]
        sx2[pair_fin] += (xi_row * xi_row)[pair_fin]
        sxy[pair_fin] += (xi_row * xi_row.T)[pair_fin]  # xt[i] * xt[j]
        w_sum[pair_fin] += 1.0
        count[pair_fin] += 1

        # Compute EWM means via running-sum / weight-sum ratios.
        # sx.T / w_sum gives the per-pair EWM of the *column* asset x_j.
        with np.errstate(divide="ignore", invalid="ignore"):
            ewm_x = np.where(w_sum > 0, sx / w_sum, np.nan)  # EWM(x_i) for pair (i,j)
            ewm_y = np.where(w_sum > 0, sx.T / w_sum, np.nan)  # EWM(x_j) for pair (i,j)
            ewm_x2 = np.where(w_sum > 0, sx2 / w_sum, np.nan)  # EWM(x_i²) for pair (i,j)
            ewm_y2 = np.where(w_sum > 0, sx2.T / w_sum, np.nan)  # EWM(x_j²) for pair (i,j)
            ewm_xy = np.where(w_sum > 0, sxy / w_sum, np.nan)  # EWM(x_i·x_j) for pair (i,j)

        var_x = np.maximum(ewm_x2 - ewm_x * ewm_x, 0.0)
        var_y = np.maximum(ewm_y2 - ewm_y * ewm_y, 0.0)
        denom = np.sqrt(var_x * var_y)
        cov = ewm_xy - ewm_x * ewm_y

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > _MIN_CORR_DENOM, cov / denom, np.nan)

        corr = np.clip(corr, -1.0, 1.0)

        # Diagonal is exactly 1.0 when the asset has enough observations
        for i in range(n_assets):
            corr[i, i] = 1.0 if count[i, i] >= min_periods else np.nan

        # Apply min_periods mask for all pairs
        corr[count < min_periods] = np.nan

        result[t] = corr

    return result


class BasanosConfig(BaseModel):
    """Configuration for correlation-aware position optimization.

    Examples:
        >>> cfg = BasanosConfig(vola=32, corr=64, clip=3.0, shrink=0.5, aum=1e8)
        >>> cfg.vola
        32
        >>> cfg.corr
        64
    """

    vola: int = Field(..., gt=0, description="EWMA lookback for volatility normalization.")
    corr: int = Field(..., gt=0, description="EWMA lookback for correlation estimation.")
    clip: float = Field(..., gt=0.0, description="Clipping threshold for volatility adjustment.")
    shrink: float = Field(
        ..., ge=0.0, le=1.0, description="Shrinkage intensity towards identity (0=no shrinkage, 1=identity)."
    )
    aum: float = Field(..., gt=0.0, description="Assets under management for portfolio scaling.")

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("corr")
    @classmethod
    def corr_greater_than_vola(cls, v: int, info: ValidationInfo) -> int:
        """Optionally enforce corr ≥ vola for stability.

        Pydantic v2 passes ValidationInfo; use info.data to access other fields.
        """
        vola = info.data.get("vola") if hasattr(info, "data") else None
        if vola is not None and v < vola:
            raise ValueError
        return v


@dataclasses.dataclass(frozen=True)
class BasanosEngine:
    """Engine to compute correlation matrices and optimize risk positions.

    Encapsulates price data and configuration to build EWM-based
    correlations, apply shrinkage, and solve for normalized positions.
    """

    prices: pl.DataFrame
    mu: pl.DataFrame
    cfg: BasanosConfig

    def __post_init__(self) -> None:
        """Validate basic invariants right after initialization.

        Ensures both ``prices`` and ``mu`` contain a ``'date'`` column and
        share identical shapes/columns, which downstream computations rely on.
        """
        # ensure 'date' column exists in prices before any other validation
        if "date" not in self.prices.columns:
            raise ValueError

        # ensure 'date' column exists in mu as well (kept for symmetry and downstream assumptions)
        if "date" not in self.mu.columns:
            raise ValueError

        # check that prices and mu have the same shape
        if self.prices.shape != self.mu.shape:
            raise ValueError

        # check that the columns of prices and mu are identical
        if not set(self.prices.columns) == set(self.mu.columns):
            raise ValueError

    @property
    def assets(self) -> list[str]:
        """List asset column names (numeric columns excluding 'date')."""
        return [c for c in self.prices.columns if c != "date" and self.prices[c].dtype.is_numeric()]

    @property
    def ret_adj(self) -> pl.DataFrame:
        """Return per-asset volatility-adjusted log returns clipped by cfg.clip.

        Uses an EWMA volatility estimate with lookback ``cfg.vola`` to
        standardize log returns for each numeric asset column.
        """
        return self.prices.with_columns(
            [vol_adj(pl.col(asset), vola=self.cfg.vola, clip=self.cfg.clip) for asset in self.assets]
        )

    @property
    def vola(self) -> pl.DataFrame:
        """Per-asset EWMA volatility of percentage returns.

        Computes percent changes for each numeric asset column and applies an
        exponentially weighted standard deviation using the lookback specified
        by ``cfg.vola``. The result is a DataFrame aligned with ``self.prices``
        whose numeric columns hold per-asset volatility estimates.
        """
        return self.prices.with_columns(
            pl.col(asset)
            .pct_change()
            .ewm_std(com=self.cfg.vola - 1, adjust=True, min_samples=self.cfg.vola)
            .alias(asset)
            for asset in self.assets
        )

    @property
    def cor(self) -> dict[object, np.ndarray]:
        """Compute per-timestamp EWM correlation matrices.

        Builds volatility-adjusted returns for all assets, computes an
        exponentially weighted correlation using a pure NumPy implementation
        (with window ``cfg.corr``), and returns a mapping from each timestamp
        to the corresponding correlation matrix as a NumPy array.

        Returns:
            dict: Mapping ``date -> np.ndarray`` of shape (n_assets, n_assets).
        """
        index = self.prices["date"]
        ret_adj_np = self.ret_adj.select(self.assets).to_numpy()
        tensor = _ewm_corr_numpy(ret_adj_np, com=self.cfg.corr, min_periods=self.cfg.corr)
        return {index[t]: tensor[t] for t in range(len(index))}

    @property
    def cor_tensor(self) -> np.ndarray:
        """Return all correlation matrices stacked as a 3-D tensor.

        Converts the per-timestamp correlation dict (see :py:attr:`cor`) into a
        single contiguous NumPy array so that the full history can be saved to
        a flat ``.npy`` file with :func:`numpy.save` and reloaded with
        :func:`numpy.load`.

        Returns:
            np.ndarray: Array of shape ``(T, N, N)`` where *T* is the number of
            timestamps and *N* the number of assets.  ``tensor[t]`` is the
            correlation matrix for the *t*-th date (same ordering as
            ``self.prices["date"]``).

        Examples:
            >>> import tempfile, pathlib
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math.optimizer import BasanosConfig, BasanosEngine
            >>> dates = pl.Series("date", list(range(100)))
            >>> rng0 = np.random.default_rng(0).lognormal(size=100)
            >>> rng1 = np.random.default_rng(1).lognormal(size=100)
            >>> prices = pl.DataFrame({"date": dates, "A": rng0, "B": rng1})
            >>> rng2 = np.random.default_rng(2).normal(size=100)
            >>> rng3 = np.random.default_rng(3).normal(size=100)
            >>> mu = pl.DataFrame({"date": dates, "A": rng2, "B": rng3})
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> tensor = engine.cor_tensor
            >>> with tempfile.TemporaryDirectory() as td:
            ...     path = pathlib.Path(td) / "cor.npy"
            ...     np.save(path, tensor)
            ...     loaded = np.load(path)
            >>> np.testing.assert_array_equal(tensor, loaded)
        """
        return np.stack(list(self.cor.values()), axis=0)

    @property
    def cash_position(self) -> pl.DataFrame:
        """Optimize correlation-aware risk positions for each timestamp.

        Computes EWMA correlations (via ``self.cor``), applies shrinkage toward
        the identity matrix with intensity ``cfg.shrink``, and solves a
        normalized linear system A x = mu per timestamp to obtain stable,
        scale-invariant positions. Non-finite or ill-posed cases yield zero
        positions for safety.

        Returns:
            pl.DataFrame: DataFrame with columns ['date'] + asset names containing
            the per-timestamp cash positions (risk divided by EWMA volatility).
        """
        # compute the correlation matrices
        cor = self.cor
        assets = self.assets

        # Compute risk positions row-by-row using correlation shrinkage (NumPy)
        prices_num = self.prices.select(assets).to_numpy()
        returns_num = np.zeros_like(prices_num, dtype=float)
        returns_num[1:] = prices_num[1:] / prices_num[:-1] - 1.0

        mu = self.mu.select(assets).to_numpy()
        risk_pos_np = np.full_like(mu, fill_value=np.nan, dtype=float)
        cash_pos_np = np.full_like(mu, fill_value=np.nan, dtype=float)
        vola_np = self.vola.select(assets).to_numpy()

        profit_variance = 1.0
        lamb = 0.99

        for i, t in enumerate(cor.keys()):
            # get the mask of finite prices for this timestamp
            mask = np.isfinite(prices_num[i])

            # Compute profit contribution using only finite returns and available positions
            if i > 0:
                ret_mask = np.isfinite(returns_num[i]) & mask
                # Profit at time i comes from yesterday's cash position applied to today's returns
                if ret_mask.any():
                    cash_pos_np[i - 1] = risk_pos_np[i - 1] / vola_np[i - 1]
                    lhs = np.nan_to_num(cash_pos_np[i - 1, ret_mask], nan=0.0)
                    rhs = np.nan_to_num(returns_num[i, ret_mask], nan=0.0)
                    profit = lhs @ rhs
                    profit_variance = lamb * profit_variance + (1 - lamb) * profit**2
            # we have no price data at all for this timestamp
            if not mask.any():
                continue

            # get the correlation matrix for this timestamp
            corr_n = cor[t]

            # shrink the correlation matrix towards identity
            matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]

            # get the expected-return vector for this timestamp
            expected_mu = np.nan_to_num(mu[i][mask])

            # Normalize solution; guard against zero/near-zero norm to avoid NaNs
            denom = inv_a_norm(expected_mu, matrix)

            if denom is None or not np.isfinite(denom) or denom <= 1e-12 or np.allclose(expected_mu, 0.0):
                pos = np.zeros_like(expected_mu)
            else:
                pos = solve(matrix, expected_mu) / denom

            risk_pos_np[i, mask] = pos / profit_variance
            cash_pos_np[i, mask] = risk_pos_np[i, mask] / vola_np[i, mask]

        # Build Polars DataFrame for risk positions (numeric columns only)
        cash_position = self.prices.with_columns(
            [(pl.lit(cash_pos_np[:, i]).alias(asset)) for i, asset in enumerate(assets)]
        )

        return cash_position

    @property
    def portfolio(self) -> Portfolio:
        """Construct a Portfolio from the optimized cash positions.

        Converts the computed cash positions into a Portfolio using the
        configured AUM.

        Returns:
            Portfolio: Instance built from cash positions with AUM scaling.
        """
        return Portfolio.from_cash_position(self.prices, self.cash_position * 1e6, aum=self.cfg.aum)
