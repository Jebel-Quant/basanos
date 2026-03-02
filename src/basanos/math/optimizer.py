"""Correlation-aware risk position optimizer (Basanos).

This module provides utilities to compute correlation-adjusted risk positions
from price data and expected-return signals. It relies on volatility-adjusted
returns to estimate a dynamic correlation matrix (via pandas EWM), applies
shrinkage towards identity, and solves a normalized linear system per
timestamp to obtain stable positions.
"""

import dataclasses

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from ..analytics import Portfolio
from ._linalg import inv_a_norm, solve
from ._signal import shrink2id, vol_adj


class BasanosConfig(BaseModel):
    """Configuration for correlation-aware position optimization."""

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
        """Compute per-timestamp EWMA correlation matrices.

        Builds volatility-adjusted returns for all assets, computes an
        exponentially weighted correlation using pandas (with window
        ``cfg.corr``), and returns a mapping from each timestamp to the
        corresponding correlation matrix as a NumPy array.

        Returns:
            dict: Mapping ``date -> np.ndarray`` of shape (n_assets, n_assets).
        """
        # All numeric columns except the date column are treated as assets.
        index = self.prices["date"]

        # Compute EWM correlation via pandas in steps to keep lines short for linters
        ret_adj_pd = self.ret_adj.select(self.assets).to_pandas()
        ewm_corr = ret_adj_pd.ewm(com=self.cfg.corr, min_periods=self.cfg.corr).corr()
        cor = ewm_corr.reset_index(names=["t", "asset"])  # (t, asset) index -> long-format DataFrame

        return {index[t]: df_t.drop(columns=["t", "asset"]).to_numpy() for t, df_t in cor.groupby("t")}

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
