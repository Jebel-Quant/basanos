"""Core data-access mixin for `BasanosEngine`.

Provides the volatility-adjusted returns, EWMA volatility, and per-timestamp
correlation properties as a reusable mixin so that ``optimizer.py`` stays
focused on the position-solving facade.

Classes in this module are **private implementation details**.  The public API
is `BasanosEngine`, which inherits from `_CoreDataMixin`.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from cvx.linalg import cov_to_corr
from cvx.linalg.covariance.ewm_cov import ewm_covariance

from ._signal import vol_adj

if TYPE_CHECKING:
    from ._engine_protocol import _EngineProtocol


class _CoreDataMixin:
    """Mixin providing the core data-access properties of `BasanosEngine`.

    The consuming class must satisfy `_EngineProtocol`, i.e. it must expose
    ``prices`` (Polars DataFrame with a ``'date'`` column) and ``cfg``
    (a `BasanosConfig`).
    """

    @property
    def assets(self: _EngineProtocol) -> list[str]:
        """List asset column names (numeric columns excluding 'date')."""
        return [c for c in self.prices.columns if c != "date" and self.prices[c].dtype.is_numeric()]

    @property
    def ret_adj(self: _EngineProtocol) -> pl.DataFrame:
        """Return per-asset volatility-adjusted log returns clipped by cfg.clip.

        Uses an EWMA volatility estimate with lookback ``cfg.vola`` to
        standardize log returns for each numeric asset column.
        """
        return self.prices.with_columns(
            [vol_adj(pl.col(asset), vola=self.cfg.vola, clip=self.cfg.clip) for asset in self.assets]
        )

    @property
    def vola(self: _EngineProtocol) -> pl.DataFrame:
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
    def cor(self: _EngineProtocol) -> dict[datetime.date, np.ndarray]:
        """Compute per-timestamp EWM correlation matrices.

        Builds volatility-adjusted returns for all assets, computes an
        exponentially weighted correlation using a pure NumPy implementation
        (with window ``cfg.corr``), and returns a mapping from each timestamp
        to the corresponding correlation matrix as a NumPy array.

        Returns:
            dict: Mapping ``date -> np.ndarray`` of shape (n_assets, n_assets).

        Performance:
            Delegates to ``ewm_covariance`` from ``cvx.linalg``.
            For large *N* or *T*, prefer ``cor_tensor`` to keep a single
            contiguous array rather than building a Python dict.
        """
        assets = list(self.assets)
        n = len(assets)
        span = 2 * self.cfg.corr + 1
        cov_dict = ewm_covariance(
            self.ret_adj,
            assets=assets,
            index_col="date",
            window=span,
            warmup=self.cfg.corr,
        )
        nan_mat = np.full((n, n), np.nan)
        return {
            date: cov_to_corr(cov_dict[date], self.cfg.min_corr_denom) if date in cov_dict else nan_mat.copy()
            for date in self.prices["date"].to_list()
        }

    @property
    def cor_tensor(self: _EngineProtocol) -> np.ndarray:
        """Return all correlation matrices stacked as a 3-D tensor.

        Converts the per-timestamp correlation dict (see `cor`) into a
        single contiguous NumPy array so that the full history can be saved to
        a flat ``.npy`` file with `save` and reloaded with
        `load`.

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
