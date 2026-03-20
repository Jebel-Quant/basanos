"""Signal-evaluation mixin for BasanosEngine.

Provides information-coefficient (IC) metrics as a reusable mixin so that
``optimizer.py`` stays focused on the core position-solving logic.

Classes in this module are **private implementation details**.  The public API
is :class:`~basanos.math.optimizer.BasanosEngine`, which inherits from
:class:`_SignalEvaluatorMixin`.
"""

import numpy as np
import polars as pl
from scipy.stats import spearmanr


class _SignalEvaluatorMixin:
    """Mixin providing cross-sectional information-coefficient (IC) metrics.

    Expects the consuming class to expose:

    * ``assets`` — list of asset column names
    * ``prices`` — Polars DataFrame with a ``'date'`` column
    * ``mu`` — Polars DataFrame of expected-return signals
    """

    def _ic_series(self, use_rank: bool) -> pl.DataFrame:
        """Compute the cross-sectional IC time series.

        For each timestamp *t* (from 0 to T-2), correlates the signal vector
        ``mu[t, :]`` with the one-period forward return vector
        ``prices[t+1, :] / prices[t, :] - 1`` across all assets where both
        quantities are finite.  When fewer than two valid asset pairs are
        available, the IC value is set to ``NaN``.

        Args:
            use_rank: When ``True`` the Spearman rank correlation is used
                (Rank IC); when ``False`` the Pearson correlation is used (IC).

        Returns:
            pl.DataFrame: Two-column frame with ``date`` (signal date) and
            either ``ic`` or ``rank_ic``.
        """
        assets = self.assets
        prices_np = self.prices.select(assets).to_numpy().astype(float)
        mu_np = self.mu.select(assets).to_numpy().astype(float)
        dates = self.prices["date"].to_list()

        col_name = "rank_ic" if use_rank else "ic"
        ic_values: list[float] = []
        ic_dates = []

        for t in range(len(dates) - 1):
            fwd_ret = prices_np[t + 1] / prices_np[t] - 1.0
            signal = mu_np[t]

            # Both signal and forward return must be finite
            mask = np.isfinite(signal) & np.isfinite(fwd_ret)
            n_valid = int(mask.sum())

            if n_valid < 2:
                ic_values.append(float("nan"))
            elif use_rank:
                corr, _ = spearmanr(signal[mask], fwd_ret[mask])
                ic_values.append(float(corr))
            else:
                ic_values.append(float(np.corrcoef(signal[mask], fwd_ret[mask])[0, 1]))

            ic_dates.append(dates[t])

        return pl.DataFrame({"date": ic_dates, col_name: pl.Series(ic_values, dtype=pl.Float64)})

    @property
    def ic(self) -> pl.DataFrame:
        """Cross-sectional Pearson Information Coefficient (IC) time series.

        For each timestamp *t* (excluding the last), computes the Pearson
        correlation between the signal ``mu[t, :]`` and the one-period forward
        return ``prices[t+1, :] / prices[t, :] - 1`` across all assets where
        both quantities are finite.

        An IC value close to +1 means the signal ranked assets in the same
        order as forward returns; close to -1 means the opposite; near 0 means
        no predictive relationship.

        Returns:
            pl.DataFrame: Frame with columns ``['date', 'ic']``.  ``date`` is
            the timestamp at which the signal was observed.  ``ic`` is a
            ``Float64`` series (``NaN`` when fewer than 2 valid asset pairs
            are available for a given timestamp).

        See Also:
            :py:attr:`rank_ic` — Spearman variant, more robust to outliers.
            :py:attr:`ic_mean`, :py:attr:`ic_std`, :py:attr:`icir` — summary
            statistics.
        """
        return self._ic_series(use_rank=False)

    @property
    def rank_ic(self) -> pl.DataFrame:
        """Cross-sectional Spearman Rank Information Coefficient time series.

        Identical to :py:attr:`ic` but uses the Spearman rank correlation
        instead of the Pearson correlation, making it more robust to fat-tailed
        return distributions and outliers.

        Returns:
            pl.DataFrame: Frame with columns ``['date', 'rank_ic']``.
            ``rank_ic`` is a ``Float64`` series.

        See Also:
            :py:attr:`ic` — Pearson variant.
            :py:attr:`rank_ic_mean`, :py:attr:`rank_ic_std` — summary
            statistics.
        """
        return self._ic_series(use_rank=True)

    @property
    def ic_mean(self) -> float:
        """Mean of the IC time series, ignoring NaN values.

        Returns:
            float: Arithmetic mean of all finite IC values, or ``NaN`` if
            no finite values exist.
        """
        arr = self.ic["ic"].drop_nulls().to_numpy()
        finite = arr[np.isfinite(arr)]
        return float(np.mean(finite)) if len(finite) > 0 else float("nan")

    @property
    def ic_std(self) -> float:
        """Standard deviation of the IC time series, ignoring NaN values.

        Uses ``ddof=1`` (sample standard deviation).

        Returns:
            float: Sample standard deviation of all finite IC values, or
            ``NaN`` if fewer than 2 finite values exist.
        """
        arr = self.ic["ic"].drop_nulls().to_numpy()
        finite = arr[np.isfinite(arr)]
        return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")

    @property
    def icir(self) -> float:
        """Information Coefficient Information Ratio (ICIR).

        Defined as ``IC mean / IC std``.  A higher absolute ICIR indicates a
        more consistent signal: the mean IC is large relative to its
        variability.

        Returns:
            float: ``ic_mean / ic_std``, or ``NaN`` when ``ic_std`` is zero
            or non-finite.
        """
        mean = self.ic_mean
        std = self.ic_std
        if not np.isfinite(std) or std == 0.0:
            return float("nan")
        return float(mean / std)

    @property
    def rank_ic_mean(self) -> float:
        """Mean of the Rank IC time series, ignoring NaN values.

        Returns:
            float: Arithmetic mean of all finite Rank IC values, or ``NaN``
            if no finite values exist.
        """
        arr = self.rank_ic["rank_ic"].drop_nulls().to_numpy()
        finite = arr[np.isfinite(arr)]
        return float(np.mean(finite)) if len(finite) > 0 else float("nan")

    @property
    def rank_ic_std(self) -> float:
        """Standard deviation of the Rank IC time series, ignoring NaN values.

        Uses ``ddof=1`` (sample standard deviation).

        Returns:
            float: Sample standard deviation of all finite Rank IC values, or
            ``NaN`` if fewer than 2 finite values exist.
        """
        arr = self.rank_ic["rank_ic"].drop_nulls().to_numpy()
        finite = arr[np.isfinite(arr)]
        return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
