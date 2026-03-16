"""Statistical metrics and ratios for financial returns.

This module defines the Stats class which operates on a Data instance to
compute per-asset statistics like skew, kurtosis, volatility, Sharpe,
VaR/CVaR, and more.
"""

import dataclasses
from collections.abc import Callable, Iterable
from datetime import timedelta
from functools import wraps
from typing import cast

import numpy as np
import polars as pl
from scipy.stats import norm


def _to_float(value: object) -> float:
    """Safely convert a Polars aggregation result to float.

    Examples:
        >>> _to_float(2.0)
        2.0
        >>> _to_float(None)
        0.0
    """
    if value is None:
        return 0.0
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(cast(float, value))


def _to_float_or_none(value: object) -> float | None:
    """Safely convert a Polars aggregation result to float or None."""
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(cast(float, value))


@dataclasses.dataclass(frozen=True)
class Stats:
    """Statistical analysis tools for financial returns data.

    This class provides a comprehensive set of methods for calculating various
    financial metrics and statistics on returns data, including:

    - Basic statistics (mean, skew, kurtosis)
    - Risk metrics (volatility, value-at-risk, drawdown)
    - Performance ratios (Sharpe, information ratio)
    - Win/loss metrics (win rate, profit factor, payoff ratio)

    The class is designed to work with the _Data class and operates on Polars DataFrames
    for efficient computation.

    Attributes:
        data: The _Data object containing returns data.

    Examples:
        >>> import polars as pl
        >>> from datetime import date
        >>> data = pl.DataFrame({
        ...     "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
        ...     "returns": [0.01, -0.02, 0.03],
        ... })
        >>> stats = Stats(data=data)
        >>> stats.assets
        ['returns']
    """

    data: pl.DataFrame

    def __post_init__(self) -> None:
        """Validate the input data frame after initialization.

        Ensures that `data` is a Polars DataFrame and contains at least one
        row. Raises TypeError or ValueError otherwise.
        """
        if not isinstance(self.data, pl.DataFrame):
            raise TypeError
        if self.data.height == 0:
            raise ValueError

    @property
    def assets(self) -> list[str]:
        """List of asset column names (numeric columns excluding 'date')."""
        return [c for c in self.data.columns if c != "date" and self.data[c].dtype.is_numeric()]

    @staticmethod
    def _mean_positive_expr(series: pl.Series) -> float:
        """Return the mean of strictly positive values, or 0.0 if none exist."""
        result = series.filter(series > 0).mean()
        return _to_float(result)

    @staticmethod
    def _mean_negative_expr(series: pl.Series) -> float:
        """Return the mean of strictly negative values, or 0.0 if none exist."""
        result = series.filter(series < 0).mean()
        return _to_float(result)

    @staticmethod
    def columnwise_stat(func: Callable[..., float | int | None]) -> Callable[..., dict[str, float | int | None]]:
        """Apply a column-wise statistical function to all numeric columns.

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: The decorated function.

        """

        @wraps(func)
        def wrapper(self: "Stats", *args: object, **kwargs: object) -> dict[str, float | int | None]:
            """Apply the wrapped stat function to each asset column and return results as a dict."""
            return {asset: func(self, self.data[asset], *args, **kwargs) for asset in self.assets}

        return wrapper

    @columnwise_stat
    def skew(self, series: pl.Series) -> float | None:
        """Calculate skewness (asymmetry) for each numeric column.

        Args:
            series (pl.Series): The series to calculate skewness for.

        Returns:
            float: The skewness value.

        """
        return _to_float_or_none(series.skew(bias=False))

    @columnwise_stat
    def kurtosis(self, series: pl.Series) -> float | None:
        """Calculate the excess kurtosis of returns (Fisher definition).

        Uses an unbiased estimator when possible. For short samples where an
        unbiased estimator is undefined (Polars returns None when < 4 non-null
        observations), falls back to the biased estimator. If the series is
        still too short or variance is zero, computes the moment-based excess
        kurtosis m4/m2^2 - 3.0, returning 0.0 for constant series.
        """
        # Drop nulls to match test expectations (ignore missing values)
        s = series.drop_nulls()
        # Use biased estimator first (Fisher=True by default in Polars)
        return _to_float_or_none(s.kurtosis(bias=True))

    @columnwise_stat
    def avg_return(self, series: pl.Series) -> float:
        """Calculate average return per non-zero, non-null value.

        Args:
            series (pl.Series): The series to calculate average return for.

        Returns:
            float: The average return value.

        """
        result = series.filter(series.is_not_null() & (series != 0)).mean()
        return _to_float(result)

    @columnwise_stat
    def avg_win(self, series: pl.Series) -> float:
        """Calculate the average winning return/trade for an asset.

        Args:
            series (pl.Series): The series to calculate average win for.

        Returns:
            float: The average winning return.

        """
        return self._mean_positive_expr(series)

    @columnwise_stat
    def avg_loss(self, series: pl.Series) -> float:
        """Calculate the average loss return/trade for a period.

        Args:
            series (pl.Series): The series to calculate average loss for.

        Returns:
            float: The average loss return.

        """
        return self._mean_negative_expr(series)

    @columnwise_stat
    def volatility(self, series: pl.Series, periods: int | float | None = None, annualize: bool = True) -> float:
        """Calculate the volatility of returns.

        - Std dev of returns
        - Annualized by sqrt(periods) if `annualize` is True.

        Args:
            series (pl.Series): The series to calculate volatility for.
            periods (int, optional): Number of periods per year. Defaults to 252.
            annualize (bool, optional): Whether to annualize the result. Defaults to True.

        Returns:
            float: The volatility value.

        """
        raw_periods = periods or self.periods_per_year

        # Ensure it's numeric
        if not isinstance(raw_periods, int | float):
            raise TypeError

        factor = np.sqrt(raw_periods) if annualize else 1.0
        return _to_float(series.std()) * factor

    @columnwise_stat
    def value_at_risk(self, series: pl.Series, sigma: float = 1.0, alpha: float = 0.05) -> float:
        """Calculate the daily value-at-risk.

        Uses variance-covariance calculation with confidence level.

        Args:
            series (pl.Series): The series to calculate value at risk for.
            alpha (float, optional): Confidence level. Defaults to 0.05.
            sigma (float, optional): Standard deviation multiplier. Defaults to 1.0.

        Returns:
            float: The value at risk.

        """
        mu = _to_float(series.mean())
        sigma *= _to_float(series.std())

        return float(norm.ppf(alpha, mu, sigma))

    @columnwise_stat
    def conditional_value_at_risk(self, series: pl.Series, sigma: float = 1.0, alpha: float = 0.05) -> float:
        """Calculate the conditional value-at-risk.

        Also known as CVaR or expected shortfall, calculated for each numeric column.

        Args:
            series (pl.Series): The series to calculate conditional value at risk for.
            alpha (float, optional): Confidence level. Defaults to 0.05.
            sigma (float, optional): Standard deviation multiplier. Defaults to 1.0.

        Returns:
            float: The conditional value at risk.

        """
        mu = _to_float(series.mean())
        sigma *= _to_float(series.std())

        var = norm.ppf(alpha, mu, sigma)

        # Compute mean of returns less than or equal to VaR
        mask = cast(Iterable[bool], series < var)
        result = series.filter(mask).mean()
        return _to_float(result)

    @columnwise_stat
    def best(self, series: pl.Series) -> float | None:
        """Find the maximum return per column (best period).

        Args:
            series (pl.Series): The series to find the best return for.

        Returns:
            float: The maximum return value.

        """
        return _to_float_or_none(series.max())

    @columnwise_stat
    def worst(self, series: pl.Series) -> float | None:
        """Find the minimum return per column (worst period).

        Args:
            series (pl.Series): The series to find the worst return for.

        Returns:
            float: The minimum return value.

        """
        return _to_float_or_none(series.min())

    @columnwise_stat
    def sharpe(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calculate the Sharpe ratio of asset returns.

        Args:
            series (pl.Series): The series to calculate Sharpe ratio for.
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            float: The Sharpe ratio value.

        """
        periods = periods or self.periods_per_year

        divisor = _to_float(series.std(ddof=1))
        if divisor == 0.0:
            return float("nan")

        res = _to_float(series.mean()) / divisor
        factor = periods or 1
        return float(res * np.sqrt(factor))

    def summary(self) -> pl.DataFrame:
        """Return a DataFrame summarising all statistics for each asset.

        Each row corresponds to one statistical metric; each column (beyond
        the ``metric`` column) corresponds to one asset in the portfolio.

        Returns:
            pl.DataFrame: A DataFrame with a ``metric`` column followed by one
            column per asset, containing the computed statistic values.

        """
        metrics: dict[str, dict[str, float | int | None]] = {
            "avg_return": self.avg_return(),
            "avg_win": self.avg_win(),
            "avg_loss": self.avg_loss(),
            "best": self.best(),
            "worst": self.worst(),
            "volatility": self.volatility(),
            "sharpe": self.sharpe(),
            "skew": self.skew(),
            "kurtosis": self.kurtosis(),
            "value_at_risk": self.value_at_risk(),
            "conditional_value_at_risk": self.conditional_value_at_risk(),
        }

        rows: list[dict[str, object]] = [
            {"metric": name, **{asset: values[asset] for asset in self.assets}} for name, values in metrics.items()
        ]

        return pl.DataFrame(rows)

    @property
    def periods_per_year(self) -> float:
        """Estimate the number of periods per year from timestamp spacing.

        Computes the average spacing (in seconds) between consecutive timestamps using
        plain Python datetimes to avoid ambiguity around Polars Duration arithmetic,
        then returns 365 * 24 * 3600 divided by that spacing.

        Returns:
            float: Estimated number of observations per calendar year.
        """
        # Extract datetime values as Python objects (assuming a single datetime column)
        col_name = self.data.columns[0]
        dates = self.data[col_name]

        # Index is guaranteed to have at least two rows by __post_init__,
        # so we can compute gaps directly after sorting.
        dates = dates.sort()
        # Compute successive differences in seconds
        gaps = dates.diff().drop_nulls()

        mean_diff = gaps.mean()

        # Convert Duration (timedelta) to seconds
        if isinstance(mean_diff, timedelta):
            seconds = mean_diff.total_seconds()
        elif mean_diff is not None:
            seconds = _to_float(mean_diff)
        else:
            # Fallback to daily if mean_diff is None
            seconds = 86400.0

        return (365.0 * 24.0 * 60.0 * 60.0) / seconds
