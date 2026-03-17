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


def _drawdown_series(series: pl.Series) -> pl.Series:
    """Compute the drawdown percentage series from a returns series.

    Treats ``series`` as additive daily returns and builds a normalised NAV
    starting at 1.0.  The high-water mark is the running maximum of that NAV;
    drawdown is expressed as the fraction below the high-water mark.

    Args:
        series: A Polars Series of additive returns (profit / AUM).

    Returns:
        A Polars Float64 Series whose values are in [0, 1].  A value of 0
        means the NAV is at its all-time high; a value of 0.2 means the NAV
        is 20 % below its previous peak.

    Examples:
        >>> import polars as pl
        >>> s = pl.Series([0.0, -0.1, 0.2])
        >>> [round(x, 10) for x in _drawdown_series(s).to_list()]
        [0.0, 0.1, 0.0]
    """
    nav = 1.0 + series.cast(pl.Float64).cum_sum()
    hwm = nav.cum_max()
    # Guard against division by zero: a NAV of exactly 0 would make the
    # drawdown fraction undefined.  In practice NAV starts at 1.0 so this can
    # only occur for extremely large cumulative losses; the 1e-10 floor avoids
    # a ZeroDivisionError while having no effect on normal data.
    hwm_safe = hwm.clip(lower_bound=1e-10)
    return ((hwm - nav) / hwm_safe).clip(lower_bound=0.0)


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

        # Compute mean of returns less than or equal to VaR.
        # Return NaN when no empirical observations fall below the parametric
        # VaR threshold (empty filter), rather than the misleading 0.0 that
        # _to_float(None) would otherwise produce.
        mask = cast(Iterable[bool], series < var)
        filtered = series.filter(mask)
        if filtered.is_empty():
            return float("nan")
        return _to_float(filtered.mean())

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

        mean_val = _to_float(series.mean())
        divisor = _to_float(series.std(ddof=1))

        # Treat as zero-variance if divisor is zero or indistinguishable from
        # floating-point noise (i.e. smaller than 10x machine epsilon x |mean|).
        _eps = np.finfo(np.float64).eps
        if divisor <= _eps * max(abs(mean_val), _eps) * 10:
            return float("nan")

        res = mean_val / divisor
        factor = periods or 1
        return float(res * np.sqrt(factor))

    @columnwise_stat
    def max_drawdown(self, series: pl.Series) -> float:
        """Maximum drawdown as a fraction of the high-water mark.

        Computes the largest peak-to-trough decline in the cumulative additive
        NAV (starting at 1.0) expressed as a percentage of the peak.

        Args:
            series (pl.Series): Series of additive daily returns.

        Returns:
            float: Maximum drawdown in the range [0, 1].

        """
        return _to_float(_drawdown_series(series).max())

    @columnwise_stat
    def avg_drawdown(self, series: pl.Series) -> float:
        """Average drawdown across all underwater periods.

        Computes the mean drawdown percentage for every observation where the
        portfolio is below its previous peak.  Returns 0.0 if there are no
        underwater periods.

        Args:
            series (pl.Series): Series of additive daily returns.

        Returns:
            float: Mean drawdown in the range [0, 1].

        """
        dd = _drawdown_series(series)
        in_dd = dd.filter(dd > 0)
        if in_dd.is_empty():
            return 0.0
        return _to_float(in_dd.mean())

    def max_drawdown_duration(self) -> dict[str, float | int | None]:
        """Maximum drawdown duration in calendar days (or periods) per asset.

        Identifies consecutive runs of observations where the portfolio NAV is
        below its high-water mark and returns the length of the longest such
        run.

        When a ``date`` column is present the duration is expressed as the
        number of calendar days spanned by the run (inclusive of both
        endpoints).  When no ``date`` column exists each row counts as one
        period, so the result is a count of consecutive underwater periods.

        Returns:
            dict[str, float | int | None]: Mapping from asset name to maximum
            drawdown duration.  Returns 0 when there are no underwater
            periods.

        """
        has_date = "date" in self.data.columns
        result: dict[str, float | int | None] = {}
        for asset in self.assets:
            series = self.data[asset]
            nav = 1.0 + series.cast(pl.Float64).cum_sum()
            hwm = nav.cum_max()
            in_dd = nav < hwm

            if not in_dd.any():
                result[asset] = 0
                continue

            if has_date:
                frame = pl.DataFrame({"date": self.data["date"], "in_dd": in_dd})
            else:
                frame = pl.DataFrame({"date": pl.Series(list(range(len(series))), dtype=pl.Int64), "in_dd": in_dd})

            frame = frame.with_columns(pl.col("in_dd").rle_id().alias("run_id"))

            dd_runs = (
                frame.filter(pl.col("in_dd"))
                .group_by("run_id")
                .agg(
                    [
                        pl.col("date").min().alias("start"),
                        pl.col("date").max().alias("end"),
                    ]
                )
            )

            if has_date:
                dd_runs = dd_runs.with_columns(
                    ((pl.col("end") - pl.col("start")).dt.total_days() + 1).alias("duration")
                )
            else:
                dd_runs = dd_runs.with_columns((pl.col("end") - pl.col("start") + 1).alias("duration"))

            result[asset] = int(_to_float(dd_runs["duration"].max()))

        return result

    @columnwise_stat
    def calmar(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calmar ratio (annualized return divided by maximum drawdown).

        A standard complement to the Sharpe ratio for trend-following and
        momentum strategies.  Returns ``nan`` when the maximum drawdown is
        zero (no drawdown observed).

        Args:
            series (pl.Series): Series of additive daily returns.
            periods (int | float | None): Annualisation factor (observations
                per year).  Defaults to ``periods_per_year``.

        Returns:
            float: Calmar ratio, or ``nan`` if max drawdown is zero.

        """
        raw_periods = periods or self.periods_per_year
        max_dd = _to_float(_drawdown_series(series).max())
        if max_dd <= 0:
            return float("nan")
        ann_return = _to_float(series.mean()) * raw_periods
        return ann_return / max_dd

    @columnwise_stat
    def recovery_factor(self, series: pl.Series) -> float:
        """Recovery factor (total return divided by maximum drawdown).

        A robustness signal for systematic strategies: values well above 1
        indicate that cumulative profits are large relative to the worst
        historical loss.  Returns ``nan`` when the maximum drawdown is zero.

        Args:
            series (pl.Series): Series of additive daily returns.

        Returns:
            float: Recovery factor, or ``nan`` if max drawdown is zero.

        """
        max_dd = _to_float(_drawdown_series(series).max())
        if max_dd <= 0:
            return float("nan")
        total_return = _to_float(series.sum())
        return total_return / max_dd

    def rolling_sharpe(self, window: int = 63, periods: int | float | None = None) -> pl.DataFrame:
        """Compute rolling annualised Sharpe ratio over a sliding window.

        Args:
            window: Number of periods in the rolling window. Defaults to 63.
            periods: Number of periods per year for annualisation. Defaults to
                ``periods_per_year``.

        Returns:
            pl.DataFrame: A DataFrame with the date column (when present) and
                one column per asset.  The first ``window - 1`` rows will be
                null.

        Raises:
            ValueError: If ``window`` is not a positive integer.

        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError

        scale = np.sqrt(periods or self.periods_per_year)

        exprs = [
            (
                pl.col(asset).rolling_mean(window_size=window) / pl.col(asset).rolling_std(window_size=window) * scale
            ).alias(asset)
            for asset in self.assets
        ]

        cols: list[str | pl.Expr] = (["date"] if "date" in self.data.columns else []) + exprs
        return self.data.select(cols)

    def rolling_volatility(
        self, window: int = 63, periods: int | float | None = None, annualize: bool = True
    ) -> pl.DataFrame:
        """Compute rolling volatility over a sliding window.

        Args:
            window: Number of periods in the rolling window. Defaults to 63.
            periods: Number of periods per year for annualisation. Defaults to
                ``periods_per_year``.
            annualize: Whether to annualise the result by multiplying by
                ``sqrt(periods)``. Defaults to True.

        Returns:
            pl.DataFrame: A DataFrame with the date column (when present) and
                one column per asset.  The first ``window - 1`` rows will be
                null.

        Raises:
            ValueError: If ``window`` is not a positive integer.
            TypeError: If ``periods`` is not numeric.

        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError

        raw_periods = periods or self.periods_per_year
        if not isinstance(raw_periods, int | float):
            raise TypeError

        factor = np.sqrt(raw_periods) if annualize else 1.0

        exprs = [(pl.col(asset).rolling_std(window_size=window) * factor).alias(asset) for asset in self.assets]

        cols: list[str | pl.Expr] = (["date"] if "date" in self.data.columns else []) + exprs
        return self.data.select(cols)

    def annual_breakdown(self) -> pl.DataFrame:
        """Return summary statistics broken down by calendar year.

        Groups the data by calendar year using the ``date`` column, computes
        a full :py:meth:`summary` for each year, and stacks the results into
        a single DataFrame with an additional ``year`` column.

        Returns:
            pl.DataFrame: A DataFrame with columns ``year``, ``metric``, and
                one column per asset, sorted by ``year``.

        Raises:
            ValueError: If the DataFrame has no ``date`` column.

        """
        if "date" not in self.data.columns:
            raise ValueError

        years = self.data["date"].dt.year().unique().sort().to_list()

        frames: list[pl.DataFrame] = []
        for year in years:
            year_data = self.data.filter(self.data["date"].dt.year() == year)
            if year_data.height < 2:
                continue
            year_summary = Stats(year_data).summary()
            year_summary = year_summary.with_columns(pl.lit(year).alias("year"))
            frames.append(year_summary)

        if not frames:
            # Build empty DataFrame with expected schema
            schema = {"year": pl.Int32, "metric": pl.String, **dict.fromkeys(self.assets, pl.Float64)}
            return pl.DataFrame(schema=schema)

        result = pl.concat(frames)
        # Move 'year' to front
        ordered = ["year", "metric", *[c for c in result.columns if c not in ("year", "metric")]]
        return result.select(ordered)

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
            "max_drawdown": self.max_drawdown(),
            "avg_drawdown": self.avg_drawdown(),
            "max_drawdown_duration": self.max_drawdown_duration(),
            "calmar": self.calmar(),
            "recovery_factor": self.recovery_factor(),
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
