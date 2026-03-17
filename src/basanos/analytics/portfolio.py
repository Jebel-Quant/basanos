"""Portfolio utilities for computing profits, NAV, and Sharpe using Polars and Plotly.

This module provides a Portfolio dataclass and helpers to compute per-asset profits,
aggregate portfolio profit, NAV, Sharpe ratio, and produce Plotly visualizations.
"""

import dataclasses

import polars as pl
import polars.selectors as cs

from ..exceptions import IntegerIndexBoundError, MissingDateColumnError
from ._plots import Plots
from ._stats import Stats


@dataclasses.dataclass(frozen=True)
class Portfolio:
    """Store prices, positions, and compute portfolio statistics.

    Attributes:
        cashposition: Polars DataFrame of positions per asset over time (includes date column if present).
        prices: Polars DataFrame of prices per asset over time (includes date column if present).
        aum: Assets under management used as base NAV offset.


    Examples:
        >>> import polars as pl
        >>> from datetime import date
        >>> prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
        >>> pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
        >>> pf = Portfolio(prices=prices, cashposition=pos)
        >>> pf.assets
        ['A']
    """

    cashposition: pl.DataFrame
    prices: pl.DataFrame
    aum: float = 1e8

    def __post_init__(self) -> None:
        """Validate input types, shapes, and parameters post-initialization."""
        # Input validation
        if not isinstance(self.prices, pl.DataFrame):
            raise TypeError
        if not isinstance(self.cashposition, pl.DataFrame):
            raise TypeError

        if self.cashposition.shape[0] != self.prices.shape[0]:
            raise ValueError
        if self.aum <= 0.0:
            raise ValueError

    @classmethod
    def from_risk_position(
        cls, prices: pl.DataFrame, risk_position: pl.DataFrame, vola: int = 32, aum: float = 1e8
    ) -> "Portfolio":
        """Create a Portfolio from per-asset risk positions by de-volatizing with EWMA volatility.

        Args:
            prices: Price levels per asset over time (may include a date column).
            risk_position: Risk units per asset (e.g., target risk exposure) aligned with prices.
            vola: EWMA lookback (span-equivalent) used to estimate volatility in trading days.
            aum: Assets under management used as the base NAV offset.

        Returns:
            A Portfolio instance whose cash positions are risk_position divided by EWMA volatility.
        """
        assets = [col for col, dtype in prices.schema.items() if dtype.is_numeric()]

        def vol(col_name: str, vola: int) -> pl.Expr:  # pragma: no cover
            """Return an EWMA volatility expression for the given column and lookback."""
            return pl.col(col_name).pct_change().ewm_std(com=vola - 1, adjust=True, min_samples=vola)

        # Join prices to risk_position to compute volatility from price data
        cash_position = risk_position.with_columns(
            (pl.col(asset) / prices[asset].pct_change().ewm_std(com=vola - 1, adjust=True, min_samples=vola)).alias(
                asset
            )
            for asset in assets
        )
        return cls(prices=prices, cashposition=cash_position, aum=aum)

    @classmethod
    def from_cash_position(cls, prices: pl.DataFrame, cash_position: pl.DataFrame, aum: float = 1e8) -> "Portfolio":
        """Create a Portfolio directly from cash positions aligned with prices.

        Args:
            prices: Price levels per asset over time (may include a date column).
            cash_position: Cash exposure per asset over time (same shape/index as prices).
            aum: Assets under management used as the base NAV offset.

        Returns:
            A Portfolio instance with the provided cash positions.
        """
        return cls(prices=prices, cashposition=cash_position, aum=aum)

    @property
    def profits(self) -> pl.DataFrame:
        """Compute per-asset daily cash profits, preserving non-numeric columns.

        Returns:
            pl.DataFrame: Per-asset daily profit series along with any non-numeric
            columns (e.g., 'date').

        Examples:
            >>> import polars as pl
            >>> prices = pl.DataFrame({"A": [100.0, 110.0, 105.0]})
            >>> pos = pl.DataFrame({"A": [1000.0, 1000.0, 1000.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos)
            >>> pf.profits.columns
            ['A']
        """
        assets = [c for c in self.prices.columns if self.prices[c].dtype.is_numeric()]

        # Compute daily profits per numeric column
        profits = self.prices.with_columns(
            (self.prices[asset].pct_change().fill_null(0.0) * self.cashposition[asset].shift(n=1).fill_null(0.0)).alias(
                asset
            )
            for asset in assets
        )

        # Ensure there are no Nulls/NaNs/Infs in numeric profit columns
        # - Fill nulls with 0.0 (should already be handled above, but double-guard)
        # - Replace non-finite values (NaN/Inf) with 0.0
        if assets:
            profits = profits.with_columns(
                pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(0.0).fill_null(0.0).alias(c) for c in assets
            )
            # Guards to guarantee cleanliness
            for c in assets:
                s = profits[c]
                if int(s.null_count()) != 0:
                    raise ValueError  # pragma: no cover
                if not bool(pl.Series(s).is_finite().all()):
                    raise ValueError  # pragma: no cover

        return profits

    @staticmethod
    def _assert_clean_series(series: pl.Series, name: str = "") -> None:
        """Raise ValueError if the series contains nulls or non-finite values."""
        if series.null_count() != 0:
            raise ValueError
        if not series.is_finite().all():
            raise ValueError

    @property
    def profit(self) -> pl.DataFrame:
        """Return total daily portfolio profit including the 'date' column.

        Ensures that no day's total profit is NaN/null by asserting the
        'profit' column has zero nulls.
        """
        df_profits = self.profits
        assets = [c for c in df_profits.columns if df_profits[c].dtype.is_numeric()]

        if not assets:
            raise ValueError

        non_assets = [c for c in df_profits.columns if c not in set(assets)]
        # numeric_cols, non_numeric_cols = split_numeric_non_numeric(df_profits)

        # Row-wise sum of numeric columns
        portfolio_daily_profit = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in assets]).alias("profit")

        # Combine with non-numeric columns (like 'date')
        result = df_profits.select([*non_assets, portfolio_daily_profit])

        # Guard: profit must not contain NaN/null values
        # Use null_count to cover both nulls and NaNs (Polars treats NaNs as not-null but we ensure
        # inputs are numeric and filled; additional check for finite values guards against NaN/Inf)
        self._assert_clean_series(series=result["profit"])

        return result

    @property
    def nav_accumulated(self) -> pl.DataFrame:
        """Compute cumulative NAV of the portfolio including 'date'."""
        # Compute cumulative sum of profit column and expose as 'NAV'
        return self.profit.with_columns((pl.col("profit").cum_sum() + self.aum).alias("NAV_accumulated"))

    @property
    def returns(self) -> pl.DataFrame:
        """Return daily returns as profit scaled by AUM, preserving 'date'.

        The returned DataFrame contains the original 'date' column with the
        'profit' column scaled by AUM (i.e., per-period returns), and also an
        additional convenience column named 'returns' with the same values for
        downstream consumers.
        """
        return self.nav_accumulated.with_columns(
            (pl.col("profit") / self.aum).alias("returns"),
        )

    @property
    def monthly(self) -> pl.DataFrame:
        """Return monthly compounded returns and calendar columns.

        Aggregates daily returns (profit/AUM) by calendar month and computes
        the compounded monthly return: prod(1 + r_d) - 1. The resulting frame
        includes:
        - date: month-end label as a Polars Date (end of the grouping window)
        - returns: compounded monthly return
        - NAV_accumulated: last NAV within the month
        - profit: summed profit within the month
        - year: integer year (e.g., 2020)
        - month: integer month number (1-12)
        - month_name: abbreviated month name (e.g., "Jan", "Feb")

        Raises:
            ValueError: If the portfolio data has no 'date' column. Monthly
                aggregation requires temporal date information.
        """
        if "date" not in self.prices.columns:
            raise MissingDateColumnError("monthly")
        daily = self.returns.select(["date", "returns", "profit", "NAV_accumulated"])  # ensure only required columns
        monthly = (
            daily.group_by_dynamic(
                "date",
                every="1mo",
                period="1mo",
                label="left",
                closed="right",
            )
            .agg(
                [
                    pl.col("profit").sum().alias("profit"),
                    pl.col("NAV_accumulated").last().alias("NAV_accumulated"),
                    (pl.col("returns") + 1.0).product().alias("gross"),
                ]
            )
            .with_columns((pl.col("gross") - 1.0).alias("returns"))
            .select(["date", "returns", "NAV_accumulated", "profit"])  # keep month-end date
            .with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                    pl.col("date").dt.strftime("%b").alias("month_name"),
                ]
            )
            .sort("date")
        )
        return monthly

    @property
    def nav_compounded(self) -> pl.DataFrame:
        """Compute compounded NAV from returns (profit/AUM), preserving 'date'."""
        # self.returns contains 'date' and scaled 'profit' (i.e., returns)
        return self.returns.with_columns(((pl.col("returns") + 1.0).cum_prod() * self.aum).alias("NAV_compounded"))

    @property
    def highwater(self) -> pl.DataFrame:
        """Return the cumulative maximum of NAV as the high-water mark series.

        The resulting DataFrame preserves the 'date' column and adds a
        'highwater' column computed as the cumulative maximum of
        'NAV_accumulated'.
        """
        return self.returns.with_columns(pl.col("NAV_accumulated").cum_max().alias("highwater"))

    @property
    def drawdown(self) -> pl.DataFrame:
        """Return drawdown as the distance from high-water mark to current NAV.

        Computes 'drawdown' = 'highwater' - 'NAV_accumulated' and preserves the
        'date' column alongside the intermediate columns.
        """
        return self.highwater.with_columns(
            (pl.col("highwater") - pl.col("NAV_accumulated")).alias("drawdown"),
            ((pl.col("highwater") - pl.col("NAV_accumulated")) / pl.col("highwater")).alias("drawdown_pct"),
        )

    @property
    def all(self) -> pl.DataFrame:
        """Return a merged view of drawdown and compounded NAV.

        When a 'date' column is present the two frames are joined on that
        column to ensure temporal alignment.  When the data is integer-indexed
        (no 'date' column) the frames are stacked horizontally - they are
        guaranteed to have identical row counts because both are derived from
        the same source portfolio.
        """
        # Start with drawdown (includes date, NAV_accumulated, highwater, drawdown, drawdown_pct, etc.)
        left = self.drawdown
        # From nav_compounded, only take the additional compounded NAV column to avoid duplicate fields
        if "date" in left.columns:
            right = self.nav_compounded.select(["date", "NAV_compounded"])
            return left.join(right, on="date", how="inner")
        else:
            right = self.nav_compounded.select(["NAV_compounded"])
            return left.hstack(right)

    @property
    def stats(self) -> Stats:
        """Return a Stats object built from the portfolio's daily returns.

        Constructs a basanos.analytics.Stats instance from the portfolio
        returns. When a 'date' column is present both 'date' and 'returns'
        are passed to Stats; otherwise only 'returns' is used.
        """
        cols = ["date", "returns"] if "date" in self.returns.columns else ["returns"]
        return Stats(data=self.returns.select(cols))

    def truncate(self, start: object = None, end: object = None) -> "Portfolio":
        """Return a new Portfolio truncated to the inclusive [start, end] range.

        When a 'date' column is present in both prices and cash positions,
        truncation is performed by comparing the 'date' column against
        ``start`` and ``end`` (which should be date/datetime values or strings
        parseable by Polars).

        When the 'date' column is absent, integer-based row slicing is used
        instead.  In this case ``start`` and ``end`` must be non-negative
        integers representing 0-based row indices.  Passing non-integer bounds
        to an integer-indexed portfolio raises ``TypeError``.

        In all cases the ``aum`` value is preserved.

        Args:
            start: Optional lower bound (inclusive). A date/datetime or
                Polars-parseable string when a 'date' column exists; a
                non-negative int row index when the data has no 'date' column.
            end: Optional upper bound (inclusive). Same type rules as ``start``.

        Returns:
            A new Portfolio instance with prices and cash positions filtered to
            the specified range.

        Raises:
            TypeError: When the portfolio has no 'date' column and a non-integer
                bound is supplied.
        """
        has_date = "date" in self.prices.columns
        if has_date:
            cond = pl.lit(True)
            if start is not None:
                cond = cond & (pl.col("date") >= pl.lit(start))
            if end is not None:
                cond = cond & (pl.col("date") <= pl.lit(end))
            pr = self.prices.filter(cond)
            cp = self.cashposition.filter(cond)
        else:
            # Integer row-index slicing for date-free portfolios
            if start is not None and not isinstance(start, int):
                raise IntegerIndexBoundError("start", type(start).__name__)
            if end is not None and not isinstance(end, int):
                raise IntegerIndexBoundError("end", type(end).__name__)
            row_start = int(start) if start is not None else 0
            row_end = int(end) + 1 if end is not None else self.prices.height
            length = max(0, row_end - row_start)
            pr = self.prices.slice(row_start, length)
            cp = self.cashposition.slice(row_start, length)
        return Portfolio(prices=pr, cashposition=cp, aum=self.aum)

    def lag(self, n: int) -> "Portfolio":
        """Return a new Portfolio with cash positions lagged by ``n`` steps.

        This method shifts the numeric asset columns in the cashposition
        DataFrame by ``n`` rows, preserving the 'date' column and any
        non-numeric columns unchanged. Positive ``n`` delays weights
        (moves them down); negative ``n`` leads them (moves them up);
        ``n == 0`` returns the current portfolio unchanged.

        Notes:
            - Missing values introduced by the shift are left as nulls;
              downstream profit computation already guards and treats
              nulls as zero when multiplying by returns.

        Args:
            n: Number of rows to shift (can be negative, zero, or positive).

        Returns:
            A new Portfolio instance with lagged cash positions and the same
            prices/AUM as the original.
        """
        if not isinstance(n, int):
            raise TypeError
        if n == 0:
            return self

        # Identify numeric asset columns (exclude 'date')
        assets = [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]

        # Shift numeric columns by n; keep others as-is
        cp_lagged = self.cashposition.with_columns(pl.col(c).shift(n) for c in assets)
        return Portfolio(prices=self.prices, cashposition=cp_lagged, aum=self.aum)

    def smoothed_holding(self, n: int) -> "Portfolio":
        """Return a new Portfolio with cash positions smoothed by a rolling mean.

        Applies a trailing window average over the last ``n`` steps for each
        numeric asset column (excluding 'date'). The window length is ``n + 1``
        so that:
        - n=0 returns the original weights (no smoothing),
        - n=1 averages the current and previous weights,
        - n=k averages the current and last k weights.

        Args:
            n: Non-negative integer specifying how many previous steps to include.

        Returns:
            A new Portfolio with smoothed cash positions and the same prices/AUM.
        """
        if not isinstance(n, int):
            raise TypeError
        if n < 0:
            raise ValueError
        if n == 0:
            return self

        # Identify numeric asset columns (exclude 'date')
        assets = [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]

        window = n + 1
        # Apply rolling mean per numeric asset column; keep others unchanged
        cp_smoothed = self.cashposition.with_columns(
            pl.col(c).rolling_mean(window_size=window, min_samples=1).alias(c) for c in assets
        )
        return Portfolio(prices=self.prices, cashposition=cp_smoothed, aum=self.aum)

    @property
    def plots(self) -> Plots:
        """Convenience accessor returning a Plots facade for this portfolio.

        Use this to create Plotly visualizations such as snapshots, lagged
        performance curves, and lead/lag IR charts.

        Returns:
            basanos.analytics._plots.Plots: Helper object with plotting methods.
        """
        return Plots(self)

    @property
    def assets(self) -> list[str]:
        """List the asset column names from prices (numeric columns).

        Returns:
            list[str]: Names of numeric columns in prices; typically excludes 'date'.
        """
        return [c for c in self.prices.columns if self.prices[c].dtype.is_numeric()]

    @property
    def tilt(self) -> "Portfolio":
        """Return the 'tilt' portfolio with constant average weights.

        Computes the time-average of each asset's cash position (ignoring nulls/NaNs)
        and builds a new Portfolio with those constant weights applied across time.
        Prices and AUM are preserved.
        """
        const_position = self.cashposition.with_columns(
            pl.col(col).drop_nulls().drop_nans().mean().alias(col) for col in self.assets
        )

        return Portfolio.from_cash_position(self.prices, const_position, aum=self.aum)

    @property
    def timing(self) -> "Portfolio":
        """Return the 'timing' portfolio capturing deviations from the tilt.

        Constructs weights as original cash positions minus the tilt's constant
        positions, per asset. This isolates timing (alloc-demeaned) effects.
        Prices and AUM are preserved.
        """
        const_position = self.tilt.cashposition
        # subtracting frames is subtle as it would also try to subtract the date column
        position = self.cashposition.with_columns((pl.col(col) - const_position[col]).alias(col) for col in self.assets)
        return Portfolio.from_cash_position(self.prices, position, aum=self.aum)

    @property
    def tilt_timing_decomp(self) -> pl.DataFrame:
        """Return the portfolio's tilt/timing NAV decomposition.

        When a 'date' column is present the three NAV series are joined on it.
        When data is integer-indexed the frames are stacked horizontally.
        """
        if "date" in self.nav_accumulated.columns:
            nav_portfolio = self.nav_accumulated.select(["date", "NAV_accumulated"])
            nav_tilt = self.tilt.nav_accumulated.select(["date", "NAV_accumulated"])
            nav_timing = self.timing.nav_accumulated.select(["date", "NAV_accumulated"])

            # Join all three DataFrames on the 'date' column
            merged_df = nav_portfolio.join(nav_tilt, on="date", how="inner", suffix="_tilt").join(
                nav_timing, on="date", how="inner", suffix="_timing"
            )
        else:
            nav_portfolio = self.nav_accumulated.select(["NAV_accumulated"])
            nav_tilt = self.tilt.nav_accumulated.select(["NAV_accumulated"]).rename(
                {"NAV_accumulated": "NAV_accumulated_tilt"}
            )
            nav_timing = self.timing.nav_accumulated.select(["NAV_accumulated"]).rename(
                {"NAV_accumulated": "NAV_accumulated_timing"}
            )
            merged_df = nav_portfolio.hstack(nav_tilt).hstack(nav_timing)

        merged_df = merged_df.rename(
            {"NAV_accumulated_tilt": "tilt", "NAV_accumulated_timing": "timing", "NAV_accumulated": "portfolio"}
        )

        return merged_df

    @property
    def turnover(self) -> pl.DataFrame:
        """Daily one-way portfolio turnover as a fraction of AUM.

        Computes the sum of absolute position changes across all assets for each
        period, normalised by AUM.  The first row is always zero because there is
        no prior position to form a difference against.

        Returns:
            pl.DataFrame: Frame with an optional ``'date'`` column and a
            ``'turnover'`` column (dimensionless fraction of AUM).

        Examples:
            >>> import polars as pl
            >>> from datetime import date
            >>> _d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
            >>> prices = pl.DataFrame({"date": _d, "A": [100.0, 110.0, 121.0]})
            >>> pos = pl.DataFrame({"date": prices["date"], "A": [1000.0, 1200.0, 900.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)
            >>> pf.turnover["turnover"].to_list()
            [0.0, 0.002, 0.003]
        """
        assets = [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]
        daily_abs_chg = (pl.sum_horizontal(pl.col(c).diff().abs().fill_null(0.0) for c in assets) / self.aum).alias(
            "turnover"
        )
        cols: list[str | pl.Expr] = []
        if "date" in self.cashposition.columns:
            cols.append("date")
        cols.append(daily_abs_chg)
        return self.cashposition.select(cols)

    @property
    def turnover_weekly(self) -> pl.DataFrame:
        """Weekly aggregated one-way portfolio turnover as a fraction of AUM.

        When a ``'date'`` column is present, sums the daily turnover within each
        calendar week (Monday-based ``group_by_dynamic``).  Without a date
        column, a rolling 5-period sum with ``min_samples=5`` is returned
        (the first four rows will be ``null``).

        Returns:
            pl.DataFrame: Frame with an optional ``'date'`` column (week start)
            and a ``'turnover'`` column (fraction of AUM, summed over the week).

        Raises:
            MissingDateColumnError: Never — returns a rolling result when date
                is absent.
        """
        daily = self.turnover
        if "date" not in daily.columns:
            return daily.with_columns(pl.col("turnover").rolling_sum(window_size=5, min_samples=5))
        return daily.group_by_dynamic("date", every="1w").agg(pl.col("turnover").sum()).sort("date")

    def turnover_summary(self) -> pl.DataFrame:
        """Return a summary DataFrame of turnover statistics.

        Computes three metrics from the daily turnover series:

        - ``mean_daily_turnover``: mean of daily one-way turnover (fraction of AUM).
        - ``mean_weekly_turnover``: mean of weekly-aggregated turnover (fraction of AUM).
        - ``turnover_std``: standard deviation of daily turnover (fraction of AUM);
          complements the mean to detect regime switches.

        Returns:
            pl.DataFrame: One row per metric with columns ``'metric'`` and
            ``'value'``.

        Examples:
            >>> import polars as pl
            >>> from datetime import date, timedelta
            >>> import numpy as np
            >>> start = date(2020, 1, 1)
            >>> dates = pl.date_range(start=start, end=start + timedelta(days=9), interval="1d", eager=True)
            >>> prices = pl.DataFrame({"date": dates, "A": pl.Series(np.ones(10) * 100.0)})
            >>> pos = pl.DataFrame({"date": dates, "A": pl.Series([float(i) * 100 for i in range(10)])})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e4)
            >>> summary = pf.turnover_summary()
            >>> list(summary["metric"])
            ['mean_daily_turnover', 'mean_weekly_turnover', 'turnover_std']
        """
        daily_col = self.turnover["turnover"]
        _mean = daily_col.mean()
        mean_daily = float(_mean) if isinstance(_mean, (int, float)) else 0.0
        _std = daily_col.std()
        std_daily = float(_std) if isinstance(_std, (int, float)) else 0.0
        weekly_col = self.turnover_weekly["turnover"].drop_nulls()
        _weekly_mean = weekly_col.mean()
        mean_weekly = (
            float(_weekly_mean) if weekly_col.len() > 0 and isinstance(_weekly_mean, (int, float)) else float("nan")
        )
        return pl.DataFrame(
            {
                "metric": ["mean_daily_turnover", "mean_weekly_turnover", "turnover_std"],
                "value": [mean_daily, mean_weekly, std_daily],
            }
        )

    def correlation(self, frame: pl.DataFrame, name: str = "portfolio") -> pl.DataFrame:
        """Compute a correlation matrix of asset returns plus the portfolio.

        Computes percentage changes for all numeric columns in ``frame``,
        appends the portfolio profit series under the provided ``name``, and
        returns the Pearson correlation matrix across all numeric columns.

        Args:
            frame: A Polars DataFrame containing at least the asset price
                columns (and a date column which will be ignored if non-numeric).
            name: The column name to use when adding the portfolio profit
                series to the input frame.

        Returns:
            A square Polars DataFrame where each cell is the correlation
            between a pair of series (values in [-1, 1]).
        """
        # 1. Compute percentage change for all float columns
        p = frame.with_columns(cs.by_dtype(pl.Float32, pl.Float64).pct_change())

        # 2. Add the portfolio column from self.profit["profit"]
        p = p.with_columns(pl.Series(name, self.profit["profit"]))

        # 3. Compute correlation matrix
        corr_matrix = p.select(cs.numeric()).fill_null(0.0).corr()

        return corr_matrix
