"""Input validation for `BasanosEngine`.

Extracted from ``optimizer.py`` so the engine facade stays focused on the
core position-solving logic.  Every name defined here is re-exported from
``optimizer`` so existing callers (and tests that import ``_validate_inputs`` /
``_validate_null_fraction`` from ``basanos.math.optimizer``) are unaffected.
"""

from __future__ import annotations

import logging

import polars as pl

from ..exceptions import (
    ColumnMismatchError,
    ExcessiveNullsError,
    MissingDateColumnError,
    MonotonicPricesError,
    NonPositivePricesError,
    ShapeMismatchError,
)
from ._config import BasanosConfig, CovarianceMode

_logger = logging.getLogger(__name__)


def _validate_required_date_columns(prices: pl.DataFrame, mu: pl.DataFrame) -> None:
    """Ensure both input frames expose the required ``date`` column."""
    if "date" not in prices.columns:
        raise MissingDateColumnError("prices")
    if "date" not in mu.columns:
        raise MissingDateColumnError("mu")


def _validate_shape_and_column_sets(prices: pl.DataFrame, mu: pl.DataFrame) -> None:
    """Ensure prices and signals are shape- and schema-compatible."""
    if prices.shape != mu.shape:
        raise ShapeMismatchError(prices.shape, mu.shape)
    if not set(prices.columns) == set(mu.columns):
        raise ColumnMismatchError(prices.columns, mu.columns)


def _numeric_assets(prices: pl.DataFrame) -> list[str]:
    """Return numeric asset columns, excluding the ``date`` column."""
    return [c for c in prices.columns if c != "date" and prices[c].dtype.is_numeric()]


def _validate_positive_prices(prices: pl.DataFrame, assets: list[str]) -> None:
    """Ensure all finite/non-null prices are strictly positive."""
    for asset in assets:
        col = prices[asset].drop_nulls()
        if col.len() > 0 and (col <= 0).any():
            raise NonPositivePricesError(asset)


def _validate_null_fraction(prices: pl.DataFrame, assets: list[str], max_nan_fraction: float) -> None:
    """Reject asset columns whose null fraction exceeds configuration bounds."""
    n_rows = prices.height
    if n_rows == 0:
        return
    for asset in assets:
        nan_frac = prices[asset].null_count() / n_rows
        if nan_frac > max_nan_fraction:
            raise ExcessiveNullsError(asset, nan_frac, max_nan_fraction)


def _validate_non_monotonic_prices(prices: pl.DataFrame, assets: list[str]) -> None:
    """Reject monotonic asset series that indicate malformed synthetic data."""
    for asset in assets:
        col = prices[asset].drop_nulls()
        if col.len() > 2:
            diffs = col.diff().drop_nulls()
            if (diffs >= 0).all() or (diffs <= 0).all():
                raise MonotonicPricesError(asset)


def _warn_short_sliding_window_data(prices: pl.DataFrame, cfg: BasanosConfig) -> None:
    """Emit a warning when data is too short relative to the configured SW window."""
    if cfg.covariance_mode == CovarianceMode.sliding_window and cfg.window is not None:
        n_rows = prices.height
        w: int = cfg.window
        if n_rows < 2 * w:
            _logger.warning(
                "Dataset length (%d rows) is less than 2 * window (%d). "
                "The first %d timestamps will yield zero positions during warm-up; "
                "consider using a longer history or reducing 'window'.",
                n_rows,
                2 * w,
                w - 1,
            )


def _validate_inputs(prices: pl.DataFrame, mu: pl.DataFrame, cfg: BasanosConfig) -> None:
    """Validate ``prices``, ``mu``, and ``cfg`` for use with `BasanosEngine`.

    Checks that both DataFrames contain a ``'date'`` column, share identical
    shapes and column sets, contain no non-positive prices, no excessive NaN
    fractions, and no monotonically non-varying price series.  Also emits a
    warning when the dataset is too short relative to a configured
    sliding-window size.

    Args:
        prices: DataFrame of price levels per asset over time.
        mu: DataFrame of expected-return signals aligned with ``prices``.
        cfg: Engine configuration instance.

    Raises:
        MissingDateColumnError: If ``'date'`` is absent from either frame.
        ShapeMismatchError: If ``prices`` and ``mu`` have different shapes.
        ColumnMismatchError: If the column sets of the two frames differ.
        NonPositivePricesError: If any asset contains a non-positive price.
        ExcessiveNullsError: If any asset column exceeds ``cfg.max_nan_fraction``.
        MonotonicPricesError: If any asset price series is monotonically
            non-decreasing or non-increasing.

    Warns:
        UserWarning (via logging): If ``cfg.covariance`` is a
            `SlidingWindowConfig` and
            ``len(prices) < 2 * cfg.covariance.window``, a warning is emitted
            via the module logger rather than an exception.  This is a
            deliberate soft boundary — callers may intentionally supply data
            shorter than the full warm-up period.  During warm-up the first
            ``window - 1`` timestamps will yield zero positions.
    """
    _validate_required_date_columns(prices, mu)
    _validate_shape_and_column_sets(prices, mu)
    assets = _numeric_assets(prices)
    _validate_positive_prices(prices, assets)
    _validate_null_fraction(prices, assets, cfg.max_nan_fraction)
    _validate_non_monotonic_prices(prices, assets)
    _warn_short_sliding_window_data(prices, cfg)
