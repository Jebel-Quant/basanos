"""Additional tests for Stats initialization error branches.

Covers the TypeError when input is not a Polars DataFrame and the
ValueError when the input frame is empty (height == 0).
"""

from __future__ import annotations

import polars as pl
import pytest

from basanos.analytics._stats import Stats


def test_stats_init_raises_typeerror_when_not_dataframe():
    """Passing a non-DataFrame to Stats should raise TypeError in __post_init__."""
    with pytest.raises(TypeError):
        _ = Stats(data={"date": [1, 2, 3], "A": [0.1, 0.2, 0.3]})


def test_stats_init_raises_valueerror_on_empty_dataframe():
    """Passing an empty DataFrame should raise ValueError in __post_init__."""
    # Build an empty frame with proper dtypes
    empty_df = pl.DataFrame(
        {
            "date": pl.Series("date", [], dtype=pl.Datetime("ns")),
            "A": pl.Series("A", [], dtype=pl.Float64),
        }
    )

    with pytest.raises(ValueError, match=r".*"):
        _ = Stats(data=empty_df)
