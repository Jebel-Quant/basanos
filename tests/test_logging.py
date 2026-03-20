"""Tests for basanos._logging (JSONFormatter).

Covers:
- JSONFormatter produces valid JSON with required fields.
- JSONFormatter includes extra "context" field when supplied.
- The optimizer warning carries structured context on the log record.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from basanos import JSONFormatter
from basanos.math import BasanosConfig, BasanosEngine

# ---------------------------------------------------------------------------
# JSONFormatter unit tests
# ---------------------------------------------------------------------------


def _make_record(
    msg: str,
    *args: object,
    level: int = logging.WARNING,
    name: str = "test.logger",
    extra: dict | None = None,
) -> logging.LogRecord:
    """Return a :class:`logging.LogRecord` suitable for formatter tests."""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=args,
        exc_info=None,
    )
    if extra:
        for k, v in extra.items():
            setattr(record, k, v)
    return record


class TestJSONFormatterRequiredFields:
    """JSONFormatter always emits timestamp, level, logger, and event."""

    def test_produces_valid_json(self) -> None:
        """format() output must be parseable as JSON."""
        formatter = JSONFormatter()
        record = _make_record("hello world")
        raw = formatter.format(record)
        json.loads(raw)  # raises if invalid

    def test_level_field(self) -> None:
        """The ``level`` field must match the record's level name."""
        formatter = JSONFormatter()
        record = _make_record("msg", level=logging.WARNING)
        data = json.loads(formatter.format(record))
        assert data["level"] == "WARNING"

    def test_logger_field(self) -> None:
        """The ``logger`` field must match the record's logger name."""
        formatter = JSONFormatter()
        record = _make_record("msg", name="basanos.math.optimizer")
        data = json.loads(formatter.format(record))
        assert data["logger"] == "basanos.math.optimizer"

    def test_event_field_contains_formatted_message(self) -> None:
        """The ``event`` field must be the %-formatted log message."""
        formatter = JSONFormatter()
        record = _make_record("value=%s", 42)
        data = json.loads(formatter.format(record))
        assert data["event"] == "value=42"

    def test_timestamp_field_present(self) -> None:
        """A non-empty ``timestamp`` field must always be present."""
        formatter = JSONFormatter()
        record = _make_record("msg")
        data = json.loads(formatter.format(record))
        assert "timestamp" in data
        assert data["timestamp"]  # non-empty string

    def test_custom_datefmt(self) -> None:
        """A custom *datefmt* must be reflected in the ``timestamp`` field."""
        formatter = JSONFormatter(datefmt="%Y/%m/%d")
        record = _make_record("msg")
        data = json.loads(formatter.format(record))
        # Verify the timestamp uses the custom format (contains slashes, not dashes).
        assert "/" in data["timestamp"]


class TestJSONFormatterExtraFields:
    """Extra fields supplied via extra= are merged into the JSON payload."""

    def test_context_field_included(self) -> None:
        """A ``context`` dict supplied via ``extra=`` must appear in the JSON."""
        formatter = JSONFormatter()
        ctx = {"denom": 0.5, "denom_tol": 1e-8, "t": "2024-01-01"}
        record = _make_record("event", extra={"context": ctx})
        data = json.loads(formatter.format(record))
        assert data["context"] == ctx

    def test_arbitrary_extra_key_included(self) -> None:
        """Arbitrary extra keys must be promoted to the top-level JSON object."""
        formatter = JSONFormatter()
        record = _make_record("event", extra={"request_id": "abc-123"})
        data = json.loads(formatter.format(record))
        assert data["request_id"] == "abc-123"

    def test_nan_value_serialised_as_string(self) -> None:
        """Non-JSON-serialisable values (e.g. nan) must not raise."""
        formatter = JSONFormatter()
        record = _make_record("event", extra={"context": {"denom": float("nan")}})
        raw = formatter.format(record)
        data = json.loads(raw)
        assert data["context"]["denom"] == "nan"

    def test_list_and_tuple_values_serialised(self) -> None:
        """List and tuple extra values are recursively serialised."""
        formatter = JSONFormatter()
        record = _make_record(
            "event",
            extra={"items": [1.0, float("inf"), (2.0, float("nan"))]},
        )
        data = json.loads(formatter.format(record))
        assert data["items"][0] == 1.0
        assert data["items"][1] == "inf"
        assert data["items"][2][1] == "nan"

    def test_exc_info_included_when_present(self) -> None:
        """exc_info is formatted and included in the JSON payload."""
        import sys

        formatter = JSONFormatter()
        exc_info = None
        try:
            _ = 1 / 0
        except ZeroDivisionError:
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="error",
            args=(),
            exc_info=exc_info,
        )
        data = json.loads(formatter.format(record))
        assert "exc_info" in data
        assert "ZeroDivisionError" in data["exc_info"]

    def test_stdlib_attrs_not_duplicated(self) -> None:
        """Standard LogRecord attributes should not appear as extra keys."""
        formatter = JSONFormatter()
        record = _make_record("msg", level=logging.DEBUG)
        data = json.loads(formatter.format(record))
        # These stdlib attrs should not be promoted to top-level JSON keys.
        for attr in ("levelno", "msecs", "relativeCreated", "thread"):
            assert attr not in data, f"Unexpected stdlib attribute in JSON: {attr!r}"


# ---------------------------------------------------------------------------
# Integration: optimizer warning carries structured context
# ---------------------------------------------------------------------------


@pytest.fixture
def degen_engine() -> BasanosEngine:
    """BasanosEngine configured so that the degenerate-denominator guard fires."""
    n = 60
    start = date(2022, 1, 3)
    dates = pl.date_range(
        start=start,
        end=start + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    rng = np.random.default_rng(0)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(100.0 + np.cumsum(rng.normal(0, 1, n)), dtype=pl.Float64),
            "B": pl.Series(200.0 + np.cumsum(rng.normal(0, 1, n)), dtype=pl.Float64),
        }
    )
    theta = np.linspace(0.0, 4.0 * np.pi, num=n)
    mu = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.tanh(np.sin(theta)), dtype=pl.Float64),
            "B": pl.Series(np.tanh(np.cos(theta)), dtype=pl.Float64),
        }
    )
    cfg = BasanosConfig(corr=20, vola=12, clip=4.0, shrink=0.7, aum=1e6, denom_tol=1e6)
    return BasanosEngine(prices=prices, cfg=cfg, mu=mu)


class TestOptimizerStructuredContext:
    """Optimizer warning includes a structured 'context' field on the log record."""

    def test_warning_has_context_attribute(self, degen_engine: BasanosEngine, caplog: pytest.LogCaptureFixture) -> None:
        """Each degenerate-denominator warning must carry a ``context`` attribute."""
        with caplog.at_level(logging.WARNING, logger="basanos.math.optimizer"):
            _ = degen_engine.cash_position

        degen = [r for r in caplog.records if "normalisation denominator is degenerate" in r.message]
        assert degen, "Expected at least one degenerate-denominator warning"
        for record in degen:
            assert hasattr(record, "context"), "Log record is missing 'context' attribute"

    def test_context_contains_expected_keys(
        self, degen_engine: BasanosEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The ``context`` dict must contain ``t``, ``denom``, and ``denom_tol``."""
        with caplog.at_level(logging.WARNING, logger="basanos.math.optimizer"):
            _ = degen_engine.cash_position

        degen = [r for r in caplog.records if "normalisation denominator is degenerate" in r.message]
        assert degen
        for record in degen:
            ctx = record.context  # type: ignore[attr-defined]
            assert isinstance(ctx, dict)
            assert "t" in ctx
            assert "denom" in ctx
            assert "denom_tol" in ctx

    def test_json_formatter_round_trips_context(
        self, degen_engine: BasanosEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        """JSONFormatter can serialise the optimizer warning to valid JSON."""
        formatter = JSONFormatter()

        with caplog.at_level(logging.WARNING, logger="basanos.math.optimizer"):
            _ = degen_engine.cash_position

        degen = [r for r in caplog.records if "normalisation denominator is degenerate" in r.message]
        assert degen
        for record in degen:
            raw = formatter.format(record)
            data = json.loads(raw)
            assert "timestamp" in data
            assert data["level"] == "WARNING"
            assert data["logger"] == "basanos.math._engine_solve"
            assert "normalisation denominator is degenerate" in data["event"]
            assert "context" in data
            assert "denom" in data["context"]
            assert "denom_tol" in data["context"]
