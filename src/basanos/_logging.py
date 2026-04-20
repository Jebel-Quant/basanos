"""Structured JSON logging for basanos.

Provides a `JSONFormatter` that applications can attach to any
`Handler` to receive log records as JSON objects with a
consistent schema::

    {
        "timestamp": "2024-01-01T00:00:00",
        "level": "WARNING",
        "logger": "basanos.math.optimizer",
        "event": "<formatted log message>",
        "context": { ... }   # present when extra={"context": {...}} is used
    }

Usage example::

    import logging
    from basanos import JSONFormatter

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.getLogger("basanos").addHandler(handler)
    logging.getLogger("basanos").setLevel(logging.DEBUG)
"""

import json
import logging
import math
from typing import Any

# Attributes that belong to logging.LogRecord itself and must not be
# re-emitted as extra context fields in the JSON payload.
_STDLIB_RECORD_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def _to_serialisable(value: Any) -> Any:
    """Recursively coerce *value* into a JSON-serialisable form.

    Non-finite `float` values (``nan``, ``inf``, ``-inf``) are
    converted to their `str` representation so that the resulting JSON
    is strictly valid (RFC 8259 does not permit ``NaN`` or ``Infinity``).
    `dict`, `list`, and `tuple` containers are traversed
    recursively; all other non-serialisable types are handled by the
    ``default=str`` fallback in `dumps`.

    Args:
        value: The value to coerce.

    Returns:
        A JSON-serialisable representation of *value*.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(v) for v in value]
    return value


class JSONFormatter(logging.Formatter):
    """Log formatter that serialises each record as a single-line JSON object.

    Applications can attach this formatter to any `Handler` to
    receive machine-readable, structured log output from the *basanos* library.

    The JSON payload always contains:

    * ``timestamp`` - wall-clock time of the record formatted with *datefmt*.
    * ``level`` - upper-case level name (e.g. ``"WARNING"``).
    * ``logger`` - dotted logger name (e.g. ``"basanos.math.optimizer"``).
    * ``event`` - the fully-formatted log message.

    Any extra fields supplied by the caller via the ``extra=`` keyword
    argument to `warning` (or equivalent) are merged
    into the JSON object at the top level.  The conventional field for
    structured context is ``"context"`` (a plain `dict`), but any
    JSON-serialisable extra key is accepted.

    Non-finite `float` values (``nan``, ``inf``) and other
    non-serialisable types are converted to their `str` representation
    automatically, so the formatter never raises on unexpected types (e.g.
    `nan`, `float64`, `date`).

    The produced JSON is strictly RFC 8259-compliant (no bare ``NaN`` or
    ``Infinity`` tokens).

    Example::

        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logging.getLogger("basanos").addHandler(handler)

    Args:
        datefmt: Optional `strftime` format string for the
            ``timestamp`` field.  Defaults to ISO-8601
            (``"%Y-%m-%dT%H:%M:%S"``).
    """

    _ISO_FMT = "%Y-%m-%dT%H:%M:%S"

    def __init__(self, datefmt: str | None = None) -> None:
        super().__init__(datefmt=datefmt or self._ISO_FMT)

    def format(self, record: logging.LogRecord) -> str:
        """Return the log record serialised as a JSON string.

        Args:
            record: The `LogRecord` to format.

        Returns:
            A single-line JSON string.
        """
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        # Merge any extra fields supplied by the caller (e.g. "context").
        for key, value in record.__dict__.items():
            if key not in _STDLIB_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = _to_serialisable(value)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)
