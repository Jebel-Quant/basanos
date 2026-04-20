"""Deprecation warning utilities for the Basanos package.

Use `warn_deprecated` whenever a public API is scheduled for removal.
The standard policy (documented in ``SECURITY.md``) is:

- Deprecated in ``0.x`` → removed no earlier than ``0.x+2``.
- A ``DeprecationWarning`` is issued on every call to the deprecated code path.

Examples:
    >>> import warnings
    >>> from basanos._deprecation import warn_deprecated
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     warn_deprecated("old_func", since="0.6", remove_in="0.8")
    ...     assert len(w) == 1
    ...     assert issubclass(w[0].category, DeprecationWarning)
    ...     assert "old_func" in str(w[0].message)
    ...     assert "0.6" in str(w[0].message)
    ...     assert "0.8" in str(w[0].message)
"""

from __future__ import annotations

import warnings


def warn_deprecated(
    name: str,
    *,
    since: str,
    remove_in: str,
    replacement: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit a `DeprecationWarning` for a deprecated public API.

    Call this function at the top of any deprecated function, method, or
    property so users see the warning on the first call.

    Args:
        name: The name of the deprecated symbol (e.g. ``"BasanosEngine.old_method"``).
        since: The version in which the deprecation was introduced (e.g. ``"0.6"``).
        remove_in: The earliest version in which the symbol may be removed (e.g. ``"0.8"``).
        replacement: Optional name of the recommended replacement.  If provided it is
            included in the warning message.
        stacklevel: Passed directly to `warn`.  The default of ``2``
            means the warning points at the *caller* of the deprecated code rather than
            at this helper function.

    Examples:
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     warn_deprecated("foo", since="0.6", remove_in="0.8")
        ...     assert len(w) == 1
        ...     assert issubclass(w[0].category, DeprecationWarning)

        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     warn_deprecated(
        ...         "foo",
        ...         since="0.6",
        ...         remove_in="0.8",
        ...         replacement="bar",
        ...     )
        ...     assert "bar" in str(w[0].message)
    """
    msg = f"{name} is deprecated since {since} and will be removed in {remove_in}."
    if replacement is not None:
        msg = f"{msg} Use {replacement} instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)
