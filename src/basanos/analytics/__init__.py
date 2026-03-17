"""Analytics subpackage for basanos.

Provides the Portfolio data model and related analytics helpers. Public
classes are kept in stable modules while implementation details live in
private modules so that documentation remains clean and pdoc renders a
hierarchical navigation.

Public API:
- basanos.analytics.portfolio — home of basanos.analytics.portfolio.Portfolio.
- basanos.analytics._async — home of basanos.analytics._async.AsyncPortfolio.

Private modules (subject to change):
- basanos.analytics._stats
- basanos.analytics._plots

Usage:
    Prefer importing the public classes from this package namespace:

    >>> from basanos.analytics import Portfolio  # doctest: +SKIP
    >>> from basanos.analytics import Portfolio
    >>> issubclass(Portfolio, object)
    True

    For async (non-blocking) usage in event-loop applications:

    >>> from basanos.analytics import AsyncPortfolio  # doctest: +SKIP
    >>> from basanos.analytics import AsyncPortfolio
    >>> issubclass(AsyncPortfolio, object)
    True

Notes:
    Direct imports from private modules are discouraged as they may change
    without notice.
"""

# Public re-exports (explicit aliases so linters recognize intent)
from ._async import AsyncPortfolio as AsyncPortfolio
from .portfolio import Portfolio as Portfolio
