"""Analytics subpackage for basanos.

Provides the Portfolio data model and related analytics helpers. Public
classes are kept in stable modules while implementation details live in
private modules so that documentation remains clean and pdoc renders a
hierarchical navigation.

Public API:
- basanos.analytics.portfolio — home of basanos.analytics.portfolio.Portfolio.

Private modules (subject to change):
- basanos.analytics._portfolio_data
- basanos.analytics._stats
- basanos.analytics._plots

Usage:
    Prefer importing the public classes from this package namespace:

    >>> from basanos.analytics import Portfolio  # doctest: +SKIP
    >>> from basanos.analytics import Portfolio
    >>> issubclass(Portfolio, object)
    True

Notes:
    Direct imports from private modules are discouraged as they may change
    without notice.
"""

# Public re-exports (explicit aliases so linters recognize intent)
from .portfolio import Portfolio as Portfolio
