"""Analytics subpackage for basanos.

Provides the Portfolio data model and related analytics helpers. Public
classes are kept in stable modules while implementation details live in
private modules so that documentation remains clean and pdoc renders a
hierarchical navigation.

Public API:
- basanos.analytics.portfolio — home of basanos.analytics.portfolio.Portfolio.
- basanos.analytics._portfolio_data — home of
  basanos.analytics._portfolio_data.PortfolioData.

Private modules (subject to change):
- basanos.analytics._stats
- basanos.analytics._plots

Usage:
    Prefer importing the public classes from this package namespace:

    >>> from basanos.analytics import Portfolio  # doctest: +SKIP
    >>> from basanos.analytics import Portfolio
    >>> issubclass(Portfolio, object)
    True
    >>> from basanos.analytics import PortfolioData
    >>> issubclass(PortfolioData, object)
    True

Notes:
    Direct imports from private modules are discouraged as they may change
    without notice.
"""

# Public re-exports (explicit aliases so linters recognize intent)
from ._portfolio_data import PortfolioData as PortfolioData
from .portfolio import Portfolio as Portfolio
