"""Correlation-aware risk position optimizer (Basanos).

This module provides utilities to compute correlation-adjusted risk positions
from price data and expected-return signals. It relies on volatility-adjusted
returns to estimate a dynamic correlation matrix (via EWM), applies shrinkage
towards identity, and solves a normalized linear system per timestamp to
obtain stable positions.

Performance characteristics
---------------------------
Let *N* be the number of assets and *T* the number of timestamps.

**Computational complexity**

+----------------------------------+------------------+--------------------------------------+
| Operation                        | Complexity       | Bottleneck                           |
+==================================+==================+======================================+
| EWM volatility (``ret_adj``,     | O(T·N)           | Linear in both T and N; negligible   |
| ``vola``)                        |                  |                                      |
+----------------------------------+------------------+--------------------------------------+
| EWM correlation (``cor``)        | O(T·N²)          | ``lfilter`` over all N² asset pairs  |
|                                  |                  | simultaneously                       |
+----------------------------------+------------------+--------------------------------------+
| Linear solve per timestamp       | O(N³)            | Cholesky / LU per row in             |
| (``cash_position``)              | * T solves       | ``cash_position``                    |
+----------------------------------+------------------+--------------------------------------+

**Memory usage** (peak, approximate)

``ewm_corr`` allocates roughly **14 float64 arrays** of shape
``(T, N, N)`` at peak (input sequences, IIR filter outputs, EWM components,
and the result tensor).  Peak RAM ≈ **112 * T * N²** bytes.  Typical
working sizes on a 16 GB machine:

+--------+--------------------------+------------------------------------+
| N      | T (daily rows)           | Peak memory (approx.)              |
+========+==========================+====================================+
| 50     | 252 (~1 yr)              | ~70 MB                             |
+--------+--------------------------+------------------------------------+
| 100    | 252 (~1 yr)              | ~280 MB                            |
+--------+--------------------------+------------------------------------+
| 100    | 2 520 (~10 yr)           | ~2.8 GB                            |
+--------+--------------------------+------------------------------------+
| 200    | 2 520 (~10 yr)           | ~11 GB                             |
+--------+--------------------------+------------------------------------+
| 500    | 2 520 (~10 yr)           | ~70 GB ⚠ exceeds typical RAM       |
+--------+--------------------------+------------------------------------+

**Practical limits (daily data)**

* **≤ 150 assets, ≤ 5 years** — well within reach on an 8 GB laptop.
* **≤ 250 assets, ≤ 10 years** — requires ~11-12 GB; feasible on a 16 GB
  workstation.
* **> 500 assets with multi-year history** — peak memory exceeds 16 GB;
  reduce the time range or switch to a chunked / streaming approach.
* **> 1 000 assets** — the O(N³) per-solve cost alone makes real-time
  optimization impractical even with adequate RAM.

See ``BENCHMARKS.md`` for measured wall-clock timings across representative
dataset sizes.

Internal structure
------------------
The implementation is split across focused private modules to keep each file
readable and independently testable:

* :mod:`basanos.math._config` — :class:`BasanosConfig` and all
  covariance-mode configuration classes.
* :mod:`basanos.math._ewm_corr` — :func:`ewm_corr`, the vectorised
  IIR-filter implementation of per-row EWM correlation matrices.
* :mod:`basanos.math._engine_solve` — private helpers providing the
  ``_iter_matrices`` and ``_iter_solve`` generators (per-timestamp solve
  logic).
* :mod:`basanos.math._engine_diagnostics` — private helpers providing
  matrix-quality diagnostics (condition number, effective rank, solver
  residual, signal utilisation).
* :mod:`basanos.math._engine_ic` — private helpers providing signal
  evaluation metrics (IC, Rank IC, ICIR, and summary statistics).
* This module — :class:`BasanosEngine`, a single flat class that wires
  every method together in clearly delimited sections.
"""

import dataclasses
import datetime
import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ..analytics import Portfolio
from ..exceptions import (
    ColumnMismatchError,
    ExcessiveNullsError,
    MissingDateColumnError,
    MonotonicPricesError,
    NonPositivePricesError,
    ShapeMismatchError,
)
from ._config import (
    BasanosConfig,
    CovarianceConfig,
    CovarianceMode,
    EwmaShrinkConfig,
    SlidingWindowConfig,
)
from ._engine_diagnostics import _DiagnosticsMixin as _DiagnosticsMixin
from ._engine_ic import _SignalEvaluatorMixin as _SignalEvaluatorMixin
from ._engine_solve import _SolveMixin as _SolveMixin
from ._ewm_corr import ewm_corr as _ewm_corr_numpy
from ._signal import vol_adj

if TYPE_CHECKING:
    from ._config_report import ConfigReport

_logger = logging.getLogger(__name__)


def _validate_inputs(prices: pl.DataFrame, mu: pl.DataFrame, cfg: "BasanosConfig") -> None:
    """Validate ``prices``, ``mu``, and ``cfg`` for use with :class:`BasanosEngine`.

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
            :class:`SlidingWindowConfig` and
            ``len(prices) < 2 * cfg.covariance.window``, a warning is emitted
            via the module logger rather than an exception.  This is a
            deliberate soft boundary — callers may intentionally supply data
            shorter than the full warm-up period.  During warm-up the first
            ``window - 1`` timestamps will yield zero positions.
    """
    # ensure 'date' column exists in prices before any other validation
    if "date" not in prices.columns:
        raise MissingDateColumnError("prices")

    # ensure 'date' column exists in mu as well (kept for symmetry and downstream assumptions)
    if "date" not in mu.columns:
        raise MissingDateColumnError("mu")

    # check that prices and mu have the same shape
    if prices.shape != mu.shape:
        raise ShapeMismatchError(prices.shape, mu.shape)

    # check that the columns of prices and mu are identical
    if not set(prices.columns) == set(mu.columns):
        raise ColumnMismatchError(prices.columns, mu.columns)

    assets = [c for c in prices.columns if c != "date" and prices[c].dtype.is_numeric()]

    # check for non-positive prices: log returns require strictly positive prices
    for asset in assets:
        col = prices[asset].drop_nulls()
        if col.len() > 0 and (col <= 0).any():
            raise NonPositivePricesError(asset)

    # check for excessive NaN values: more than cfg.max_nan_fraction null in any asset column
    n_rows = prices.height
    if n_rows > 0:
        for asset in assets:
            nan_frac = prices[asset].null_count() / n_rows
            if nan_frac > cfg.max_nan_fraction:
                raise ExcessiveNullsError(asset, nan_frac, cfg.max_nan_fraction)

    # check for monotonic price series: a strictly non-decreasing or non-increasing
    # series has no variance in its return sign, indicating malformed or synthetic data
    for asset in assets:
        col = prices[asset].drop_nulls()
        if col.len() > 2:
            diffs = col.diff().drop_nulls()
            if (diffs >= 0).all() or (diffs <= 0).all():
                raise MonotonicPricesError(asset)

    # warn when the dataset is too short to benefit from the sliding window
    if cfg.covariance_mode == CovarianceMode.sliding_window and cfg.window is not None:
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


# ---------------------------------------------------------------------------
# Re-export config symbols so ``from basanos.math.optimizer import …`` keeps
# working for existing callers.
# ---------------------------------------------------------------------------
__all__ = [
    "BasanosConfig",
    "BasanosEngine",
    "CovarianceConfig",
    "CovarianceMode",
    "EwmaShrinkConfig",
    "SlidingWindowConfig",
]


@dataclasses.dataclass(frozen=True)
class BasanosEngine(_DiagnosticsMixin, _SignalEvaluatorMixin, _SolveMixin):
    """Engine to compute correlation matrices and optimize risk positions.

    Encapsulates price data and configuration to build EWM-based
    correlations, apply shrinkage, and solve for normalized positions.

    Public methods are organised into clearly delimited sections (some
    inherited from the private mixin classes):

    * **Core data access** — :attr:`assets`, :attr:`ret_adj`, :attr:`vola`,
      :attr:`cor`, :attr:`cor_tensor`
    * **Solve / position logic** — :attr:`cash_position`,
      :attr:`position_status`, :attr:`risk_position`,
      :attr:`position_leverage`, :meth:`warmup_state`
      (solve helpers inherited from :class:`~._engine_solve._SolveMixin`)
    * **Portfolio and performance** — :attr:`portfolio`,
      :attr:`naive_sharpe`, :meth:`sharpe_at_shrink`,
      :meth:`sharpe_at_window_factors`
    * **Matrix diagnostics** — :attr:`condition_number`,
      :attr:`effective_rank`, :attr:`solver_residual`,
      :attr:`signal_utilisation`
      (inherited from :class:`~._engine_diagnostics._DiagnosticsMixin`)
    * **Signal evaluation** — :attr:`ic`, :attr:`rank_ic`, :attr:`ic_mean`,
      :attr:`ic_std`, :attr:`icir`, :attr:`rank_ic_mean`,
      :attr:`rank_ic_std`
      (inherited from :class:`~._engine_ic._SignalEvaluatorMixin`)
    * **Reporting** — :attr:`config_report`

    Data-flow diagram
    -----------------

    .. code-block:: text

        prices (pl.DataFrame)
          │
          ├─ vol_adj ──► ret_adj (volatility-adjusted log returns)
          │                │
          │                ├─ ewm_corr ──► cor / cor_tensor
          │                │                │
          │                │                └─ shrink2id / FactorModel
          │                │                        │
          │              vola                 covariance matrix
          │                │                        │
          └── mu ──────────┴── _iter_solve ──────────┘
                                    │
                              cash_position
                                    │
                           ┌────────┴────────┐
                       portfolio          diagnostics
                      (Portfolio)    (condition_number,
                                      effective_rank,
                                      solver_residual,
                                      signal_utilisation,
                                      ic, rank_ic, …)

    Attributes:
        prices: Polars DataFrame of price levels per asset over time.  Must
            contain a ``'date'`` column and at least one numeric asset column
            with strictly positive values that are not monotonically
            non-decreasing or non-increasing (i.e. they must vary in sign).
        mu: Polars DataFrame of expected-return signals aligned with *prices*.
            Must share the same shape and column names as *prices*.
        cfg: Immutable :class:`BasanosConfig` controlling EWMA half-lives,
            clipping, shrinkage intensity, and AUM.

    Examples:
        Build an engine with two synthetic assets over 30 days and inspect the
        optimized positions and diagnostic properties.

        >>> import numpy as np
        >>> import polars as pl
        >>> from basanos.math import BasanosConfig, BasanosEngine
        >>> dates = list(range(30))
        >>> rng = np.random.default_rng(42)
        >>> prices = pl.DataFrame({
        ...     "date": dates,
        ...     "A": np.cumprod(1 + rng.normal(0.001, 0.02, 30)) * 100.0,
        ...     "B": np.cumprod(1 + rng.normal(0.001, 0.02, 30)) * 150.0,
        ... })
        >>> mu = pl.DataFrame({
        ...     "date": dates,
        ...     "A": rng.normal(0.0, 0.5, 30),
        ...     "B": rng.normal(0.0, 0.5, 30),
        ... })
        >>> cfg = BasanosConfig(vola=5, corr=10, clip=2.0, shrink=0.5, aum=1_000_000)
        >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
        >>> engine.assets
        ['A', 'B']
        >>> engine.cash_position.shape
        (30, 3)
        >>> engine.position_leverage.columns
        ['date', 'leverage']
    """

    prices: pl.DataFrame
    mu: pl.DataFrame
    cfg: BasanosConfig

    def __post_init__(self) -> None:
        """Validate inputs by delegating to :func:`_validate_inputs`."""
        _validate_inputs(self.prices, self.mu, self.cfg)

    # ------------------------------------------------------------------
    # Core data-access properties
    # ------------------------------------------------------------------

    @property
    def assets(self) -> list[str]:
        """List asset column names (numeric columns excluding 'date')."""
        return [c for c in self.prices.columns if c != "date" and self.prices[c].dtype.is_numeric()]

    @property
    def ret_adj(self) -> pl.DataFrame:
        """Return per-asset volatility-adjusted log returns clipped by cfg.clip.

        Uses an EWMA volatility estimate with lookback ``cfg.vola`` to
        standardize log returns for each numeric asset column.
        """
        return self.prices.with_columns(
            [vol_adj(pl.col(asset), vola=self.cfg.vola, clip=self.cfg.clip) for asset in self.assets]
        )

    @property
    def vola(self) -> pl.DataFrame:
        """Per-asset EWMA volatility of percentage returns.

        Computes percent changes for each numeric asset column and applies an
        exponentially weighted standard deviation using the lookback specified
        by ``cfg.vola``. The result is a DataFrame aligned with ``self.prices``
        whose numeric columns hold per-asset volatility estimates.
        """
        return self.prices.with_columns(
            pl.col(asset)
            .pct_change()
            .ewm_std(com=self.cfg.vola - 1, adjust=True, min_samples=self.cfg.vola)
            .alias(asset)
            for asset in self.assets
        )

    @property
    def cor(self) -> dict[datetime.date, np.ndarray]:
        """Compute per-timestamp EWM correlation matrices.

        Builds volatility-adjusted returns for all assets, computes an
        exponentially weighted correlation using a pure NumPy implementation
        (with window ``cfg.corr``), and returns a mapping from each timestamp
        to the corresponding correlation matrix as a NumPy array.

        Returns:
            dict: Mapping ``date -> np.ndarray`` of shape (n_assets, n_assets).

        Performance:
            Delegates to :func:`ewm_corr`, which is O(T·N²) in both
            time and memory.  The returned dict holds *T* references into the
            result tensor (one N*N view per date); no extra copies are made.
            For large *N* or *T*, prefer ``cor_tensor`` to keep a single
            contiguous array rather than building a Python dict.
        """
        index = self.prices["date"]
        ret_adj_np = self.ret_adj.select(self.assets).to_numpy()
        tensor = _ewm_corr_numpy(
            ret_adj_np,
            com=self.cfg.corr,
            min_periods=self.cfg.corr,
            min_corr_denom=self.cfg.min_corr_denom,
        )
        return {index[t]: tensor[t] for t in range(len(index))}

    @property
    def cor_tensor(self) -> np.ndarray:
        """Return all correlation matrices stacked as a 3-D tensor.

        Converts the per-timestamp correlation dict (see :py:attr:`cor`) into a
        single contiguous NumPy array so that the full history can be saved to
        a flat ``.npy`` file with :func:`numpy.save` and reloaded with
        :func:`numpy.load`.

        Returns:
            np.ndarray: Array of shape ``(T, N, N)`` where *T* is the number of
            timestamps and *N* the number of assets.  ``tensor[t]`` is the
            correlation matrix for the *t*-th date (same ordering as
            ``self.prices["date"]``).

        Examples:
            >>> import tempfile, pathlib
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math.optimizer import BasanosConfig, BasanosEngine
            >>> dates = pl.Series("date", list(range(100)))
            >>> rng0 = np.random.default_rng(0).lognormal(size=100)
            >>> rng1 = np.random.default_rng(1).lognormal(size=100)
            >>> prices = pl.DataFrame({"date": dates, "A": rng0, "B": rng1})
            >>> rng2 = np.random.default_rng(2).normal(size=100)
            >>> rng3 = np.random.default_rng(3).normal(size=100)
            >>> mu = pl.DataFrame({"date": dates, "A": rng2, "B": rng3})
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> tensor = engine.cor_tensor
            >>> with tempfile.TemporaryDirectory() as td:
            ...     path = pathlib.Path(td) / "cor.npy"
            ...     np.save(path, tensor)
            ...     loaded = np.load(path)
            >>> np.testing.assert_array_equal(tensor, loaded)
        """
        return np.stack(list(self.cor.values()), axis=0)

    # ------------------------------------------------------------------
    # Internal solve helpers — inherited from _SolveMixin
    # ------------------------------------------------------------------
    # (_compute_mask, _check_signal, _scale_to_cash, _row_early_check,
    #  _denom_guard_yield, _compute_position, _replay_positions,
    #  _iter_matrices, _iter_solve, warmup_state)
    # Implementations live in _engine_solve.py; patch targets remain in that
    # module's namespace, e.g. ``patch("basanos.math._engine_solve.solve")``.

    # ------------------------------------------------------------------
    # Position properties
    # ------------------------------------------------------------------

    @property
    def cash_position(self) -> pl.DataFrame:
        r"""Optimize correlation-aware risk positions for each timestamp.

        Supports two covariance modes controlled by ``cfg.covariance_config``:

        * :class:`EwmaShrinkConfig` (default): Computes EWMA correlations, applies
          linear shrinkage toward the identity, and solves a normalised linear
          system :math:`C\,x = \mu` per timestamp via Cholesky / LU.

        * :class:`SlidingWindowConfig`: At each timestamp uses the
          ``cfg.covariance_config.window`` most recent vol-adjusted returns to fit a
          rank-``cfg.covariance_config.n_factors`` factor model via truncated SVD and
          solves the system via the Woodbury identity at :math:`O(k^3 + kn)` rather
          than :math:`O(n^3)` per step.

        Non-finite or ill-posed cases yield zero positions for safety.

        Returns:
            pl.DataFrame: DataFrame with columns ['date'] + asset names containing
            the per-timestamp cash positions (risk divided by EWMA volatility).

        Performance:
            For ``ewma_shrink``: dominant cost is ``self.cor`` (O(T·N²) time,
            O(T·N²) memory — see :func:`ewm_corr`).  The per-timestamp
            linear solve adds O(N³) per row.

            For ``sliding_window``: O(T·W·N·k) for sliding SVDs plus
            O(T·(k³ + kN)) for Woodbury solves.  Memory is O(W·N) per step,
            independent of T.
        """
        assets = self.assets

        # Compute risk positions row-by-row using _replay_positions.
        prices_num = self.prices.select(assets).to_numpy()

        risk_pos_np = np.full_like(prices_num, fill_value=np.nan, dtype=float)
        cash_pos_np = np.full_like(prices_num, fill_value=np.nan, dtype=float)
        vola_np = self.vola.select(assets).to_numpy()

        self._replay_positions(risk_pos_np, cash_pos_np, vola_np)

        # Build Polars DataFrame for cash positions (numeric columns only)
        cash_position = self.prices.with_columns(
            [(pl.lit(cash_pos_np[:, i]).alias(asset)) for i, asset in enumerate(assets)]
        )

        return cash_position

    @property
    def position_status(self) -> pl.DataFrame:
        """Per-timestamp reason code explaining each :attr:`cash_position` row.

        Labels every row with exactly one of four :class:`~basanos.math.SolveStatus`
        codes (which compare equal to their string equivalents):

        * ``'warmup'``: Insufficient history for the sliding-window
          covariance mode (``i + 1 < cfg.covariance_config.window``).
          Positions are ``NaN`` for all assets at this timestamp.
        * ``'zero_signal'``: The expected-return vector ``mu`` was
          all-zeros (or all-NaN) at this timestamp; the optimizer
          short-circuited and returned zero positions without solving.
        * ``'degenerate'``: The normalisation denominator was non-finite
          or below ``cfg.denom_tol``, the Cholesky / Woodbury solve
          failed, or no asset had a finite price; positions were zeroed
          for safety.
        * ``'valid'``: The linear system was solved successfully and
          positions are non-trivially non-zero.

        The codes map one-to-one onto the three NaN / zero cases
        described in the issue and allow downstream consumers (backtests,
        risk monitors) to distinguish data gaps from signal silence from
        numerical ill-conditioning without re-inspecting ``mu`` or the
        engine configuration.

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'status': ...}``
            with one row per timestamp.  The ``status`` column has
            ``Polars`` dtype ``String``.
        """
        statuses = [status for _i, _t, _mask, _pos, status in self._iter_solve()]
        return pl.DataFrame({"date": self.prices["date"], "status": pl.Series(statuses, dtype=pl.String)})

    @property
    def risk_position(self) -> pl.DataFrame:
        """Risk positions (before EWMA-volatility scaling) at each timestamp.

        Derives the un-volatility-scaled position by multiplying the cash
        position by the per-asset EWMA volatility.  Equivalently, this is
        the quantity solved by the correlation-adjusted linear system before
        dividing by ``vola``.

        Relationship to other properties::

            cash_position = risk_position / vola
            risk_position = cash_position * vola

        Returns:
            pl.DataFrame: DataFrame with columns ``['date'] + assets`` where
            each value is ``cash_position_i * vola_i`` at the given timestamp.
        """
        assets = self.assets
        cp_np = self.cash_position.select(assets).to_numpy()
        vola_np = self.vola.select(assets).to_numpy()
        with np.errstate(invalid="ignore"):
            risk_pos = cp_np * vola_np
        return self.prices.with_columns([pl.lit(risk_pos[:, i]).alias(asset) for i, asset in enumerate(assets)])

    @property
    def position_leverage(self) -> pl.DataFrame:
        """L1 norm of cash positions (gross leverage) at each timestamp.

        Sums the absolute values of all asset cash positions at each row.
        NaN positions are treated as zero (they contribute nothing to gross
        leverage).

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'leverage': ...}``
            where ``leverage`` is the L1 norm of the cash-position vector.
        """
        assets = self.assets
        cp_np = self.cash_position.select(assets).to_numpy()
        leverage = np.nansum(np.abs(cp_np), axis=1)
        return pl.DataFrame({"date": self.prices["date"], "leverage": pl.Series(leverage, dtype=pl.Float64)})

    # ------------------------------------------------------------------
    # Portfolio and performance
    # ------------------------------------------------------------------

    @property
    def portfolio(self) -> Portfolio:
        """Construct a Portfolio from the optimized cash positions.

        Converts the computed cash positions into a Portfolio using the
        configured AUM.  The ``cost_per_unit`` from :attr:`cfg` is forwarded
        so that :attr:`~basanos.analytics.Portfolio.net_cost_nav` and
        :attr:`~basanos.analytics.Portfolio.position_delta_costs` work out
        of the box without any further configuration.

        Returns:
            Portfolio: Instance built from cash positions with AUM scaling.
        """
        cp = self.cash_position
        assets = [c for c in cp.columns if c != "date" and cp[c].dtype.is_numeric()]
        scaled = cp.with_columns(pl.col(a) * self.cfg.position_scale for a in assets)
        return Portfolio.from_cash_position(self.prices, scaled, aum=self.cfg.aum, cost_per_unit=self.cfg.cost_per_unit)

    def sharpe_at_shrink(self, shrink: float) -> float:
        r"""Return the annualised portfolio Sharpe ratio for the given shrinkage weight.

        Constructs a new :class:`BasanosEngine` with all parameters identical to
        ``self`` except that ``cfg.shrink`` is replaced by ``shrink``, then
        returns the annualised Sharpe ratio of the resulting portfolio.

        This is the canonical single-argument callable required by the benchmarks
        specification: ``f(λ) → Sharpe``.  Use it to sweep λ across ``[0, 1]``
        and measure whether correlation adjustment adds value over the
        signal-proportional baseline (λ = 0) or the unregularised limit (λ = 1).

        Corner cases:
            * **λ = 0** — the shrunk matrix equals the identity, so the
              optimiser treats all assets as uncorrelated and positions are
              purely signal-proportional (no correlation adjustment).
            * **λ = 1** — the raw EWMA correlation matrix is used without
              shrinkage.

        Args:
            shrink: Retention weight λ ∈ [0, 1].  See
                :attr:`BasanosConfig.shrink` for full documentation.

        Returns:
            Annualised Sharpe ratio of the portfolio returns as a ``float``.
            Returns ``float("nan")`` when the Sharpe ratio cannot be computed
            (e.g. zero-variance returns).

        Raises:
            ValidationError: When ``shrink`` is outside [0, 1] (delegated to
                :class:`BasanosConfig` field validation).

        Examples:
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math.optimizer import BasanosConfig, BasanosEngine
            >>> dates = pl.Series("date", list(range(200)))
            >>> rng = np.random.default_rng(0)
            >>> prices = pl.DataFrame({"date": dates, "A": rng.lognormal(size=200), "B": rng.lognormal(size=200)})
            >>> mu = pl.DataFrame({"date": dates, "A": rng.normal(size=200), "B": rng.normal(size=200)})
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> s = engine.sharpe_at_shrink(0.5)
            >>> isinstance(s, float)
            True
        """
        new_cfg = self.cfg.replace(shrink=shrink)
        engine = BasanosEngine(prices=self.prices, mu=self.mu, cfg=new_cfg)
        return float(engine.portfolio.stats.sharpe().get("returns") or float("nan"))

    def sharpe_at_window_factors(self, window: int, n_factors: int) -> float:
        r"""Return the annualised portfolio Sharpe ratio for the given sliding-window parameters.

        Constructs a new :class:`BasanosEngine` with ``covariance_mode`` set to
        ``"sliding_window"`` and the supplied ``window`` / ``n_factors``, keeping
        all other configuration identical to ``self``.

        Use this method to sweep ``(W, k)`` and compare the sliding-window
        estimator against the EWMA baseline (via :meth:`sharpe_at_shrink`).

        Args:
            window: Rolling window length :math:`W \geq 1`.
                Rule of thumb: :math:`W \geq 2 \cdot n_{\text{assets}}`.
            n_factors: Number of latent factors :math:`k \geq 1`.

        Returns:
            Annualised Sharpe ratio of the portfolio returns as a ``float``.
            Returns ``float("nan")`` when the Sharpe ratio cannot be computed
            (e.g. not enough history to fill the first window).

        Raises:
            ValidationError: When ``window`` or ``n_factors`` fail field
                constraints (delegated to :class:`BasanosConfig`).

        Examples:
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math.optimizer import BasanosConfig, BasanosEngine
            >>> dates = pl.Series("date", list(range(200)))
            >>> rng = np.random.default_rng(0)
            >>> prices = pl.DataFrame({"date": dates, "A": rng.lognormal(size=200), "B": rng.lognormal(size=200)})
            >>> mu = pl.DataFrame({"date": dates, "A": rng.normal(size=200), "B": rng.normal(size=200)})
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> s = engine.sharpe_at_window_factors(window=40, n_factors=2)
            >>> isinstance(s, float)
            True
        """
        new_cfg = self.cfg.replace(
            covariance_config=SlidingWindowConfig(window=window, n_factors=n_factors),
        )
        engine = BasanosEngine(prices=self.prices, mu=self.mu, cfg=new_cfg)
        return float(engine.portfolio.stats.sharpe().get("returns") or float("nan"))

    @property
    def naive_sharpe(self) -> float:
        r"""Sharpe ratio of the naïve equal-weight signal (μ = 1 for every asset/timestamp).

        Replaces the expected-return signal ``mu`` with a constant matrix of
        ones, then runs the optimiser with the current configuration and returns
        the annualised Sharpe ratio of the resulting portfolio.

        This provides the baseline answer to *"does the signal add value?"*:
        a real signal should produce a higher Sharpe than the naïve benchmark.
        Combined with :meth:`sharpe_at_shrink`, this yields a three-way
        comparison:

        +--------------------+----------------------------------------------+
        | Benchmark          | What it measures                             |
        +====================+==============================================+
        | ``naive_sharpe``   | No signal skill; pure correlation routing   |
        +--------------------+----------------------------------------------+
        | ``sharpe_at_shrink(0.0)`` | Signal skill, no correlation adj.  |
        +--------------------+----------------------------------------------+
        | ``sharpe_at_shrink(cfg.shrink)`` | Signal + correlation adj.  |
        +--------------------+----------------------------------------------+

        Returns:
            Annualised Sharpe ratio of the equal-weight portfolio as a ``float``.
            Returns ``float("nan")`` when the Sharpe ratio cannot be computed.

        Examples:
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math.optimizer import BasanosConfig, BasanosEngine
            >>> dates = pl.Series("date", list(range(200)))
            >>> rng = np.random.default_rng(0)
            >>> prices = pl.DataFrame({"date": dates, "A": rng.lognormal(size=200), "B": rng.lognormal(size=200)})
            >>> mu = pl.DataFrame({"date": dates, "A": rng.normal(size=200), "B": rng.normal(size=200)})
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> s = engine.naive_sharpe
            >>> isinstance(s, float)
            True
        """
        naive_mu = self.mu.with_columns(pl.lit(1.0).alias(asset) for asset in self.assets)
        engine = BasanosEngine(prices=self.prices, mu=naive_mu, cfg=self.cfg)
        return float(engine.portfolio.stats.sharpe().get("returns") or float("nan"))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def config_report(self) -> "ConfigReport":
        """Return a :class:`~basanos.math._config_report.ConfigReport` facade for this engine.

        Returns a :class:`~basanos.math._config_report.ConfigReport` that
        includes the full **lambda-sweep chart** — an interactive plot of the
        annualised Sharpe ratio as :attr:`~BasanosConfig.shrink` (λ) is swept
        across [0, 1] — in addition to the parameter table, shrinkage-guidance
        table, and theory section available from
        :attr:`BasanosConfig.report`.

        Returns:
            basanos.math._config_report.ConfigReport: Report facade with
            ``to_html()`` and ``save()`` methods.

        Examples:
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math.optimizer import BasanosConfig, BasanosEngine
            >>> dates = pl.Series("date", list(range(200)))
            >>> rng = np.random.default_rng(0)
            >>> prices = pl.DataFrame({"date": dates, "A": rng.lognormal(size=200), "B": rng.lognormal(size=200)})
            >>> mu = pl.DataFrame({"date": dates, "A": rng.normal(size=200), "B": rng.normal(size=200)})
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> report = engine.config_report
            >>> html = report.to_html()
            >>> "Lambda" in html
            True
        """
        from ._config_report import ConfigReport

        return ConfigReport(config=self.cfg, engine=self)

    # ------------------------------------------------------------------
    # Matrix diagnostics — inherited from _DiagnosticsMixin
    # ------------------------------------------------------------------
    # (condition_number, effective_rank, solver_residual, signal_utilisation)
    # Implementations live in _engine_diagnostics.py; patch targets remain in
    # that module's namespace, e.g.
    # ``patch("basanos.math._engine_diagnostics.solve")``.

    # ------------------------------------------------------------------
    # Signal evaluation — inherited from _SignalEvaluatorMixin
    # ------------------------------------------------------------------
    # (_ic_series, ic, rank_ic, ic_mean, ic_std, icir,
    #  rank_ic_mean, rank_ic_std)
    # Implementations live in _engine_ic.py.
