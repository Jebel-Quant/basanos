# Streaming API Guide

`BasanosStream` provides an incremental, one-row-at-a-time interface for
computing optimised positions in real-time without re-running the full batch
engine.  It is designed for live trading systems, paper-trading loops, and
research workflows where data arrives sequentially.

For the complete API reference see [Streaming API](api/stream.md).

---

## When to use BasanosStream vs BasanosEngine

| | `BasanosEngine` | `BasanosStream` |
|---|---|---|
| **Use case** | Batch backtests, research | Live feeds, sequential updates |
| **Input** | Full price + signal history | One row at a time |
| **State** | Recomputed from scratch | Maintained incrementally |
| **Memory** | O(T·N²) peak | O(N²) — only current state |
| **Per-step cost** | — | O(N²) correlation update + O(N³) or O(k³+kN) solve |

---

## Quick start

### Step 1 — warm up from historical data

```python
import numpy as np
import polars as pl
from basanos.math import BasanosConfig, BasanosStream

cfg = BasanosConfig(vola=16, corr=32, clip=3.5, shrink=0.5, aum=1e6)

# Use a history of prices + signals to initialise state
# prices and mu are pl.DataFrames with a "date" column and one column per asset
stream = BasanosStream.from_warmup(prices=prices, mu=mu, cfg=cfg)
```

### Step 2 — advance one row at a time

```python
# new_prices and new_mu are single-row pl.DataFrames or dicts
result = stream.step(new_prices=new_prices, new_mu=new_mu)

# result.status is one of: "warmup", "zero_signal", "degenerate", "valid"
if result.status == "valid":
    cash_pos = result.cash_position  # dict[str, float] — one value per asset
```

---

## StepResult

Each call to `stream.step()` returns a `StepResult` frozen dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `status` | `str` | `"warmup"` / `"zero_signal"` / `"degenerate"` / `"valid"` |
| `cash_position` | `dict[str, float] \| None` | Cash positions, or `None` when status is not `"valid"` |
| `risk_position` | `dict[str, float] \| None` | Risk positions before vol-scaling, or `None` |

---

## Status codes

| Status | Meaning |
|--------|---------|
| `warmup` | Not enough rows have been processed yet (`step_count < cfg.corr`); positions are zero |
| `zero_signal` | All signal values are zero or below `cfg.denom_tol`; no solve performed |
| `degenerate` | Correlation matrix is singular or ill-conditioned; positions are zeroed for safety |
| `valid` | Normal operation; `cash_position` contains usable values |

---

## Immutability

`BasanosStream` instances are **immutable** — `stream.step()` returns a *new*
`BasanosStream` with updated internal state rather than mutating the original.
This makes it safe to checkpoint state and replay from any point:

```python
# Save a checkpoint before processing a risky period
checkpoint = stream

# Process live rows
for row_prices, row_mu in live_feed:
    result = stream.step(row_prices, row_mu)
    stream = result.next_stream  # advance to the updated stream

# Roll back to checkpoint if needed
stream = checkpoint
```

---

## Performance notes

- **Per-step cost** is O(N²) for the IIR correlation update and O(N³) (EWMA
  mode) or O(k³ + kN) (sliding-window mode) for the linear solve.
- State is a fixed-size object regardless of how many rows have been processed —
  memory does not grow over time.
- For large universes (N > 200) prefer `SlidingWindowConfig` to reduce the
  per-step solve cost; see [Factor Models](factor-models.md) and
  [Concepts — Choosing Between Modes](concepts.md#choosing-between-modes).
