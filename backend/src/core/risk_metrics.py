"""Pure risk metrics.

Used by the Phase 4 risk service after fills update the equity curve.
This module is **pure** so the metrics can be tested exhaustively in
isolation and so §5 invariant 4 (no future data in metrics) can be
enforced structurally by passing only past data in.

Conventions:

  * Returns are simple period returns (e.g. day-over-day equity change).
  * Sharpe is annualised by ``sqrt(252)`` trading days. Risk-free rate
    is 0 in V1 — these are practice statistics, not regulatory reports.
  * ``win_rate`` counts strictly positive trade P&Ls as wins.
  * ``average_loss`` returns the **signed** mean of negative P&Ls so
    callers compose ``win_rate * avg_win + (1 - win_rate) * avg_loss``
    without re-introducing a sign.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

TRADING_DAYS_PER_YEAR = 252


# --- Sharpe ----------------------------------------------------------------


def sharpe(returns: Sequence[float]) -> float | None:
    """Annualised Sharpe ratio. Returns None when undefined.

    Undefined when fewer than two observations exist or when stddev is
    zero (identical returns) — neither yields a meaningful ratio.
    """
    n = len(returns)
    if n < 2:
        return None

    mean = sum(returns) / n
    # Sample (n-1) variance — these are observations, not the population.
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if variance == 0:
        return None

    stddev = math.sqrt(variance)
    return (mean / stddev) * math.sqrt(TRADING_DAYS_PER_YEAR)


# --- Drawdown -------------------------------------------------------------


def drawdown_series(equity: Sequence[float]) -> list[float]:
    """Per-bar drawdown from the running peak, as a fraction ``[0, 1]``.

    A drawdown of ``0.30`` means equity is currently 30% below its
    historical peak. Returns an empty list for empty input.
    """
    if not equity:
        return []

    series: list[float] = []
    peak = equity[0]
    for value in equity:
        if value > peak:
            peak = value
        dd = 0.0 if peak == 0 else (peak - value) / peak
        series.append(dd)
    return series


def max_drawdown(equity: Sequence[float]) -> float:
    """Maximum drawdown observed across the equity curve."""
    series = drawdown_series(equity)
    return max(series) if series else 0.0


# --- Trade statistics -----------------------------------------------------


def win_rate(trade_pnls: Sequence[float]) -> float | None:
    """Fraction of trades with strictly positive P&L. None if no trades."""
    if not trade_pnls:
        return None
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls)


def average_win(trade_pnls: Sequence[float]) -> float | None:
    """Mean of strictly positive P&Ls. None if no winning trades."""
    wins = [p for p in trade_pnls if p > 0]
    if not wins:
        return None
    return sum(wins) / len(wins)


def average_loss(trade_pnls: Sequence[float]) -> float | None:
    """Mean of strictly negative P&Ls (returned as a signed, negative number).

    Returning the signed mean keeps the expectancy formula sign-clean:
    ``E = wr * aw + (1 - wr) * al`` with no explicit subtraction.
    """
    losses = [p for p in trade_pnls if p < 0]
    if not losses:
        return None
    return sum(losses) / len(losses)


def expectancy(trade_pnls: Sequence[float]) -> float | None:
    """Expected P&L per trade.

    ``E = win_rate * avg_win + (1 - win_rate) * avg_loss``

    A pure-wins or pure-losses series substitutes 0 for the missing
    side, so the formula collapses cleanly. ``None`` when no trades.
    """
    if not trade_pnls:
        return None
    wr = win_rate(trade_pnls)
    aw = average_win(trade_pnls) or 0.0
    al = average_loss(trade_pnls) or 0.0
    # wr cannot be None here because trade_pnls is non-empty.
    assert wr is not None
    return wr * aw + (1 - wr) * al
