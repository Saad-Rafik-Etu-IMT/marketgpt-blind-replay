"""Tests for pure risk metrics — Phase 4, Module 15.

All functions are pure. The metrics:

  * sharpe(returns)                 — None if < 2 returns or zero stddev
  * max_drawdown(equity_curve)      — fraction (0.0..1.0)
  * win_rate(trade_pnls)            — None if no trades
  * average_win / average_loss      — None if no wins / losses
  * expectancy(trade_pnls)          — None if no trades

Sharpe is annualised by sqrt(252) trading days. Risk-free rate is 0 in
V1 — practice statistics, not regulatory reporting.
"""

import math

import pytest

from src.core.risk_metrics import (
    average_loss,
    average_win,
    drawdown_series,
    expectancy,
    max_drawdown,
    sharpe,
    win_rate,
)

# --- Sharpe ----------------------------------------------------------------


def test_sharpe_none_for_empty_returns() -> None:
    assert sharpe([]) is None


def test_sharpe_none_for_single_return() -> None:
    assert sharpe([0.01]) is None


def test_sharpe_none_when_stddev_zero() -> None:
    """Identical returns have zero stddev — Sharpe is undefined, not infinite."""
    assert sharpe([0.01, 0.01, 0.01]) is None


def test_sharpe_positive_for_positive_mean() -> None:
    s = sharpe([0.01, -0.005, 0.02, 0.005, 0.01])
    assert s is not None
    assert s > 0


def test_sharpe_negative_for_negative_mean() -> None:
    s = sharpe([-0.01, 0.005, -0.02, -0.005, -0.01])
    assert s is not None
    assert s < 0


def test_sharpe_is_annualised() -> None:
    """The reported number multiplies the daily ratio by sqrt(252)."""
    returns = [0.01, -0.005, 0.02, 0.005, 0.01]
    annual = sharpe(returns)
    assert annual is not None

    # Recompute manually using sample stddev.
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    stddev = math.sqrt(var)
    expected = (mean / stddev) * math.sqrt(252)
    assert annual == pytest.approx(expected, rel=1e-9)


# --- Drawdown -------------------------------------------------------------


def test_max_drawdown_zero_on_monotonic_increase() -> None:
    equity = [100.0, 110.0, 120.0, 130.0]
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_simple_drop() -> None:
    """Peak 100 → trough 80 → DD = 0.20."""
    equity = [100.0, 90.0, 80.0, 95.0]
    assert max_drawdown(equity) == pytest.approx(0.20)


def test_max_drawdown_uses_running_peak() -> None:
    """A later, deeper drop after recovery becomes the max DD."""
    equity = [100.0, 80.0, 120.0, 60.0, 90.0]
    # peak 120 → trough 60 → DD 0.50
    assert max_drawdown(equity) == pytest.approx(0.5)


def test_max_drawdown_empty_is_zero() -> None:
    assert max_drawdown([]) == 0.0


def test_max_drawdown_single_value_is_zero() -> None:
    assert max_drawdown([100.0]) == 0.0


def test_drawdown_series_matches_definition() -> None:
    """drawdown_series[i] = (peak_so_far - equity[i]) / peak_so_far."""
    equity = [100.0, 90.0, 80.0, 95.0]
    series = drawdown_series(equity)
    assert series == pytest.approx([0.0, 0.10, 0.20, 0.05])


# --- Win rate, average win/loss ------------------------------------------


def test_win_rate_none_when_no_trades() -> None:
    assert win_rate([]) is None


def test_win_rate_basic() -> None:
    # 3 wins, 2 losses → 0.6
    assert win_rate([10.0, -5.0, 20.0, -3.0, 15.0]) == pytest.approx(0.6)


def test_win_rate_zero_pnl_is_loss() -> None:
    """A zero P&L trade does not count as a win (>0 required)."""
    assert win_rate([0.0, 10.0]) == pytest.approx(0.5)


def test_average_win_none_when_no_wins() -> None:
    assert average_win([-1.0, -2.0]) is None


def test_average_loss_none_when_no_losses() -> None:
    assert average_loss([1.0, 2.0]) is None


def test_average_win_is_mean_of_positive_pnls() -> None:
    assert average_win([10.0, -5.0, 20.0]) == pytest.approx(15.0)


def test_average_loss_is_signed_mean_of_negative_pnls() -> None:
    """average_loss returns the signed (negative) mean — not absolute."""
    assert average_loss([10.0, -5.0, -15.0]) == pytest.approx(-10.0)


# --- Expectancy -----------------------------------------------------------


def test_expectancy_none_when_no_trades() -> None:
    assert expectancy([]) is None


def test_expectancy_formula() -> None:
    """expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss.

    Note: avg_loss is signed (negative), so the sign comes naturally.
    """
    pnls = [10.0, -5.0, 20.0, -3.0, 15.0]
    wr = 3 / 5
    aw = (10 + 20 + 15) / 3
    al = (-5 + -3) / 2
    expected = wr * aw + (1 - wr) * al
    assert expectancy(pnls) == pytest.approx(expected)


def test_expectancy_all_wins_equals_average_win() -> None:
    """If every trade is a win, expectancy collapses to average_win."""
    assert expectancy([10.0, 20.0, 5.0]) == pytest.approx(35 / 3)
