"""Tests for pure risk limits — Phase 4, Module 16.

``check_risk_limits`` evaluates a proposed trade against a snapshot of
the account and a set of limits, returning a list of breach reasons
(empty = pass). Pure function — no DB, no side effects.

The four limits enforced (from §4.5):

  * max_drawdown_pct           current_drawdown >= limit → BREACH
  * max_risk_per_trade_pct     |trade_risk| / equity > limit → BREACH
  * max_open_positions         current open + 1 > limit → BREACH
  * max_exposure_pct           (total exposure incl. new) / equity > limit → BREACH

``trade_risk`` is supplied by the caller — when a stop_loss is set,
it's ``qty * abs(price - stop_loss)``; otherwise the full notional
``qty * price`` (an order without a stop puts the whole position at
risk in principle).
"""

import pytest

from src.core.risk_limits import (
    RiskCheckInputs,
    RiskLimits,
    check_risk_limits,
)


def _limits(**overrides) -> RiskLimits:
    base = dict(
        max_drawdown_pct=10.0,
        max_risk_per_trade_pct=2.0,
        max_open_positions=3,
        max_exposure_pct=100.0,
    )
    base.update(overrides)
    return RiskLimits(**base)


def _inputs(**overrides) -> RiskCheckInputs:
    base = dict(
        equity=100_000.0,
        current_drawdown_pct=0.0,
        open_position_count=0,
        current_exposure=0.0,
        proposed_notional=10_000.0,
        proposed_risk=1_000.0,  # 1% of equity → under 2% default limit
    )
    base.update(overrides)
    return RiskCheckInputs(**base)


# --- All clear -------------------------------------------------------------


def test_clean_trade_passes() -> None:
    result = check_risk_limits(_inputs(), _limits())
    assert result.allowed is True
    assert result.breaches == []


# --- Per-trade risk -------------------------------------------------------


def test_blocks_when_trade_risk_exceeds_per_trade_limit() -> None:
    """Trade risking 3% of equity vs. a 2% limit → BREACH."""
    result = check_risk_limits(
        _inputs(proposed_risk=3_000.0), _limits(max_risk_per_trade_pct=2.0)
    )
    assert result.allowed is False
    assert any(b.code == "MAX_RISK_PER_TRADE" for b in result.breaches)


def test_per_trade_limit_at_exact_threshold_passes() -> None:
    """Exactly the limit (2.0%) is allowed; only > is a breach."""
    result = check_risk_limits(
        _inputs(proposed_risk=2_000.0), _limits(max_risk_per_trade_pct=2.0)
    )
    assert result.allowed is True


# --- Drawdown -------------------------------------------------------------


def test_blocks_when_drawdown_at_or_above_limit() -> None:
    result = check_risk_limits(
        _inputs(current_drawdown_pct=10.0), _limits(max_drawdown_pct=10.0)
    )
    assert result.allowed is False
    assert any(b.code == "MAX_DRAWDOWN" for b in result.breaches)


def test_drawdown_below_limit_passes() -> None:
    result = check_risk_limits(
        _inputs(current_drawdown_pct=9.99), _limits(max_drawdown_pct=10.0)
    )
    assert result.allowed is True


# --- Open positions ------------------------------------------------------


def test_blocks_when_new_position_exceeds_max_open() -> None:
    """Already at the limit — adding another would breach."""
    result = check_risk_limits(
        _inputs(open_position_count=3), _limits(max_open_positions=3)
    )
    assert result.allowed is False
    assert any(b.code == "MAX_OPEN_POSITIONS" for b in result.breaches)


def test_within_position_count_passes() -> None:
    result = check_risk_limits(
        _inputs(open_position_count=2), _limits(max_open_positions=3)
    )
    assert result.allowed is True


# --- Exposure -----------------------------------------------------------


def test_blocks_when_exposure_exceeds_limit() -> None:
    """Existing 90k + new 20k = 110k on 100k equity = 110% > 100%."""
    result = check_risk_limits(
        _inputs(current_exposure=90_000.0, proposed_notional=20_000.0),
        _limits(max_exposure_pct=100.0),
    )
    assert result.allowed is False
    assert any(b.code == "MAX_EXPOSURE" for b in result.breaches)


def test_exposure_at_exact_threshold_passes() -> None:
    """Exactly 100% is allowed; only > is a breach."""
    result = check_risk_limits(
        _inputs(current_exposure=90_000.0, proposed_notional=10_000.0),
        _limits(max_exposure_pct=100.0),
    )
    assert result.allowed is True


# --- Multiple breaches reported together -------------------------------


def test_multiple_breaches_all_reported() -> None:
    """Don't short-circuit — caller gets the full picture."""
    result = check_risk_limits(
        _inputs(
            current_drawdown_pct=15.0,
            proposed_risk=5_000.0,
            open_position_count=10,
        ),
        _limits(),
    )
    assert result.allowed is False
    codes = {b.code for b in result.breaches}
    assert {"MAX_DRAWDOWN", "MAX_RISK_PER_TRADE", "MAX_OPEN_POSITIONS"} <= codes


def test_breach_carries_diagnostic_values() -> None:
    """Each breach includes the offending and limit values, for UI display."""
    result = check_risk_limits(
        _inputs(proposed_risk=3_500.0), _limits(max_risk_per_trade_pct=2.0)
    )
    breach = next(b for b in result.breaches if b.code == "MAX_RISK_PER_TRADE")
    assert breach.actual == pytest.approx(3.5)
    assert breach.limit == pytest.approx(2.0)


# --- Degenerate inputs -------------------------------------------------


def test_zero_equity_treated_as_breach_not_division_error() -> None:
    """Equity at zero should never divide-by-zero — it's a hard breach."""
    result = check_risk_limits(_inputs(equity=0.0), _limits())
    assert result.allowed is False
    assert any(b.code == "ZERO_EQUITY" for b in result.breaches)


def test_negative_equity_blocks_all_trades() -> None:
    result = check_risk_limits(_inputs(equity=-100.0), _limits())
    assert result.allowed is False
