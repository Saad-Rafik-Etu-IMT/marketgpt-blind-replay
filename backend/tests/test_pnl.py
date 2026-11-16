"""Tests for the pure P&L module — Phase 4, Module 14.

Position-tracking conventions (resolved as self-elected decisions; see
PROJECT_STATUS.md):

  * A position is one-sided (long | short | flat).
  * Adding to a position uses weighted-average cost basis.
  * Reducing a position realises P&L on the closed quantity.
  * A signed-quantity overshoot fully closes the old side and opens
    the new side with the remainder at the trade's price.

Everything in this module is pure — no DB, no time.
"""

import pytest

from src.core.pnl import (
    PositionState,
    apply_trade,
    compute_equity,
    unrealized_pnl,
)


def _flat() -> PositionState:
    """A flat position — no exposure, no P&L."""
    return PositionState(
        instrument="SPY",
        side="flat",
        quantity=0,
        avg_entry_price=0.0,
        realized_pnl=0.0,
    )


# --- Opening positions -----------------------------------------------------


def test_buy_into_flat_opens_long() -> None:
    after = apply_trade(_flat(), side="buy", quantity=10, price=100.0)
    assert after.side == "long"
    assert after.quantity == 10
    assert after.avg_entry_price == 100.0
    assert after.realized_pnl == 0.0


def test_sell_into_flat_opens_short() -> None:
    after = apply_trade(_flat(), side="sell", quantity=10, price=100.0)
    assert after.side == "short"
    assert after.quantity == 10
    assert after.avg_entry_price == 100.0
    assert after.realized_pnl == 0.0


# --- Adding to a position --------------------------------------------------


def test_buy_into_long_uses_weighted_average_cost() -> None:
    pos = apply_trade(_flat(), "buy", 10, 100.0)
    pos = apply_trade(pos, "buy", 10, 110.0)
    assert pos.side == "long"
    assert pos.quantity == 20
    assert pos.avg_entry_price == 105.0
    assert pos.realized_pnl == 0.0


def test_sell_into_short_uses_weighted_average_cost() -> None:
    pos = apply_trade(_flat(), "sell", 10, 100.0)
    pos = apply_trade(pos, "sell", 30, 110.0)
    # (10 * 100 + 30 * 110) / 40 = 107.5
    assert pos.side == "short"
    assert pos.quantity == 40
    assert pos.avg_entry_price == 107.5


# --- Reducing a position ---------------------------------------------------


def test_partial_sell_from_long_realises_pnl() -> None:
    pos = apply_trade(_flat(), "buy", 10, 100.0)
    pos = apply_trade(pos, "sell", 4, 110.0)
    assert pos.side == "long"
    assert pos.quantity == 6
    assert pos.avg_entry_price == 100.0  # unchanged on reduction
    assert pos.realized_pnl == pytest.approx(40.0)  # 4 * (110 - 100)


def test_partial_buy_from_short_realises_pnl() -> None:
    pos = apply_trade(_flat(), "sell", 10, 100.0)
    pos = apply_trade(pos, "buy", 4, 95.0)
    assert pos.side == "short"
    assert pos.quantity == 6
    assert pos.avg_entry_price == 100.0
    assert pos.realized_pnl == pytest.approx(20.0)  # 4 * (100 - 95)


def test_exact_close_returns_flat() -> None:
    pos = apply_trade(_flat(), "buy", 10, 100.0)
    pos = apply_trade(pos, "sell", 10, 110.0)
    assert pos.side == "flat"
    assert pos.quantity == 0
    assert pos.avg_entry_price == 0.0
    assert pos.realized_pnl == pytest.approx(100.0)


def test_realized_pnl_accumulates_across_trades() -> None:
    pos = apply_trade(_flat(), "buy", 10, 100.0)
    pos = apply_trade(pos, "sell", 5, 110.0)  # +50
    pos = apply_trade(pos, "sell", 5, 90.0)  # -50
    assert pos.side == "flat"
    assert pos.realized_pnl == pytest.approx(0.0)


# --- Flipping the side -----------------------------------------------------


def test_sell_overshoots_long_and_flips_to_short() -> None:
    pos = apply_trade(_flat(), "buy", 10, 100.0)
    pos = apply_trade(pos, "sell", 15, 110.0)
    # Closes 10 long at 110 → +100 realized.
    # Opens 5 short at 110.
    assert pos.side == "short"
    assert pos.quantity == 5
    assert pos.avg_entry_price == 110.0
    assert pos.realized_pnl == pytest.approx(100.0)


def test_buy_overshoots_short_and_flips_to_long() -> None:
    pos = apply_trade(_flat(), "sell", 10, 100.0)
    pos = apply_trade(pos, "buy", 15, 95.0)
    # Closes 10 short at 95 → +50 realized.
    # Opens 5 long at 95.
    assert pos.side == "long"
    assert pos.quantity == 5
    assert pos.avg_entry_price == 95.0
    assert pos.realized_pnl == pytest.approx(50.0)


# --- Unrealized P&L --------------------------------------------------------


def test_unrealized_long():
    pos = apply_trade(_flat(), "buy", 10, 100.0)
    assert unrealized_pnl(pos, mark_price=110.0) == pytest.approx(100.0)


def test_unrealized_short() -> None:
    pos = apply_trade(_flat(), "sell", 10, 100.0)
    assert unrealized_pnl(pos, mark_price=95.0) == pytest.approx(50.0)


def test_unrealized_flat_is_zero() -> None:
    assert unrealized_pnl(_flat(), mark_price=100.0) == 0.0


# --- Equity ----------------------------------------------------------------


def test_compute_equity_flat() -> None:
    """Equity == cash when no positions."""
    eq = compute_equity(cash=100_000.0, positions={}, mark_prices={})
    assert eq == 100_000.0


def test_compute_equity_long() -> None:
    """cash unaffected; equity adds unrealized."""
    positions = {"SPY": apply_trade(_flat(), "buy", 10, 100.0)}
    eq = compute_equity(cash=99_000.0, positions=positions, mark_prices={"SPY": 110.0})
    # cash 99,000 + unrealized (10 * (110 - 100)) = 99,100.
    assert eq == pytest.approx(100_100.0)


def test_compute_equity_short() -> None:
    positions = {"SPY": apply_trade(_flat(), "sell", 10, 100.0)}
    eq = compute_equity(cash=101_000.0, positions=positions, mark_prices={"SPY": 95.0})
    # cash 101,000 + short unrealized (10 * (100 - 95)) = 101,050.
    assert eq == pytest.approx(100_050.0)


def test_compute_equity_missing_mark_treats_position_at_avg_entry() -> None:
    """Defensive: if a price is missing, fall back to avg_entry (zero unrealized)."""
    positions = {"SPY": apply_trade(_flat(), "buy", 10, 100.0)}
    eq = compute_equity(cash=99_000.0, positions=positions, mark_prices={})
    # mark falls back to avg_entry → unrealized = 0 → equity = cash
    assert eq == pytest.approx(100_000.0)


# --- Validation ------------------------------------------------------------


def test_apply_trade_rejects_zero_quantity() -> None:
    with pytest.raises(ValueError):
        apply_trade(_flat(), "buy", 0, 100.0)


def test_apply_trade_rejects_negative_quantity() -> None:
    with pytest.raises(ValueError):
        apply_trade(_flat(), "buy", -5, 100.0)


def test_apply_trade_rejects_negative_price() -> None:
    with pytest.raises(ValueError):
        apply_trade(_flat(), "buy", 10, -1.0)


# --- Contract multiplier ---------------------------------------------------


def _flat_mult(multiplier: float) -> PositionState:
    """A flat position carrying a contract multiplier."""
    return PositionState(
        instrument="ES",
        side="flat",
        quantity=0,
        avg_entry_price=0.0,
        realized_pnl=0.0,
        multiplier=multiplier,
        futures_style=True,
    )


def test_multiplier_defaults_to_one() -> None:
    assert _flat().multiplier == 1.0


def test_multiplier_preserved_through_open_and_add() -> None:
    pos = apply_trade(_flat_mult(50.0), "buy", 2, 4000.0)
    assert pos.multiplier == 50.0
    pos = apply_trade(pos, "buy", 2, 4100.0)
    assert pos.multiplier == 50.0


def test_unrealized_pnl_scales_with_multiplier_long() -> None:
    pos = apply_trade(_flat_mult(50.0), "buy", 2, 4000.0)
    # 2 contracts * (4010 - 4000) * 50 = 1000
    assert unrealized_pnl(pos, 4010.0) == pytest.approx(1000.0)


def test_unrealized_pnl_scales_with_multiplier_short() -> None:
    pos = apply_trade(_flat_mult(50.0), "sell", 1, 4000.0)
    # 1 * (4000 - 3990) * 50 = 500
    assert unrealized_pnl(pos, 3990.0) == pytest.approx(500.0)


def test_realized_pnl_scales_with_multiplier_on_close() -> None:
    pos = apply_trade(_flat_mult(50.0), "buy", 2, 4000.0)
    closed = apply_trade(pos, "sell", 2, 4020.0)
    # 2 * (4020 - 4000) * 50 = 2000
    assert closed.side == "flat"
    assert closed.realized_pnl == pytest.approx(2000.0)


def test_realized_pnl_scales_with_multiplier_on_partial_close() -> None:
    pos = apply_trade(_flat_mult(50.0), "buy", 3, 4000.0)
    reduced = apply_trade(pos, "sell", 1, 4020.0)
    # 1 * (4020 - 4000) * 50 = 1000
    assert reduced.quantity == 2
    assert reduced.realized_pnl == pytest.approx(1000.0)


def test_flip_preserves_multiplier_and_realizes_scaled_pnl() -> None:
    pos = apply_trade(_flat_mult(50.0), "buy", 2, 4000.0)
    flipped = apply_trade(pos, "sell", 5, 4010.0)
    # closes 2 long: 2 * (4010 - 4000) * 50 = 1000 realized; opens 3 short.
    assert flipped.side == "short"
    assert flipped.quantity == 3
    assert flipped.multiplier == 50.0
    assert flipped.realized_pnl == pytest.approx(1000.0)


def test_compute_equity_applies_multiplier() -> None:
    positions = {"ES": apply_trade(_flat_mult(50.0), "buy", 2, 4000.0)}
    eq = compute_equity(cash=50_000.0, positions=positions, mark_prices={"ES": 4010.0})
    # 50,000 + 2 * (4010 - 4000) * 50 = 51,000
    assert eq == pytest.approx(51_000.0)


def test_apply_trade_rejects_non_positive_multiplier() -> None:
    with pytest.raises(ValueError):
        apply_trade(_flat_mult(0.0), "buy", 1, 100.0)
