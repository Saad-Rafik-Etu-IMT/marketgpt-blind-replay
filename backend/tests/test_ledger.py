"""Ledger / P&L invariants — prompt-mandated ``test_ledger.py``.

The prompt's §9 test list names ``test_ledger.py`` explicitly. The
ledger's behaviour lives in two layers:

  * ``src.core.pnl`` — the *pure* position math (open / add / reduce /
    close / flip, realised & unrealised P&L, equity). This file
    exercises that layer exhaustively, with no DB and no fixtures, so
    the accounting identities are pinned independently of persistence.
  * ``src.services.portfolio_service`` — the DB-backed snapshot writer,
    covered by ``test_portfolio_service.py``.

Together they satisfy the mandated ledger coverage. Keeping the
accounting laws here, against the pure layer, means a regression in the
money math fails loudly and in isolation — not buried in an integration
test.
"""

from __future__ import annotations

import pytest

from src.core.pnl import (
    PositionState,
    apply_trade,
    compute_equity,
    unrealized_pnl,
)


def _flat(instrument: str = "SPY") -> PositionState:
    return PositionState(
        instrument=instrument,
        side="flat",
        quantity=0,
        avg_entry_price=0.0,
        realized_pnl=0.0,
    )


# --- Opening ---------------------------------------------------------------


def test_buy_from_flat_opens_long() -> None:
    pos = apply_trade(_flat(), side="buy", quantity=10, price=100.0)
    assert pos.side == "long"
    assert pos.quantity == 10
    assert pos.avg_entry_price == 100.0
    assert pos.realized_pnl == 0.0


def test_sell_from_flat_opens_short() -> None:
    pos = apply_trade(_flat(), side="sell", quantity=5, price=200.0)
    assert pos.side == "short"
    assert pos.quantity == 5
    assert pos.avg_entry_price == 200.0


# --- Adding (weighted-average cost) ----------------------------------------


def test_adding_to_long_uses_weighted_average_entry() -> None:
    pos = apply_trade(_flat(), side="buy", quantity=10, price=100.0)
    pos = apply_trade(pos, side="buy", quantity=10, price=120.0)
    assert pos.quantity == 20
    # (10*100 + 10*120) / 20 == 110
    assert pos.avg_entry_price == pytest.approx(110.0)
    assert pos.realized_pnl == 0.0  # adding never realises


def test_adding_to_short_uses_weighted_average_entry() -> None:
    pos = apply_trade(_flat(), side="sell", quantity=10, price=100.0)
    pos = apply_trade(pos, side="sell", quantity=30, price=140.0)
    assert pos.side == "short"
    assert pos.quantity == 40
    # (10*100 + 30*140) / 40 == 130
    assert pos.avg_entry_price == pytest.approx(130.0)


# --- Reducing (partial close realises P&L) ---------------------------------


def test_partial_close_of_long_realises_on_closed_quantity_only() -> None:
    pos = apply_trade(_flat(), side="buy", quantity=10, price=100.0)
    pos = apply_trade(pos, side="sell", quantity=4, price=130.0)
    assert pos.side == "long"
    assert pos.quantity == 6
    assert pos.avg_entry_price == pytest.approx(100.0)  # entry unchanged
    # 4 * (130 - 100) == 120
    assert pos.realized_pnl == pytest.approx(120.0)


def test_partial_close_of_short_realises_correct_sign() -> None:
    pos = apply_trade(_flat(), side="sell", quantity=10, price=100.0)
    pos = apply_trade(pos, side="buy", quantity=4, price=80.0)
    assert pos.side == "short"
    assert pos.quantity == 6
    # short profits when cover price < entry: 4 * (100 - 80) == 80
    assert pos.realized_pnl == pytest.approx(80.0)


# --- Exact close -----------------------------------------------------------


def test_exact_close_returns_to_flat_and_keeps_realised() -> None:
    pos = apply_trade(_flat(), side="buy", quantity=10, price=100.0)
    pos = apply_trade(pos, side="sell", quantity=10, price=90.0)
    assert pos.side == "flat"
    assert pos.quantity == 0
    assert pos.avg_entry_price == 0.0
    # 10 * (90 - 100) == -100 (a loss, preserved after returning flat)
    assert pos.realized_pnl == pytest.approx(-100.0)


# --- Flip (overshoot closes then opens the remainder) ----------------------


def test_flip_long_to_short_closes_then_opens_remainder() -> None:
    pos = apply_trade(_flat(), side="buy", quantity=10, price=100.0)
    # Sell 15: closes the 10 long (@ +0 here) and opens 5 short @ 110.
    pos = apply_trade(pos, side="sell", quantity=15, price=110.0)
    assert pos.side == "short"
    assert pos.quantity == 5
    assert pos.avg_entry_price == pytest.approx(110.0)
    # Realised only on the closed 10: 10 * (110 - 100) == 100
    assert pos.realized_pnl == pytest.approx(100.0)


def test_flip_preserves_accumulated_realised_across_the_close() -> None:
    pos = apply_trade(_flat(), side="sell", quantity=8, price=50.0)
    pos = apply_trade(pos, side="buy", quantity=20, price=40.0)
    assert pos.side == "long"
    assert pos.quantity == 12
    assert pos.avg_entry_price == pytest.approx(40.0)
    # Closed 8 short @ profit: 8 * (50 - 40) == 80
    assert pos.realized_pnl == pytest.approx(80.0)


# --- Unrealised mark-to-market --------------------------------------------


def test_unrealized_long_and_short_signs() -> None:
    long_pos = apply_trade(_flat(), side="buy", quantity=10, price=100.0)
    assert unrealized_pnl(long_pos, 110.0) == pytest.approx(100.0)
    assert unrealized_pnl(long_pos, 90.0) == pytest.approx(-100.0)

    short_pos = apply_trade(_flat(), side="sell", quantity=10, price=100.0)
    assert unrealized_pnl(short_pos, 90.0) == pytest.approx(100.0)
    assert unrealized_pnl(short_pos, 110.0) == pytest.approx(-100.0)


def test_unrealized_of_flat_is_zero() -> None:
    assert unrealized_pnl(_flat(), 123.45) == 0.0


# --- Equity aggregation ----------------------------------------------------


def test_equity_is_cash_plus_signed_market_value() -> None:
    long_pos = apply_trade(_flat(), side="buy", quantity=10, price=100.0)
    equity = compute_equity(
        cash=99_000.0,
        positions={"SPY": long_pos},
        mark_prices={"SPY": 105.0},
    )
    # 100_000 + 10 * (105 - 100) == 100_050
    assert equity == pytest.approx(100_050.0)


def test_equity_values_missing_mark_at_entry_no_crash() -> None:
    long_pos = apply_trade(_flat(), side="buy", quantity=10, price=100.0)
    # No mark for SPY → valued at entry → zero unrealised contribution.
    equity = compute_equity(cash=50_000.0, positions={"SPY": long_pos}, mark_prices={})
    assert equity == pytest.approx(51_000.0)


def test_flat_positions_excluded_from_equity() -> None:
    equity = compute_equity(
        cash=42.0, positions={"SPY": _flat()}, mark_prices={"SPY": 999.0}
    )
    assert equity == pytest.approx(42.0)


# --- Input validation ------------------------------------------------------


def test_non_positive_quantity_rejected() -> None:
    with pytest.raises(ValueError):
        apply_trade(_flat(), side="buy", quantity=0, price=100.0)


def test_negative_price_rejected() -> None:
    with pytest.raises(ValueError):
        apply_trade(_flat(), side="buy", quantity=1, price=-1.0)


# --- A small round-trip "ledger" sequence ----------------------------------


def test_full_round_trip_accounting_identity() -> None:
    """Open, scale in, scale out, flip, close — realised P&L must add up."""
    pos = _flat()
    pos = apply_trade(pos, side="buy", quantity=10, price=100.0)  # long 10 @100
    pos = apply_trade(pos, side="buy", quantity=10, price=110.0)  # long 20 @105
    pos = apply_trade(pos, side="sell", quantity=5, price=120.0)  # close 5: +75
    pos = apply_trade(
        pos, side="sell", quantity=20, price=130.0
    )  # close 15 (+375), open short 5 @130

    # Realised so far: 5*(120-105) + 15*(130-105) = 75 + 375 = 450
    assert pos.side == "short"
    assert pos.quantity == 5
    assert pos.avg_entry_price == pytest.approx(130.0)
    assert pos.realized_pnl == pytest.approx(450.0)

    pos = apply_trade(pos, side="buy", quantity=5, price=120.0)  # close short: +50
    assert pos.side == "flat"
    assert pos.quantity == 0
    assert pos.realized_pnl == pytest.approx(500.0)
