"""Pure profit-and-loss math.

Position-tracking conventions (per §4.4 and the Phase 4 decisions log):

  * A position is one-sided — ``long``, ``short``, or ``flat``.
  * Adding to a position recomputes the entry price as a quantity-
    weighted average.
  * Reducing a position realises P&L on the closed quantity:
      long  → realized += closed_qty * (sale_price  - avg_entry)
      short → realized += closed_qty * (avg_entry - cover_price)
  * A trade that overshoots fully closes the existing side and opens
    a new position with the remainder at the trade's price.

This module is **pure**. It performs no I/O and reads no global state,
which is what makes the §5 invariants testable here at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

Side = Literal["buy", "sell"]
PositionSide = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class PositionState:
    """An immutable snapshot of one instrument's position.

    ``multiplier`` is the contract multiplier (a.k.a. point value): the
    cash P&L per one point of price move per unit of quantity. Cash
    equities use ``1.0``; a futures contract such as the E-mini S&P 500
    uses ``50.0``. It is carried on the state so the pure P&L math stays
    self-contained, and defaults to ``1.0`` for full backward
    compatibility with cash-equity positions.
    """

    instrument: str
    side: PositionSide
    quantity: int
    avg_entry_price: float
    realized_pnl: float
    multiplier: float = 1.0
    futures_style: bool = False


def apply_trade(
    position: PositionState,
    side: Side,
    quantity: int,
    price: float,
) -> PositionState:
    """Return the position after applying ``(side, quantity, price)``.

    Raises ``ValueError`` on invalid inputs (non-positive quantity or
    negative price). Never mutates ``position``.
    """
    if quantity <= 0:
        raise ValueError(f"quantity must be > 0 (got {quantity})")
    if price < 0:
        raise ValueError(f"price must be >= 0 (got {price})")
    if position.multiplier <= 0:
        raise ValueError(f"multiplier must be > 0 (got {position.multiplier})")

    if position.side == "flat":
        return _open(position, side, quantity, price)
    if _is_adding(position.side, side):
        return _add(position, quantity, price)
    return _reduce_or_flip(position, side, quantity, price)


def unrealized_pnl(position: PositionState, mark_price: float) -> float:
    """Mark-to-market P&L on an open position (signed by side)."""
    if position.side == "long":
        return (
            position.quantity
            * (mark_price - position.avg_entry_price)
            * position.multiplier
        )
    if position.side == "short":
        return (
            position.quantity
            * (position.avg_entry_price - mark_price)
            * position.multiplier
        )
    return 0.0


def compute_equity(
    cash: float,
    positions: Mapping[str, PositionState],
    mark_prices: Mapping[str, float],
) -> float:
    """Equity = cash + signed securities value + futures unrealized P&L.

    If a mark price is missing for an open position, the position is
    valued at its average entry — equivalent to zero unrealized — so a
    transient missing price does not crash the snapshot.
    """
    equity = cash
    for instrument, position in positions.items():
        if position.side == "flat":
            continue
        mark = mark_prices.get(instrument, position.avg_entry_price)
        if position.futures_style:
            equity += unrealized_pnl(position, mark)
        else:
            sign = 1.0 if position.side == "long" else -1.0
            equity += sign * position.quantity * mark * position.multiplier
    return equity


# --- Internals -------------------------------------------------------------


def _open(
    position: PositionState, side: Side, quantity: int, price: float
) -> PositionState:
    """Open a fresh position from flat."""
    return replace(
        position,
        side="long" if side == "buy" else "short",
        quantity=quantity,
        avg_entry_price=price,
    )


def _is_adding(position_side: PositionSide, trade_side: Side) -> bool:
    """A buy adds to longs; a sell adds to shorts."""
    return (position_side == "long" and trade_side == "buy") or (
        position_side == "short" and trade_side == "sell"
    )


def _add(position: PositionState, quantity: int, price: float) -> PositionState:
    """Add to an existing position — weighted-average cost basis."""
    total_qty = position.quantity + quantity
    new_avg = (
        position.avg_entry_price * position.quantity + price * quantity
    ) / total_qty
    return replace(position, quantity=total_qty, avg_entry_price=new_avg)


def _reduce_or_flip(
    position: PositionState, side: Side, quantity: int, price: float
) -> PositionState:
    """A trade in the opposite direction — reduce, close, or flip."""
    if quantity < position.quantity:
        return _reduce(position, quantity, price)
    if quantity == position.quantity:
        return _close(position, price)
    return _flip(position, side, quantity, price)


def _reduce(position: PositionState, quantity: int, price: float) -> PositionState:
    """Partial close — realise P&L on the closed portion."""
    realized = _realized_for_close(
        position.side, quantity, position.avg_entry_price, price, position.multiplier
    )
    return replace(
        position,
        quantity=position.quantity - quantity,
        realized_pnl=position.realized_pnl + realized,
    )


def _close(position: PositionState, price: float) -> PositionState:
    """Exact close — return to flat with accumulated realized P&L."""
    realized = _realized_for_close(
        position.side,
        position.quantity,
        position.avg_entry_price,
        price,
        position.multiplier,
    )
    return replace(
        position,
        side="flat",
        quantity=0,
        avg_entry_price=0.0,
        realized_pnl=position.realized_pnl + realized,
    )


def _flip(
    position: PositionState, side: Side, quantity: int, price: float
) -> PositionState:
    """Close the existing side fully and open the remainder on the new side."""
    closed = _close(position, price)
    remainder = quantity - position.quantity
    opened = _open(closed, side, remainder, price)
    # Preserve the realized P&L accumulated by the close.
    return replace(opened, realized_pnl=closed.realized_pnl)


def _realized_for_close(
    side: PositionSide,
    quantity: int,
    avg_entry: float,
    exit_price: float,
    multiplier: float = 1.0,
) -> float:
    """Realised P&L on closing ``quantity`` of a long or short position."""
    if side == "long":
        return quantity * (exit_price - avg_entry) * multiplier
    if side == "short":
        return quantity * (avg_entry - exit_price) * multiplier
    return 0.0
