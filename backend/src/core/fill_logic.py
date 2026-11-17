"""Pure fill logic.

This module is the heart of the simulated broker. Given an order's
parameters and a single OHLCV bar, ``decide_fill`` returns the
fill decision and the price at which it would fill, per §7:

  * Market: next bar, fill at ``bar.open``.
  * Limit buy:  N+1.. when ``low <= limit_price``; price = ``min(open, limit)``.
  * Limit sell: N+1.. when ``high >= limit_price``; price = ``max(open, limit)``.
  * Stop buy:   N+1.. when ``high >= stop_price``;  price = ``max(open, stop)``.
  * Stop sell:  N+1.. when ``low <= stop_price``;   price = ``min(open, stop)``.

The function additionally enforces §5 invariant 3: every returned
fill price lies within ``[bar.low, bar.high]``. If a computation
somehow leaves the window (a bug, not a data issue), the function
raises ``FillLogicError``; the broker treats that as a fatal error
rather than silently writing a corrupt fill. A raised exception (not
an ``assert``) is deliberate: ``python -O`` strips asserts, which
would silently disable the product's central guarantee.

The module is pure: no DB, no time, no global state. This is why
the §5 anti-lookahead invariants are testable here at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.schemas.market import OHLCVBar

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop"]


class FillLogicError(RuntimeError):
    """A fill computation violated a §5 invariant (a logic bug, not bad data).

    Raised instead of using ``assert`` so the guarantee holds even under
    ``python -O``, which strips assert statements.
    """


@dataclass(frozen=True)
class OrderInputs:
    """The slice of an Order needed to decide a fill.

    Kept independent of the persistence model so the fill logic stays
    coupling-free. ``limit_price`` is required for limit orders;
    ``stop_price`` is required for stop orders. Either is harmless on
    a market order.
    """

    side: OrderSide
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None

    def __post_init__(self) -> None:
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit order requires limit_price")
        if self.order_type == "stop" and self.stop_price is None:
            raise ValueError("stop order requires stop_price")


@dataclass(frozen=True)
class FillDecision:
    """Result of evaluating an order against one bar."""

    should_fill: bool
    fill_price: float | None = None


def decide_fill(
    order: OrderInputs,
    bar: OHLCVBar,
    *,
    submitted_at_bar: int,
    current_bar: int,
) -> FillDecision:
    """Decide whether ``order`` fills against ``bar``, and at what price.

    ``submitted_at_bar`` is the bar index at submission time;
    ``current_bar`` is the bar index whose data is in ``bar``. The
    relationship between these two indices encodes §5 invariant 2:

      * Market orders only fill when ``current_bar > submitted_at_bar``.
      * Limit/stop orders only fill when ``current_bar > submitted_at_bar``.

    A fill price outside ``[bar.low, bar.high]`` raises FillLogicError
    — that condition indicates a logic bug, not bad data.
    """
    if order.order_type == "market":
        if current_bar <= submitted_at_bar:
            return FillDecision(should_fill=False)
        return _validated(bar.open, bar)

    # Limit/stop orders cannot fill on their submission bar.
    if current_bar <= submitted_at_bar:
        return FillDecision(should_fill=False)

    if order.order_type == "limit":
        return _limit_decision(order, bar)
    if order.order_type == "stop":
        return _stop_decision(order, bar)

    # Unreachable given the OrderType literal; defensive.
    raise ValueError(f"unknown order_type: {order.order_type}")


def _limit_decision(order: OrderInputs, bar: OHLCVBar) -> FillDecision:
    if order.limit_price is None:  # OrderInputs invariant
        raise FillLogicError("limit order reached fill logic without a limit_price")
    if order.side == "buy":
        if bar.low <= order.limit_price:
            return _validated(min(bar.open, order.limit_price), bar)
        return FillDecision(should_fill=False)
    # sell
    if bar.high >= order.limit_price:
        return _validated(max(bar.open, order.limit_price), bar)
    return FillDecision(should_fill=False)


def _stop_decision(order: OrderInputs, bar: OHLCVBar) -> FillDecision:
    if order.stop_price is None:  # OrderInputs invariant
        raise FillLogicError("stop order reached fill logic without a stop_price")
    if order.side == "buy":
        if bar.high >= order.stop_price:
            return _validated(max(bar.open, order.stop_price), bar)
        return FillDecision(should_fill=False)
    # sell
    if bar.low <= order.stop_price:
        return _validated(min(bar.open, order.stop_price), bar)
    return FillDecision(should_fill=False)


def _validated(price: float, bar: OHLCVBar) -> FillDecision:
    """Enforce §5 invariant 3: fill price ∈ [bar.low, bar.high]."""
    if not (bar.low <= price <= bar.high):
        raise FillLogicError(
            f"fill_price {price} outside bar range [{bar.low}, {bar.high}] — "
            "logic error in fill rules"
        )
    return FillDecision(should_fill=True, fill_price=price)


# --- Fill-time affordability (execution policy) ----------------------------

# Recorded in benchmark run metadata. Bump when fill/affordability rules change.
EXECUTION_POLICY_VERSION = "fill-time-cash-check-v1"


def cash_after_cash_buy(
    cash: float, quantity: int, price: float, multiplier: float, fee: float
) -> float:
    """Cash left after a cash-instrument buy fill.

    Mirrors ``portfolio_service.record_fill`` operation by operation (same
    float rounding), so "affordable" means the recorded cash would be >= 0.
    ``price`` is the final executable price (spread included); ``fee`` is the
    total commission for the fill.
    """
    notional = quantity * price * multiplier
    cash_delta = -notional
    cash_delta -= fee
    return cash + cash_delta
