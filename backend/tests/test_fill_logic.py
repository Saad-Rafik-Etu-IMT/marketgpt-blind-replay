"""Tests for the pure fill-logic functions — Phase 3, Module 8.

The fill rules from §7 transcribed into tests:

  Market order:
    - Fills SAME bar at the bar's open.
  Limit buy:
    - Fills on bar N+1.. when low <= limit_price.
    - Fill price = min(open, limit_price) of fill bar.
  Limit sell:
    - Fills on bar N+1.. when high >= limit_price.
    - Fill price = max(open, limit_price) of fill bar.
  Stop buy:
    - Fills on bar N+1.. when high >= stop_price.
    - Fill price = max(open, stop_price) of fill bar.
  Stop sell:
    - Fills on bar N+1.. when low <= stop_price.
    - Fill price = min(open, stop_price) of fill bar.

Plus: every fill price must lie within [bar.low, bar.high] — §5 invariant 3.
"""

import subprocess
import sys
from datetime import date

import pytest

from src.core.fill_logic import (
    FillDecision,
    FillLogicError,
    OrderInputs,
    _validated,
    decide_fill,
)
from src.schemas.market import OHLCVBar


def _bar(open_: float, high: float, low: float, close: float) -> OHLCVBar:
    return OHLCVBar(
        date=date(2023, 1, 3),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1_000_000,
        adjusted_close=close,
    )


# --- Market orders ---------------------------------------------------------


def test_market_buy_fills_next_bar_at_open() -> None:
    order = OrderInputs(side="buy", order_type="market")
    decision = decide_fill(
        order, _bar(100, 105, 98, 102), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is True
    assert decision.fill_price == 100.0


def test_market_sell_fills_next_bar_at_open() -> None:
    order = OrderInputs(side="sell", order_type="market")
    decision = decide_fill(
        order, _bar(100, 105, 98, 102), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is True
    assert decision.fill_price == 100.0


# --- Limit buy -------------------------------------------------------------


def test_limit_buy_fills_when_low_below_limit() -> None:
    order = OrderInputs(side="buy", order_type="limit", limit_price=99.0)
    # bar: open=100, low=98 → low <= limit (99); fill at min(open, limit) = 99
    decision = decide_fill(
        order, _bar(100, 105, 98, 102), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is True
    assert decision.fill_price == 99.0


def test_limit_buy_open_below_limit_fills_at_open() -> None:
    """Gap-down: open is already below limit → fill at the gapped open, not at the limit."""
    order = OrderInputs(side="buy", order_type="limit", limit_price=99.0)
    # open=95, limit=99, low=94 → fill at min(95, 99) = 95
    decision = decide_fill(
        order, _bar(95, 100, 94, 98), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is True
    assert decision.fill_price == 95.0


def test_limit_buy_does_not_fill_when_price_above_limit() -> None:
    order = OrderInputs(side="buy", order_type="limit", limit_price=99.0)
    # low=100 > limit=99 → no fill.
    decision = decide_fill(
        order, _bar(102, 105, 100, 104), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is False


def test_limit_buy_does_not_fill_same_bar_it_was_submitted() -> None:
    """Limit orders cannot fill on the bar of submission (§5 invariant 2)."""
    order = OrderInputs(side="buy", order_type="limit", limit_price=99.0)
    decision = decide_fill(
        order, _bar(100, 105, 98, 102), submitted_at_bar=5, current_bar=5
    )
    assert decision.should_fill is False


# --- Limit sell ------------------------------------------------------------


def test_limit_sell_fills_when_high_above_limit() -> None:
    order = OrderInputs(side="sell", order_type="limit", limit_price=104.0)
    # high=105 >= limit (104); fill at max(open, limit) = max(100, 104) = 104
    decision = decide_fill(
        order, _bar(100, 105, 98, 102), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is True
    assert decision.fill_price == 104.0


def test_limit_sell_open_above_limit_fills_at_open() -> None:
    """Gap-up: open already above limit → fill at the gapped open."""
    order = OrderInputs(side="sell", order_type="limit", limit_price=104.0)
    # open=108, limit=104, high=110 → fill at max(108, 104) = 108
    decision = decide_fill(
        order, _bar(108, 110, 105, 109), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is True
    assert decision.fill_price == 108.0


def test_limit_sell_does_not_fill_when_price_below_limit() -> None:
    order = OrderInputs(side="sell", order_type="limit", limit_price=110.0)
    decision = decide_fill(
        order, _bar(100, 105, 98, 102), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is False


# --- Stop buy --------------------------------------------------------------


def test_stop_buy_triggers_when_high_above_stop() -> None:
    order = OrderInputs(side="buy", order_type="stop", stop_price=104.0)
    # high=105 >= stop=104; fill at max(open, stop) = max(100, 104) = 104
    decision = decide_fill(
        order, _bar(100, 105, 98, 102), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is True
    assert decision.fill_price == 104.0


def test_stop_buy_gap_above_stop_fills_at_open() -> None:
    """Gap-up above stop → fill at open (worse than stop, that's slippage)."""
    order = OrderInputs(side="buy", order_type="stop", stop_price=104.0)
    decision = decide_fill(
        order, _bar(108, 110, 106, 109), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is True
    assert decision.fill_price == 108.0  # max(108, 104)


def test_stop_buy_does_not_trigger_when_high_below_stop() -> None:
    order = OrderInputs(side="buy", order_type="stop", stop_price=110.0)
    decision = decide_fill(
        order, _bar(100, 105, 98, 102), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is False


# --- Stop sell -------------------------------------------------------------


def test_stop_sell_triggers_when_low_below_stop() -> None:
    order = OrderInputs(side="sell", order_type="stop", stop_price=99.0)
    # low=98 <= stop=99; fill at min(open, stop) = min(100, 99) = 99
    decision = decide_fill(
        order, _bar(100, 105, 98, 102), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is True
    assert decision.fill_price == 99.0


def test_stop_sell_gap_below_stop_fills_at_open() -> None:
    order = OrderInputs(side="sell", order_type="stop", stop_price=99.0)
    # open=95 < stop=99 → fill at min(95, 99) = 95
    decision = decide_fill(
        order, _bar(95, 100, 90, 92), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is True
    assert decision.fill_price == 95.0


def test_stop_sell_does_not_trigger_when_low_above_stop() -> None:
    order = OrderInputs(side="sell", order_type="stop", stop_price=95.0)
    decision = decide_fill(
        order, _bar(100, 105, 98, 102), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is False


# --- §5 invariant 3 — fill price within [low, high] -----------------------


def test_every_fill_price_is_within_bar_range() -> None:
    """No fill price may exceed [bar.low, bar.high] (§5 invariant 3)."""
    bar = _bar(100, 105, 98, 102)
    fill_cases: list[OrderInputs] = [
        OrderInputs(side="buy", order_type="market"),
        OrderInputs(side="sell", order_type="market"),
        OrderInputs(side="buy", order_type="limit", limit_price=99.0),
        OrderInputs(side="sell", order_type="limit", limit_price=104.0),
        OrderInputs(side="buy", order_type="stop", stop_price=104.0),
        OrderInputs(side="sell", order_type="stop", stop_price=99.0),
    ]
    for order in fill_cases:
        # Use a later current_bar so limit/stop are eligible too.
        decision = decide_fill(order, bar, submitted_at_bar=0, current_bar=1)
        if decision.should_fill:
            assert bar.low <= decision.fill_price <= bar.high, (
                f"violation for {order}: {decision.fill_price} not in "
                f"[{bar.low}, {bar.high}]"
            )


def test_market_fills_are_clamped_to_bar_range() -> None:
    """Even pathological inputs cannot escape the [low, high] window."""
    bar = _bar(100, 105, 98, 102)
    # Market orders fill at bar.open which by construction is in range.
    decision = decide_fill(
        OrderInputs(side="buy", order_type="market"),
        bar,
        submitted_at_bar=5,
        current_bar=6,
    )
    assert 98 <= decision.fill_price <= 105


# --- Validation ------------------------------------------------------------


def test_limit_order_requires_limit_price() -> None:
    with pytest.raises(ValueError):
        OrderInputs(side="buy", order_type="limit")  # missing limit_price


def test_stop_order_requires_stop_price() -> None:
    with pytest.raises(ValueError):
        OrderInputs(side="sell", order_type="stop")  # missing stop_price


def test_market_order_ignores_extraneous_prices() -> None:
    """Providing limit_price on a market order is harmless (ignored)."""
    order = OrderInputs(side="buy", order_type="market", limit_price=99.0)
    decision = decide_fill(
        order, _bar(100, 105, 98, 102), submitted_at_bar=5, current_bar=6
    )
    assert decision.should_fill is True
    assert decision.fill_price == 100.0


# --- Decision type ---------------------------------------------------------


def test_decision_has_documented_fields() -> None:
    decision: FillDecision = decide_fill(
        OrderInputs(side="buy", order_type="market"),
        _bar(100, 105, 98, 102),
        submitted_at_bar=5,
        current_bar=6,
    )
    assert decision.should_fill is True
    assert decision.fill_price == 100.0
    # A non-fill decision carries fill_price=None.
    no_fill = decide_fill(
        OrderInputs(side="buy", order_type="limit", limit_price=50.0),
        _bar(100, 105, 98, 102),
        submitted_at_bar=5,
        current_bar=6,
    )
    assert no_fill.should_fill is False
    assert no_fill.fill_price is None


# --- §5 invariant 3 raises (not asserts) — survives python -O --------------


def test_out_of_range_fill_price_raises_fill_logic_error() -> None:
    """A fill price outside [low, high] must raise, not silently pass."""
    bar = _bar(100, 105, 98, 102)
    with pytest.raises(FillLogicError):
        _validated(120.0, bar)  # above high
    with pytest.raises(FillLogicError):
        _validated(90.0, bar)  # below low


def test_invariant_holds_under_python_dash_o() -> None:
    """The guarantee must survive `python -O`, which strips `assert`.

    Runs a fresh interpreter with optimizations enabled and asserts that an
    out-of-range fill price still raises FillLogicError.
    """
    code = (
        "from datetime import date\n"
        "from src.core.fill_logic import _validated, FillLogicError\n"
        "from src.schemas.market import OHLCVBar\n"
        "bar = OHLCVBar(date=date(2023,1,3), open=100, high=105, low=98,"
        " close=102, volume=1, adjusted_close=102)\n"
        "try:\n"
        "    _validated(120.0, bar)\n"
        "except FillLogicError:\n"
        "    print('RAISED')\n"
        "else:\n"
        "    print('NO_RAISE')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "RAISED"
