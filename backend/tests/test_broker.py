"""Tests for the broker fill engine — Phase 3.

The broker is the integration point: it walks pending orders against
the current bar and applies the pure ``fill_logic.decide_fill``,
persisting fills and transitioning order status.
"""

from datetime import date

import pytest
from sqlalchemy.orm import Session

from src.models.fill import FillORM
from src.models.market import OHLCVBarORM
from src.models.order import OrderORM
from src.schemas.order import CreateOrderRequest
from src.services.broker import process_pending_orders
from src.services.order_service import submit_order
from src.services.replay_engine import create_session, step_session


@pytest.fixture
def seeded(db_session: Session) -> Session:
    """Seed 10 SPY bars with varied price action for fill tests."""
    # Crafted so:
    #  - bar 0 (Jan 3): tight range around 100
    #  - bar 1 (Jan 4): dips to 96 — triggers buy limits at 99, buy stops at 100
    #  - bar 2 (Jan 5): rallies to 110 — triggers sell limits at 105, sell stops at 100
    rows = [
        (3, 100, 102, 99, 101),
        (4, 100, 103, 96, 98),
        (5, 98, 110, 97, 108),
        (6, 108, 112, 106, 110),
        (9, 110, 115, 108, 113),
        (10, 113, 116, 110, 114),
        (11, 114, 118, 112, 117),
        (12, 117, 120, 115, 119),
        (13, 119, 121, 117, 120),
        (16, 120, 122, 118, 121),
    ]
    for day, o, h, low, c in rows:
        db_session.add(
            OHLCVBarORM(
                instrument="SPY",
                date=date(2023, 1, day),
                open=o,
                high=h,
                low=low,
                close=c,
                volume=1_000_000,
                adjusted_close=c,
            )
        )
    db_session.commit()
    return db_session


def _session_id(db: Session) -> str:
    sess = create_session(db, "SPY", date(2023, 1, 3), date(2023, 1, 16))
    return sess.session_id


# --- Market orders ---------------------------------------------------------


def test_market_order_fills_next_bar_at_open(seeded: Session) -> None:
    sid = _session_id(seeded)
    order = submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="market",
            quantity=10,
            rationale="market entry test",
        ),
    )
    assert order.status == "pending"
    step_session(seeded, sid, n=1)
    order = seeded.get(OrderORM, order.order_id)
    # submit_order itself drives the same-bar fill via the broker.
    assert order.status == "filled"

    fill = seeded.query(FillORM).filter_by(order_id=order.order_id).one()
    assert fill.fill_price == 100.0  # bar 0 open
    assert fill.fill_bar == 1
    assert fill.fill_quantity == 10


def test_market_sell_at_subsequent_bar(seeded: Session) -> None:
    sid = _session_id(seeded)
    step_session(seeded, sid, n=2)  # now at bar 2 (open=98)
    order = submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="sell",
            order_type="market",
            quantity=5,
            stop_loss=120,
            rationale="market short",
        ),
    )
    assert order.status == "pending"
    step_session(seeded, sid, n=1)
    order = seeded.get(OrderORM, order.order_id)
    assert order.status == "filled"
    fill = seeded.query(FillORM).filter_by(order_id=order.order_id).one()
    assert fill.fill_price == 108.0
    assert fill.fill_bar == 3


# --- Limit orders ----------------------------------------------------------


def test_limit_buy_stays_pending_until_price_hits(seeded: Session) -> None:
    sid = _session_id(seeded)
    order = submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="limit",
            quantity=10,
            limit_price=97.0,
            rationale="below market",
        ),
    )
    # Bar 0 low = 99 → does NOT trigger limit at 97.
    assert order.status == "pending"

    # Step to bar 1 (low=96, dip below 97) → should fill.
    step_session(seeded, sid, n=1)
    process_pending_orders(seeded, sid)

    refreshed = seeded.get(OrderORM, order.order_id)
    assert refreshed is not None
    assert refreshed.status == "filled"
    fill = seeded.query(FillORM).filter_by(order_id=order.order_id).one()
    # Bar 1: open=100, limit=97 → fill at min(100, 97) = 97
    assert fill.fill_price == 97.0
    assert fill.fill_bar == 1


def test_limit_buy_does_not_fill_on_submission_bar(seeded: Session) -> None:
    """A limit that would fill at submission's bar still stays pending (§5)."""
    sid = _session_id(seeded)
    # Bar 0: low=99. A limit-buy at 99.5 would qualify (low <= limit), but
    # the rule forbids same-bar fills for limit orders.
    order = submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="limit",
            quantity=10,
            limit_price=99.5,
            rationale="same-bar test",
        ),
    )
    assert order.status == "pending"


# --- Stop orders -----------------------------------------------------------


def test_stop_sell_triggers_on_dip(seeded: Session) -> None:
    sid = _session_id(seeded)
    order = submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="sell",
            order_type="stop",
            quantity=10,
            stop_price=97.5,
            stop_loss=110,
            rationale="protective stop",
        ),
    )
    assert order.status == "pending"

    step_session(seeded, sid, n=1)
    process_pending_orders(seeded, sid)

    refreshed = seeded.get(OrderORM, order.order_id)
    assert refreshed is not None
    assert refreshed.status == "filled"
    fill = seeded.query(FillORM).filter_by(order_id=order.order_id).one()
    # Bar 1: open=100, stop=97.5, low=96 → fill at min(100, 97.5) = 97.5
    assert fill.fill_price == 97.5


# --- Cancellation ---------------------------------------------------------


def test_cancelled_orders_are_skipped(seeded: Session) -> None:
    from src.services.order_service import cancel_order

    sid = _session_id(seeded)
    order = submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="limit",
            quantity=10,
            limit_price=97.0,
            rationale="will cancel",
        ),
    )
    cancel_order(seeded, order.order_id)

    step_session(seeded, sid, n=1)
    process_pending_orders(seeded, sid)

    refreshed = seeded.get(OrderORM, order.order_id)
    assert refreshed.status == "cancelled"
    assert seeded.query(FillORM).filter_by(order_id=order.order_id).count() == 0


# --- §5 invariant 3 — every persisted fill is within bar range -------------


def test_persisted_fill_prices_are_within_bar_range(seeded: Session) -> None:
    """All fills satisfy bar.low <= fill_price <= bar.high (§5 invariant 3)."""
    sid = _session_id(seeded)
    submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid, side="buy", order_type="market", quantity=10, rationale="m1"
        ),
    )
    step_session(seeded, sid, n=1)
    submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="limit",
            quantity=10,
            limit_price=97.0,
            rationale="l1",
        ),
    )
    step_session(seeded, sid, n=1)
    process_pending_orders(seeded, sid)

    fills = seeded.query(FillORM).all()
    bars = seeded.query(OHLCVBarORM).order_by(OHLCVBarORM.date).all()
    for f in fills:
        bar = bars[f.fill_bar]
        assert bar.low <= f.fill_price <= bar.high, (
            f"Invariant 3 violated: fill {f.fill_id} @ {f.fill_price} "
            f"outside bar {f.fill_bar} [{bar.low}, {bar.high}]"
        )
