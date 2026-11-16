"""OCO (one-cancels-other) tests — Phase 3.

When a parent order fills with stop_loss and/or take_profit:
  * A child stop order (opposite side) is created at stop_loss.
  * A child limit order (opposite side) is created at take_profit.
  * Both children share an oco_pair_id.
  * When either child fills, the broker auto-cancels its sibling.
  * If only one of SL/TP is provided, no OCO pairing exists.
"""

from datetime import date

import pytest
from sqlalchemy.orm import Session

from src.models.market import OHLCVBarORM
from src.models.order import OrderORM
from src.schemas.order import CreateOrderRequest
from src.services.broker import process_pending_orders
from src.services.order_service import submit_order
from src.services.replay_engine import create_session, step_session


@pytest.fixture
def seeded(db_session: Session) -> Session:
    """10 bars; bar 0 flat, bar 1 dips, bar 2 rallies (matches test_broker)."""
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


def _sid(db: Session) -> str:
    return create_session(db, "SPY", date(2023, 1, 3), date(2023, 1, 16)).session_id


def _children(db: Session, parent_id: str) -> list[OrderORM]:
    return db.query(OrderORM).filter_by(parent_order_id=parent_id).all()


# --- Child creation --------------------------------------------------------


def test_filled_parent_creates_sl_and_tp_children(seeded: Session) -> None:
    sid = _sid(seeded)
    parent = submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="market",
            quantity=10,
            stop_loss=97.0,
            take_profit=110.0,
            rationale="entry with protection",
        ),
    )
    step_session(seeded, sid, n=1)
    parent = seeded.get(OrderORM, parent.order_id)
    assert parent.status == "filled"

    kids = _children(seeded, parent.order_id)
    assert len(kids) == 2

    sl = next(k for k in kids if k.order_type == "stop")
    tp = next(k for k in kids if k.order_type == "limit")

    # Opposite side of the parent.
    assert sl.side == "sell" and tp.side == "sell"
    assert sl.stop_price == 97.0
    assert tp.limit_price == 110.0

    # OCO pairing.
    assert sl.oco_pair_id is not None
    assert sl.oco_pair_id == tp.oco_pair_id


def test_no_children_when_no_protection_specified(seeded: Session) -> None:
    sid = _sid(seeded)
    parent = submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="market",
            quantity=10,
            rationale="no protection",
        ),
    )
    assert _children(seeded, parent.order_id) == []


def test_only_sl_creates_only_one_child_no_oco(seeded: Session) -> None:
    """If only stop_loss is set, the lone child has no oco_pair_id."""
    sid = _sid(seeded)
    parent = submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="market",
            quantity=10,
            stop_loss=97.0,
            rationale="sl only",
        ),
    )
    step_session(seeded, sid, n=1)
    parent = seeded.get(OrderORM, parent.order_id)
    kids = _children(seeded, parent.order_id)
    assert len(kids) == 1
    assert kids[0].order_type == "stop"
    assert kids[0].oco_pair_id is None  # no sibling to pair with


# --- One-cancels-other ----------------------------------------------------


def test_sl_fill_cancels_tp(seeded: Session) -> None:
    sid = _sid(seeded)
    parent = submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="market",
            quantity=10,
            stop_loss=97.0,
            take_profit=120.0,
            rationale="entry",
        ),
    )

    # Step to bar 1 (low=96 < 97). SL should trigger; TP should be cancelled.
    step_session(seeded, sid, n=1)
    process_pending_orders(seeded, sid)

    kids = _children(seeded, parent.order_id)
    sl = next(k for k in kids if k.order_type == "stop")
    tp = next(k for k in kids if k.order_type == "limit")
    assert sl.status == "filled"
    assert tp.status == "cancelled"


def test_tp_fill_cancels_sl(seeded: Session) -> None:
    sid = _sid(seeded)
    parent = submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="market",
            quantity=10,
            stop_loss=90.0,
            take_profit=108.0,
            rationale="entry",
        ),
    )

    # Step to bar 2 (high=110 >= 108). TP should fill; SL should cancel.
    # NB: bar 1 has high=103, so TP doesn't trigger there.
    step_session(seeded, sid, n=2)
    process_pending_orders(seeded, sid)

    kids = _children(seeded, parent.order_id)
    sl = next(k for k in kids if k.order_type == "stop")
    tp = next(k for k in kids if k.order_type == "limit")
    assert tp.status == "filled"
    assert sl.status == "cancelled"
