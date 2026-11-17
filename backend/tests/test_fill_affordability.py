"""Fill-time affordability (execution policy fill-time-cash-check-v1).

Synthetic bars only. Some tests insert pending orders directly: the order
service allows one pending top-level order at a time, so competing orders can
only reach the broker this way; it exercises the broker in isolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.config import Config
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session

from alembic import command
from src.api.dependencies import _BACKEND_DIR
from src.core.fill_logic import cash_after_cash_buy
from src.database import build_engine
from src.models.fill import FillORM
from src.models.market import OHLCVBarORM
from src.models.order import OrderORM
from src.models.portfolio import PositionORM
from src.schemas.order import INSUFFICIENT_CASH, CreateOrderRequest
from src.schemas.risk import RiskLimitsUpdate
from src.services.order_service import list_orders, submit_order
from src.services.portfolio_service import current_snapshot
from src.services.replay_engine import create_session, step_session
from src.services.risk_service import update_limits

BENCHMARK_LIMITS = RiskLimitsUpdate(
    max_drawdown_pct=100, max_risk_per_trade_pct=100, max_open_positions=1, max_exposure_pct=100
)


def seed(
    db: Session, instrument: str, opens: Sequence[float], closes: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
) -> None:
    closes = closes or opens
    day = date(2022, 1, 3)
    for i, (o, c) in enumerate(zip(opens, closes, strict=True)):
        low = lows[i] if lows else min(o, c) - 1
        db.add(OHLCVBarORM(
            instrument=instrument, date=day + timedelta(days=i), open=o, high=max(o, c) + 1,
            low=low, close=c, volume=1_000, adjusted_close=c,
        ))
    db.commit()


def session(db: Session, instrument: str = "AAA", **kwargs: float) -> str:
    return create_session(db, instrument, date(2022, 1, 1), date(2022, 12, 31), **kwargs).session_id


def pending(db: Session, sid: str, side: str, qty: int, *, seconds: int = 0, **extra: object) -> OrderORM:
    order = OrderORM(
        order_id=str(uuid4()), session_id=sid, side=side, order_type="market", quantity=qty,
        status="pending", submitted_at_bar=0, rationale="test",
        submitted_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
        **extra,
    )
    db.add(order)
    db.commit()
    return order


def fills(db: Session, order: OrderORM) -> int:
    return db.scalar(select(func.count()).where(FillORM.order_id == order.order_id)) or 0


def position_qty(db: Session, sid: str) -> int:
    pos = db.scalar(select(PositionORM).where(PositionORM.session_id == sid))
    return 0 if pos is None else pos.quantity


def cash(db: Session, sid: str) -> float:
    return current_snapshot(db, sid).cash


def assert_rejected_without_effect(db: Session, order: OrderORM) -> None:
    db.refresh(order)
    assert order.status == "rejected"
    assert order.rejection_reason == INSUFFICIENT_CASH
    assert fills(db, order) == 0
    assert db.scalar(
        select(func.count()).where(OrderORM.parent_order_id == order.order_id)
    ) == 0


# --- Rejection and exact affordability ---------------------------------------


def test_gap_up_rejects_buy_sized_at_close_n(db_session: Session) -> None:
    seed(db_session, "AAA", opens=[100, 110, 111], closes=[100, 112, 111])
    sid = session(db_session, initial_cash=1000.0)
    update_limits(db_session, sid, BENCHMARK_LIMITS)
    order = submit_order(db_session, CreateOrderRequest(
        session_id=sid, side="buy", order_type="market", quantity=10, rationale="all in",
    ))
    assert order.status == "pending"  # N+1 timing: nothing is evaluated on bar N

    step_session(db_session, sid, 1)

    orm = db_session.get(OrderORM, order.order_id)
    assert orm is not None
    assert_rejected_without_effect(db_session, orm)
    assert cash(db_session, sid) == 1000.0 and position_qty(db_session, sid) == 0
    read = next(o for o in list_orders(db_session, sid) if o.order_id == order.order_id)
    assert (read.status, read.rejection_reason) == ("rejected", "INSUFFICIENT_CASH")
    step_session(db_session, sid, 1)  # a rejected order is terminal
    assert orm.status == "rejected" and fills(db_session, orm) == 0


@pytest.mark.parametrize(("open_price", "filled"), [(99.0, True), (99.01, False)])
def test_exact_affordability_and_just_insufficient_cash(
    db_session: Session, open_price: float, filled: bool
) -> None:
    """Cash 1000; 10 units at 99 plus 1.00 per-unit fee costs exactly 1000."""
    seed(db_session, "AAA", opens=[100, open_price])
    sid = session(db_session, initial_cash=1000.0, fee_per_unit=1.0)
    order = pending(db_session, sid, "buy", 10)

    step_session(db_session, sid, 1)

    if filled:
        db_session.refresh(order)
        assert order.status == "filled" and order.rejection_reason is None
        assert cash(db_session, sid) == 0.0 and position_qty(db_session, sid) == 10
    else:
        assert_rejected_without_effect(db_session, order)
        assert cash(db_session, sid) == 1000.0 and position_qty(db_session, sid) == 0


def test_spread_and_fees_count_toward_required_cash(db_session: Session) -> None:
    seed(db_session, "AAA", opens=[100, 100])
    # Notional 1000 alone is affordable; the 20 bp full spread (10 bp per side) is not.
    spread = session(db_session, initial_cash=1000.0, spread_bps=20.0)
    spread_order = pending(db_session, spread, "buy", 10)
    # Notional 1000 alone is affordable; the 0.50 per-unit fee is not.
    fee = session(db_session, initial_cash=1000.0, fee_per_unit=0.5)
    fee_order = pending(db_session, fee, "buy", 10)
    room = session(db_session, initial_cash=1002.0, spread_bps=20.0)
    room_order = pending(db_session, room, "buy", 10)

    for sid in (spread, fee, room):
        step_session(db_session, sid, 1)

    assert_rejected_without_effect(db_session, spread_order)
    assert_rejected_without_effect(db_session, fee_order)
    fill = db_session.scalar(select(FillORM).where(FillORM.order_id == room_order.order_id))
    assert fill is not None and fill.fill_price > 100.0
    assert cash(db_session, room) == cash_after_cash_buy(1002.0, 10, fill.fill_price, 1.0, 0.0)
    assert cash(db_session, room) >= 0


def test_competing_buys_read_cash_after_earlier_fills(db_session: Session) -> None:
    seed(db_session, "AAA", opens=[100, 100])
    sid = session(db_session, initial_cash=1000.0)
    first = pending(db_session, sid, "buy", 6, seconds=0)
    second = pending(db_session, sid, "buy", 6, seconds=1)

    step_session(db_session, sid, 1)

    db_session.refresh(first)
    assert first.status == "filled"
    assert_rejected_without_effect(db_session, second)
    assert cash(db_session, sid) == 400.0 and position_qty(db_session, sid) == 6


def test_rejection_does_not_block_a_later_eligible_order(db_session: Session) -> None:
    seed(db_session, "AAA", opens=[100, 100])
    sid = session(db_session, initial_cash=1000.0)
    too_big = pending(db_session, sid, "buy", 20, seconds=0)
    eligible = pending(db_session, sid, "buy", 5, seconds=1)

    step_session(db_session, sid, 1)

    assert_rejected_without_effect(db_session, too_big)
    db_session.refresh(eligible)
    assert eligible.status == "filled" and fills(db_session, eligible) == 1
    assert cash(db_session, sid) == 500.0 and position_qty(db_session, sid) == 5


# --- Paths outside the check ------------------------------------------------------


def test_sell_exits_fill_at_zero_cash_under_zero_fee(db_session: Session) -> None:
    """Supported assumption: zero commission, so exit proceeds are never negative."""
    seed(db_session, "AAA", opens=[100, 100, 90])
    sid = session(db_session, initial_cash=1000.0, spread_bps=1.0, fee_per_unit=0.0)
    buy = pending(db_session, sid, "buy", 9)
    step_session(db_session, sid, 1)
    db_session.refresh(buy)
    assert buy.status == "filled"
    cash_after_buy = cash(db_session, sid)
    sell = pending(db_session, sid, "sell", 9, seconds=5, reduce_only=True)
    sell.submitted_at_bar = 1
    db_session.commit()

    step_session(db_session, sid, 1)

    db_session.refresh(sell)
    assert sell.status == "filled" and sell.rejection_reason is None
    assert cash(db_session, sid) > cash_after_buy >= 0
    assert position_qty(db_session, sid) == 0


def test_protective_children_are_created_only_for_accepted_entries(db_session: Session) -> None:
    # Bar 1 opens at 100 and trades down to 90: the stop child fills on the entry bar.
    seed(db_session, "AAA", opens=[100, 100, 100], closes=[100, 95, 100], lows=[99, 90, 99])
    rejected_sid = session(db_session, initial_cash=999.0)
    rejected = pending(db_session, rejected_sid, "buy", 10, stop_loss=95.0, take_profit=120.0)
    accepted_sid = session(db_session, initial_cash=1000.0)
    accepted = pending(db_session, accepted_sid, "buy", 10, stop_loss=95.0, take_profit=120.0)

    step_session(db_session, rejected_sid, 1)
    step_session(db_session, accepted_sid, 1)

    assert_rejected_without_effect(db_session, rejected)
    children = list(db_session.scalars(
        select(OrderORM).where(OrderORM.parent_order_id == accepted.order_id)
    ))
    assert {c.order_type: c.status for c in children} == {"stop": "filled", "limit": "cancelled"}
    # Entry left cash at 0; the protective sell is an exit and is never cash-checked.
    assert cash(db_session, accepted_sid) == 950.0 and position_qty(db_session, accepted_sid) == 0


def test_futures_behaviour_is_unchanged(db_session: Session) -> None:
    """Futures margin is not modelled; the cash check does not apply to =F."""
    seed(db_session, "ES=F", opens=[4000, 4000, 4010])
    sid = session(db_session, "ES=F", initial_cash=1000.0)
    order = pending(db_session, sid, "buy", 1)

    step_session(db_session, sid, 1)

    db_session.refresh(order)
    assert order.status == "filled" and order.rejection_reason is None
    assert position_qty(db_session, sid) == 1


def test_affordable_market_buy_still_fills_at_next_open(db_session: Session) -> None:
    seed(db_session, "AAA", opens=[100, 101, 102], closes=[100, 101, 102])
    sid = session(db_session, initial_cash=1000.0)
    order = pending(db_session, sid, "buy", 5)

    step_session(db_session, sid, 1)

    fill = db_session.scalar(select(FillORM).where(FillORM.order_id == order.order_id))
    assert fill is not None and (fill.fill_bar, fill.fill_price) == (1, 101.0)


# --- Migration ---------------------------------------------------------------------


def test_migration_adds_nullable_rejection_reason_from_0001(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'upgrade.sqlite'}"
    cfg = Config(str(_BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(_BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "0001")
    engine = build_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO replay_sessions (session_id, instrument, start_date, end_date, "
                "current_bar_index, status, initial_cash, multiplier, simulation_version, "
                "fee_per_unit, spread_bps, created_at) VALUES ('s', 'AAA', '2022-01-01', "
                "'2022-12-31', 0, 'active', 1000, 1, 2, 0, 0, '2026-01-01')"
            ))
            conn.execute(text(
                "INSERT INTO orders (order_id, session_id, side, order_type, quantity, "
                "instrument_kind, status, reduce_only, submitted_at_bar, submitted_at, rationale) "
                "VALUES ('o', 's', 'buy', 'market', 1, 'underlying', 'filled', 0, 0, "
                "'2026-01-01', 'old')"
            ))

        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            columns = {c["name"]: c for c in inspect(conn).get_columns("orders")}
            assert columns["rejection_reason"]["nullable"]
            assert conn.execute(text(
                "SELECT status, rejection_reason FROM orders WHERE order_id = 'o'"
            )).one() == ("filled", None)

        command.downgrade(cfg, "0001")
        with engine.connect() as conn:
            assert "rejection_reason" not in {c["name"] for c in inspect(conn).get_columns("orders")}
            assert conn.execute(text("SELECT count(*) FROM orders")).scalar_one() == 1
    finally:
        engine.dispose()
