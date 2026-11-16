"""Independent ledger, chronology, costs, risk and protection regressions."""

from datetime import date

import pytest

from src.models.fill import FillORM
from src.models.market import OHLCVBarORM
from src.models.order import OrderORM
from src.models.session import ReplaySessionORM
from src.schemas.order import CreateOrderRequest
from src.services.order_service import OrderRejectedError, submit_order
from src.services.portfolio_service import current_snapshot
from src.services.replay_engine import create_session, jump_to_bar, step_session


def setup(db, instrument="SPY", base=100, **kwargs):
    for day in range(1, 7):
        db.add(
            OHLCVBarORM(
                instrument=instrument,
                date=date(2023, 1, day),
                open=base,
                high=base + 3,
                low=base - 3,
                close=base,
                volume=10000,
            )
        )
    db.commit()
    return create_session(
        db, instrument, date(2023, 1, 1), date(2023, 1, 6), **kwargs
    ).session_id


def order(db, sid, **kwargs):
    values = dict(
        session_id=sid,
        side="buy",
        order_type="market",
        quantity=10,
        rationale="Independent financial regression test",
    )
    values.update(kwargs)
    return submit_order(db, CreateOrderRequest(**values))


def test_no_same_bar_profit_or_fill(db_session):
    sid = setup(db_session)
    order(db_session, sid)
    assert db_session.query(FillORM).count() == 0
    assert current_snapshot(db_session, sid).equity == 100000
    step_session(db_session, sid)
    fill = db_session.query(FillORM).one()
    assert fill.fill_bar == 1
    assert current_snapshot(db_session, sid).cash == 99000
    assert current_snapshot(db_session, sid).equity == 100000


def test_flat_roundtrip_loses_spread_and_fees_once(db_session):
    sid = setup(db_session, fee_per_unit=0.1, spread_bps=20)
    order(db_session, sid)
    step_session(db_session, sid)
    snap = current_snapshot(db_session, sid)
    assert snap.cash == pytest.approx(98998)  # 1001 notional + 1 fee
    assert snap.equity == pytest.approx(99998)  # asset marked at mid
    order(db_session, sid, side="sell")
    step_session(db_session, sid)
    snap = current_snapshot(db_session, sid)
    assert snap.cash == snap.equity == pytest.approx(99996)
    assert snap.total_realized_pnl == pytest.approx(-2)
    assert snap.total_fees == pytest.approx(2)


def test_future_does_not_pay_notional_and_uses_multiplier(db_session):
    sid = setup(db_session, "ES=F", 5000, initial_cash=1000000, fee_per_unit=2.5)
    assert db_session.get(ReplaySessionORM, sid).multiplier == 50
    order(db_session, sid, quantity=1, stop_loss=4990)
    step_session(db_session, sid)
    snap = current_snapshot(db_session, sid)
    assert snap.cash == snap.equity == pytest.approx(999997.5)
    assert snap.positions[0].multiplier == 50


def test_fake_future_multiplier_rejected(db_session):
    setup(db_session, "ES=F", 5000)
    with pytest.raises(ValueError, match="requires multiplier 50"):
        create_session(
            db_session, "ES=F", date(2023, 1, 1), date(2023, 1, 6), multiplier=1
        )


def test_short_liability_is_not_profit(db_session):
    sid = setup(db_session)
    order(db_session, sid, side="sell", stop_loss=110)
    step_session(db_session, sid)
    snap = current_snapshot(db_session, sid)
    assert snap.cash == 101000 and snap.equity == 100000


def test_sells_cannot_stack_past_remaining_position(db_session):
    sid = setup(db_session)
    order(db_session, sid)
    step_session(db_session, sid)
    order(db_session, sid, side="sell")
    with pytest.raises(OrderRejectedError, match="pending order"):
        order(db_session, sid, side="sell")


def test_stop_and_manual_close_cannot_reverse_position(db_session):
    sid = setup(db_session)
    parent = order(db_session, sid, stop_loss=95)
    step_session(db_session, sid)
    order(db_session, sid, side="sell")
    bar = db_session.query(OHLCVBarORM).filter_by(date=date(2023, 1, 3)).one()
    bar.open = 94
    bar.low = 90
    db_session.commit()
    step_session(db_session, sid)
    snap = current_snapshot(db_session, sid)
    assert all(p.side == "flat" for p in snap.positions)
    assert snap.total_realized_pnl == -60
    assert (
        db_session.query(OrderORM)
        .filter_by(parent_order_id=parent.order_id)
        .one()
        .status
        == "filled"
    )


def test_limit_entry_creates_protection_and_oco_adverse_first(db_session):
    sid = setup(db_session)
    parent = order(
        db_session,
        sid,
        order_type="limit",
        limit_price=100,
        stop_loss=98,
        take_profit=102,
    )
    step_session(db_session, sid)
    children = (
        db_session.query(OrderORM).filter_by(parent_order_id=parent.order_id).all()
    )
    assert len(children) == 2 and all(c.status == "pending" for c in children)
    step_session(db_session, sid)
    assert next(c for c in children if c.order_type == "stop").status == "filled"
    assert next(c for c in children if c.order_type == "limit").status == "cancelled"
    assert current_snapshot(db_session, sid).total_realized_pnl == -20


def test_synthetic_limit_does_not_violate_limit_price(db_session):
    sid = setup(db_session, spread_bps=20)
    parent = order(db_session, sid, order_type="limit", limit_price=99)
    step_session(db_session, sid)
    assert (
        db_session.query(FillORM).filter_by(order_id=parent.order_id).one().fill_price
        == 99
    )


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"side": "sell"}, "planned stop"),
        ({"stop_loss": 105}, "Stop loss"),
        ({"quantity": 100}, "risk limits"),
    ],
)
def test_unbounded_and_hidden_risks_are_rejected(db_session, changes, message):
    sid = setup(db_session)
    with pytest.raises(OrderRejectedError, match=message):
        order(db_session, sid, **changes)


def test_reduce_allowed_during_risk_pause(db_session):
    sid = setup(db_session)
    order(db_session, sid)
    step_session(db_session, sid)
    db_session.get(ReplaySessionORM, sid).status = "paused"
    db_session.commit()
    order(db_session, sid, side="sell")
    step_session(db_session, sid)
    assert all(p.side == "flat" for p in current_snapshot(db_session, sid).positions)


def test_legacy_session_cannot_mix_accounting_rules(db_session):
    sid = setup(db_session)
    db_session.get(ReplaySessionORM, sid).simulation_version = 1
    db_session.commit()
    with pytest.raises(ValueError, match="Legacy"):
        step_session(db_session, sid)
    with pytest.raises(OrderRejectedError, match="Legacy"):
        order(db_session, sid)


def test_completed_session_does_not_reopen(db_session):
    sid = setup(db_session)
    step_session(db_session, sid, n=5)
    assert jump_to_bar(db_session, sid, 5).status == "completed"


def test_quarantine_preserves_existing_mislabeled_wti(db_session):
    from src.services.data_loader import has_quarantined_wti

    setup(db_session)
    for row in db_session.query(OHLCVBarORM).all():
        db_session.add(
            OHLCVBarORM(
                instrument="CL=F",
                date=row.date,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
            )
        )
    db_session.commit()
    assert has_quarantined_wti(db_session)
    with pytest.raises(ValueError, match="quarantined"):
        create_session(db_session, "CL=F", date(2023, 1, 1), date(2023, 1, 6))
    assert db_session.query(OHLCVBarORM).filter_by(instrument="CL=F").count() == 6


