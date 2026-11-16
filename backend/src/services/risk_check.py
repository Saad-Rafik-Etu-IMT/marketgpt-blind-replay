"""Pre-trade risk evaluation — bridges DB state and the pure check.

Computes ``RiskCheckInputs`` from the session's latest snapshot and
proposed order, then delegates to ``core.risk_limits.check_risk_limits``.

Convention for ``proposed_risk`` (the dollar amount at risk):

  * If ``stop_loss`` is set: ``qty * |entry - stop_loss|`` — the
    canonical pro convention.
  * Otherwise: use the full cash outlay for an unlevered long.
    Short and futures orders without stops are rejected by the caller.

All dollar exposures and risks include the contract multiplier.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.risk_limits import RiskCheckInputs, RiskCheckResult, check_risk_limits
from src.models.market import OHLCVBarORM
from src.models.portfolio import PortfolioSnapshotORM, PositionORM
from src.models.session import ReplaySessionORM
from src.schemas.order import CreateOrderRequest


def evaluate_pre_trade(
    db: Session,
    request: CreateOrderRequest,
    session_orm: ReplaySessionORM,
) -> RiskCheckResult:
    """Build ``RiskCheckInputs`` from current state and evaluate."""
    snap = db.execute(
        select(PortfolioSnapshotORM)
        .where(PortfolioSnapshotORM.session_id == request.session_id)
        .order_by(PortfolioSnapshotORM.bar_index.desc())
        .limit(1)
    ).scalar_one_or_none()
    equity = snap.equity if snap is not None else session_orm.initial_cash
    current_dd = snap.drawdown_pct if snap is not None else 0.0

    entry_price = _entry_price(db, request, session_orm)

    from src.models.order import OrderORM
    from src.services.order_service import OrderRejectedError
    from src.services.risk_service import core_limits_for

    key = session_orm.instrument
    position = db.scalar(
        select(PositionORM).where(
            PositionORM.session_id == request.session_id,
            PositionORM.contract_key == key,
        )
    )
    signed = (
        position.quantity * (1 if position.side == "long" else -1)
        if position and position.side != "flat"
        else 0
    )
    trade = request.quantity * (1 if request.side == "buy" else -1)
    reducing = signed * trade < 0 and abs(trade) <= abs(signed)
    pending = db.scalar(
        select(OrderORM).where(
            OrderORM.session_id == request.session_id,
            OrderORM.status == "pending",
            OrderORM.parent_order_id.is_(None),
        )
    )
    if pending is not None:
        raise OrderRejectedError(
            "Execute or cancel the pending order before submitting another"
        )
    if reducing:
        # Exits must not be blocked by risk limits; the broker caps their size
        # to what remains if a protective child executes first.
        return RiskCheckResult(allowed=True)
    if session_orm.status == "paused":
        raise OrderRejectedError("Risk pause: only reductions are allowed")

    open_count = _count_open_positions(db, request.session_id) - (1 if signed else 0)
    current_exposure = _exposure(db, request.session_id)
    mult = session_orm.multiplier
    proposed_notional = request.quantity * abs(entry_price) * mult
    if request.stop_loss is None:
        if request.side == "sell" or session_orm.instrument.endswith("=F"):
            raise OrderRejectedError(
                "A planned stop is required for short/futures risk. It is not a guaranteed maximum loss"
            )
        proposed_risk = proposed_notional  # cash long can lose its full value
    else:
        valid_stop = (
            request.stop_loss < entry_price
            if request.side == "buy"
            else request.stop_loss > entry_price
        )
        if not valid_stop:
            raise OrderRejectedError(
                "Stop loss must be below a buy entry or above a sell entry"
            )
        proposed_risk = _proposed_risk(request, entry_price) * mult
    if request.take_profit is not None:
        valid_target = (
            request.take_profit > entry_price
            if request.side == "buy"
            else request.take_profit < entry_price
        )
        if not valid_target:
            raise OrderRejectedError(
                "Take profit must be beyond the entry in the profitable direction"
            )
    proposed_risk += 2 * request.quantity * session_orm.fee_per_unit
    proposed_risk += proposed_notional * session_orm.spread_bps / 10000
    cash = snap.cash if snap is not None else session_orm.initial_cash
    cash_purchase = request.side == "buy" and not session_orm.instrument.endswith("=F")
    estimated_outlay = (
        proposed_notional * (1 + session_orm.spread_bps / 20000)
        + request.quantity * session_orm.fee_per_unit
    )
    if cash_purchase and estimated_outlay > cash:
        raise OrderRejectedError(
            "Insufficient cash at the indicative quote; cash borrowing is not modeled"
        )
    limits = core_limits_for(db, request.session_id)

    return check_risk_limits(
        RiskCheckInputs(
            equity=equity,
            current_drawdown_pct=current_dd,
            open_position_count=open_count,
            current_exposure=current_exposure,
            proposed_notional=proposed_notional,
            proposed_risk=proposed_risk,
        ),
        limits,
    )


def _entry_price(
    db: Session,
    request: CreateOrderRequest,
    session_orm: ReplaySessionORM,
) -> float:
    """Estimate the entry price for risk calculation purposes.

    Market orders: the current visible close, not the unknown next opening.
    Limit/stop:    the limit_price or stop_price as supplied.
    """
    if request.order_type == "market":
        bars = (
            db.execute(
                select(OHLCVBarORM)
                .where(OHLCVBarORM.instrument == session_orm.instrument)
                .where(OHLCVBarORM.date >= session_orm.start_date)
                .where(OHLCVBarORM.date <= session_orm.end_date)
                .order_by(OHLCVBarORM.date)
            )
            .scalars()
            .all()
        )
        idx = session_orm.current_bar_index
        if 0 <= idx < len(bars):
            return bars[idx].close
        return 0.0
    if request.order_type == "limit":
        return request.limit_price or 0.0
    return request.stop_price or 0.0


def _count_open_positions(db: Session, session_id: str) -> int:
    return (
        db.query(PositionORM)
        .filter(PositionORM.session_id == session_id)
        .filter(PositionORM.side != "flat")
        .count()
    )


def _exposure(db: Session, session_id: str) -> float:
    from src.services.portfolio_service import _marks_at_bar

    session = db.get(ReplaySessionORM, session_id)
    marks = _marks_at_bar(db, session_id, session.current_bar_index)
    positions = (
        db.execute(
            select(PositionORM)
            .where(PositionORM.session_id == session_id)
            .where(PositionORM.side != "flat")
        )
        .scalars()
        .all()
    )
    return sum(
        p.quantity
        * abs(marks.get(p.contract_key or p.instrument, p.avg_entry_price))
        * p.multiplier
        for p in positions
    )


def _proposed_risk(request: CreateOrderRequest, entry_price: float) -> float:
    if request.stop_loss is not None:
        return request.quantity * abs(entry_price - request.stop_loss)
    # Unlevered cash long: full outlay, never an invented implicit stop.
    return request.quantity * abs(entry_price)


