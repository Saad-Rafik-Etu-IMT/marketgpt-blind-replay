"""Order service.

Public entry points:
  * ``submit_order(db, request)`` — validate and queue for a future bar.
  * ``cancel_order(db, order_id)`` — set a pending order to cancelled.
  * ``list_orders(db, session_id)`` — listing for the routes layer.

When a parent order *fills* with stop_loss and/or take_profit set, this
service creates the protective children:
  * SL: stop order on the opposite side at the stop_loss price.
  * TP: limit order on the opposite side at the take_profit price.
When both children exist, they share an oco_pair_id so the broker
auto-cancels the sibling on the first fill.

Pre-trade checks estimate risk using visible prices; gaps can exceed estimates.
"""

import logging
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.fill import FillORM
from src.models.order import OrderORM
from src.models.session import ReplaySessionORM
from src.schemas.order import CreateOrderRequest, OrderRead

logger = logging.getLogger(__name__)


class OrderRejectedError(ValueError):
    """The order failed pre-trade validation."""


def submit_order(db: Session, request: CreateOrderRequest) -> OrderRead:
    """Persist a new pending order; market execution is at the next open.

    Pre-trade gate (Phase 4):
      1. Session must exist and be active.
      2. Risk limits (drawdown, per-trade risk, exposure, max-open)
         must not be breached by the new trade.
    On breach the order is rejected — nothing is persisted.
    """
    session_orm = db.get(ReplaySessionORM, request.session_id)
    if session_orm is None:
        raise OrderRejectedError(f"session not found: {request.session_id}")
    if session_orm.simulation_version < 2:
        raise OrderRejectedError(
            "Legacy simulation: create a new session with corrected execution and accounting"
        )
    if session_orm.status not in ("active", "paused"):
        raise OrderRejectedError(
            f"cannot submit orders to session with status {session_orm.status!r}"
        )

    if not request.rationale.strip():
        raise OrderRejectedError("Explain the decision before submitting it.")

    # Risk pre-check. Imported lazily to avoid a module-level cycle.
    from src.services.risk_check import evaluate_pre_trade

    result = evaluate_pre_trade(db, request, session_orm)
    if not result.allowed:
        breach_summary = "; ".join(b.detail for b in result.breaches)
        logger.warning(
            "order rejected: session=%s reason=%s",
            request.session_id,
            breach_summary,
        )
        raise OrderRejectedError(f"risk limits breached: {breach_summary}")

    from src.models.portfolio import PositionORM
    key = session_orm.instrument
    position = db.scalar(
        select(PositionORM).where(
            PositionORM.session_id == request.session_id,
            PositionORM.contract_key == key,
        )
    )
    reducing = bool(
        position
        and position.side != "flat"
        and (position.side == "long") != (request.side == "buy")
        and request.quantity <= position.quantity
    )
    order = OrderORM(
        order_id=str(uuid4()),
        session_id=request.session_id,
        side=request.side,
        order_type=request.order_type,
        quantity=request.quantity,
        limit_price=request.limit_price,
        stop_price=request.stop_price,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        status="pending",
        reduce_only=reducing,
        submitted_at_bar=session_orm.current_bar_index,
        submitted_at=datetime.now(timezone.utc),
        rationale=request.rationale,
        instrument_kind=request.instrument_kind,
    )
    db.add(order)
    db.commit()

    return _to_read(order)


def cancel_order(db: Session, order_id: str) -> OrderRead:
    order = db.get(OrderORM, order_id)
    if order is None:
        raise LookupError(f"order not found: {order_id}")
    if order.status != "pending":
        raise OrderRejectedError(f"cannot cancel order in status {order.status!r}")
    order.status = "cancelled"
    db.commit()
    return _to_read(order)


def list_orders(db: Session, session_id: str) -> list[OrderRead]:
    rows = (
        db.execute(
            select(OrderORM)
            .where(OrderORM.session_id == session_id)
            .order_by(OrderORM.submitted_at_bar, OrderORM.submitted_at)
        )
        .scalars()
        .all()
    )
    return [_to_read(o) for o in rows]


# --- OCO -------------------------------------------------------------------


def _create_protective_children(db: Session, parent: OrderORM) -> None:
    """Create SL/TP children for a freshly-filled parent.

    Children are submitted at the parent's *fill* bar so they cannot
    accidentally fill on that same bar (limit/stop orders require
    current_bar > submitted_at_bar per the fill rules).
    """
    fill_bar = _fill_bar_for(db, parent)
    opposite_side = "sell" if parent.side == "buy" else "buy"

    children: list[OrderORM] = []
    if parent.stop_loss is not None:
        children.append(
            OrderORM(
                order_id=str(uuid4()),
                session_id=parent.session_id,
                side=opposite_side,
                order_type="stop",
                quantity=parent.quantity,
                stop_price=parent.stop_loss,
                status="pending",
                submitted_at_bar=fill_bar,
                submitted_at=datetime.now(timezone.utc),
                rationale=f"auto SL for parent {parent.order_id}",
                parent_order_id=parent.order_id,
                reduce_only=True,
            )
        )
    if parent.take_profit is not None:
        children.append(
            OrderORM(
                order_id=str(uuid4()),
                session_id=parent.session_id,
                side=opposite_side,
                order_type="limit",
                quantity=parent.quantity,
                limit_price=parent.take_profit,
                status="pending",
                submitted_at_bar=fill_bar,
                submitted_at=datetime.now(timezone.utc),
                rationale=f"auto TP for parent {parent.order_id}",
                parent_order_id=parent.order_id,
                reduce_only=True,
            )
        )

    # Pair the children only when BOTH exist — a lone child has no sibling.
    if len(children) == 2:
        pair_id = str(uuid4())
        for child in children:
            child.oco_pair_id = pair_id

    for child in children:
        db.add(child)


def _fill_bar_for(db: Session, order: OrderORM) -> int:
    """Latest fill_bar for an order — used to time-stamp its children."""
    fill = db.execute(
        select(FillORM)
        .where(FillORM.order_id == order.order_id)
        .order_by(FillORM.fill_bar.desc())
        .limit(1)
    ).scalar_one_or_none()
    if fill is None:
        # Defensive: a filled order without a fill row shouldn't happen,
        # but if it does, fall back to the order's submission bar.
        return order.submitted_at_bar
    return fill.fill_bar


# --- Conversion -----------------------------------------------------------


def _to_read(orm: OrderORM) -> OrderRead:
    return OrderRead(
        order_id=orm.order_id,
        session_id=orm.session_id,
        side=orm.side,  # type: ignore[arg-type]
        order_type=orm.order_type,  # type: ignore[arg-type]
        quantity=orm.quantity,
        limit_price=orm.limit_price,
        stop_price=orm.stop_price,
        stop_loss=orm.stop_loss,
        take_profit=orm.take_profit,
        status=orm.status,  # type: ignore[arg-type]
        rejection_reason=orm.rejection_reason,  # type: ignore[arg-type]
        submitted_at_bar=orm.submitted_at_bar,
        submitted_at=orm.submitted_at,
        rationale=orm.rationale,
        parent_order_id=orm.parent_order_id,
        oco_pair_id=orm.oco_pair_id,
        instrument_kind=orm.instrument_kind,  # type: ignore[arg-type]
    )
