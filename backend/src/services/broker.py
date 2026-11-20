"""The broker — fill engine.

Walks pending orders for a session and applies the pure
``core.fill_logic.decide_fill`` to each one against the current bar.
Persists fills and transitions order status. When a child of an OCO
pair fills, its sibling is auto-cancelled.

The broker reads ``current_bar_index`` via the replay engine. It never
mutates that index — only ``replay_engine.step_session`` is permitted
to (§5 invariant 6).
"""

import logging
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.fill_logic import OrderInputs, cash_after_cash_buy, decide_fill
from src.models.fill import FillORM
from src.models.market import OHLCVBarORM
from src.models.order import OrderORM
from src.models.session import ReplaySessionORM
from src.schemas.market import OHLCVBar
from src.schemas.order import INSUFFICIENT_CASH
from src.services.bar_queries import bar_at

logger = logging.getLogger(__name__)


def process_pending_orders(
    db: Session, session_id: str, *, commit: bool = True
) -> list[FillORM]:
    """Process every pending order for ``session_id`` against the current bar.

    Returns the list of fills created in this pass. Quietly does nothing
    if the session has no pending orders.

    ``commit`` lets the replay engine keep the whole bar (index advance,
    fills, snapshot) inside a single transaction: it passes ``commit=False``
    and commits once itself, so a mid-bar failure rolls everything back
    instead of leaving fills without a matching snapshot.
    """
    session_orm = db.get(ReplaySessionORM, session_id)
    if session_orm is None:
        return []

    current_index = session_orm.current_bar_index
    bar = bar_at(db, session_orm, current_index)
    if bar is None:
        return []  # defensive — replay_engine should prevent this
    current_bar = _to_schema(bar)

    pending = (
        db.execute(
            select(OrderORM)
            .where(OrderORM.session_id == session_id)
            .where(OrderORM.status == "pending")
        )
        .scalars()
        .all()
    )

    new_fills: list[FillORM] = []
    rejected = False
    pending.sort(
        key=lambda o: (0 if o.order_type == "stop" else 1, o.submitted_at, o.order_id)
    )
    for order in pending:
        fill = _try_fill(db, order, current_bar, current_index)
        rejected = rejected or order.status == "rejected"
        if fill is not None:
            new_fills.append(fill)
            logger.info(
                "order filled: session=%s order=%s side=%s qty=%s price=%.4f bar=%d",
                session_id,
                order.order_id,
                order.side,
                fill.fill_quantity,
                fill.fill_price,
                current_index,
            )

    if (new_fills or rejected) and commit:
        db.commit()
    return new_fills


def _try_fill(
    db: Session, order: OrderORM, bar: OHLCVBar, current_bar: int
) -> FillORM | None:
    """Decide and persist a fill for one order, returning the FillORM or None.

    Re-checks ``order.status`` first: an earlier iteration of the loop
    may have cancelled this order via OCO. Without this guard a
    cancelled order could be revived to "filled" within the same pass.
    """
    if order.status != "pending":
        return None

    if order.parent_order_id is not None or order.reduce_only:
        from src.models.portfolio import PositionORM
        session = db.get(ReplaySessionORM, order.session_id)
        key = session.instrument
        position = db.scalar(
            select(PositionORM).where(
                PositionORM.session_id == order.session_id,
                PositionORM.contract_key == key,
            )
        )
        expected_side = "long" if order.side == "sell" else "short"
        if position is None or position.side != expected_side or position.quantity == 0:
            order.status = "cancelled"
            return None
        order.quantity = min(order.quantity, position.quantity)

    decision = decide_fill(
        OrderInputs(
            side=order.side,  # type: ignore[arg-type]
            order_type=order.order_type,  # type: ignore[arg-type]
            limit_price=order.limit_price,
            stop_price=order.stop_price,
        ),
        bar,
        submitted_at_bar=order.submitted_at_bar,
        current_bar=current_bar,
    )
    if not decision.should_fill or decision.fill_price is None:
        return None

    fill_price = decision.fill_price
    reference = order.limit_price or order.stop_price or decision.fill_price
    slippage = decision.fill_price - reference

    session_orm = db.get(ReplaySessionORM, order.session_id)
    # Synthetic bid/ask around the reference. Not observed market quotes.
    half_spread = abs(fill_price) * session_orm.spread_bps / 20000.0
    executable = fill_price + (half_spread if order.side == "buy" else -half_spread)
    if order.order_type == "limit":
        factor = 1 + (session_orm.spread_bps / 20000.0) * (
            1 if order.side == "buy" else -1
        )
        if order.side == "buy":
            if bar.low * factor > order.limit_price:
                return None
            executable = min(bar.open * factor, order.limit_price)
        else:
            if bar.high * factor < order.limit_price:
                return None
            executable = max(bar.open * factor, order.limit_price)
    slippage += executable - fill_price
    fill_price = max(0.0, executable)

    fee = order.quantity * session_orm.fee_per_unit
    if _requires_cash_check(order, session_orm):
        from src.services.portfolio_service import _latest_cash

        # Latest snapshot cash already includes earlier fills in this bar pass.
        cash = _latest_cash(db, order.session_id, session_orm.initial_cash)
        remaining = cash_after_cash_buy(
            cash, order.quantity, fill_price, session_orm.multiplier, fee
        )
        if remaining < 0:
            # Atomic rejection: no fill, cash, position or protective children.
            order.status = "rejected"
            order.rejection_reason = INSUFFICIENT_CASH
            logger.warning(
                "order rejected at fill: session=%s order=%s reason=%s "
                "required=%.4f available=%.4f bar=%d",
                order.session_id,
                order.order_id,
                INSUFFICIENT_CASH,
                cash - remaining,
                cash,
                current_bar,
            )
            return None

    fill = FillORM(
        fill_id=str(uuid4()),
        order_id=order.order_id,
        fill_price=fill_price,
        fill_quantity=order.quantity,
        fill_bar=current_bar,
        slippage=slippage,
        fee=fee,
        timestamp=datetime.now(timezone.utc),
    )
    db.add(fill)
    order.status = "filled"

    # Update positions/cash/snapshot for this fill.
    from src.services.portfolio_service import record_fill as _record_fill

    _record_fill(db, fill)

    if (
        order.parent_order_id is None
        and order.instrument_kind == "underlying"
        and (order.stop_loss or order.take_profit)
    ):
        from src.services.order_service import _create_protective_children

        _create_protective_children(db, order)
        db.flush()
        # Market entry happens at the open: both protective levels may be
        # touched later in this bar. Choose stop first if OHLC is ambiguous.
        if order.order_type == "market":
            children = list(
                db.scalars(
                    select(OrderORM).where(OrderORM.parent_order_id == order.order_id)
                )
            )
            children.sort(key=lambda c: 0 if c.order_type == "stop" else 1)
            for child in children:
                child.submitted_at_bar = current_bar - 1
                _try_fill(db, child, bar, current_bar)

    # OCO: when a child fills, cancel its sibling.
    if order.oco_pair_id is not None:
        _cancel_oco_sibling(db, order)

    return fill


def _requires_cash_check(order: OrderORM, session: ReplaySessionORM) -> bool:
    """Opening/increasing cash-instrument buys only.

    Exits (reduce-only orders and protective children) are never blocked;
    futures margin and short-sale accounting are outside this check.
    """
    return (
        order.side == "buy"
        and not order.reduce_only
        and order.parent_order_id is None
        and not session.instrument.endswith("=F")
    )


def _cancel_oco_sibling(db: Session, order: OrderORM) -> None:
    """Cancel any other pending order sharing ``order.oco_pair_id``."""
    siblings = (
        db.execute(
            select(OrderORM)
            .where(OrderORM.oco_pair_id == order.oco_pair_id)
            .where(OrderORM.order_id != order.order_id)
            .where(OrderORM.status == "pending")
        )
        .scalars()
        .all()
    )
    for sibling in siblings:
        sibling.status = "cancelled"


def _to_schema(orm: OHLCVBarORM) -> OHLCVBar:
    return OHLCVBar(
        date=orm.date,
        open=orm.open,
        high=orm.high,
        low=orm.low,
        close=orm.close,
        volume=orm.volume,
        adjusted_close=orm.adjusted_close,
    )
