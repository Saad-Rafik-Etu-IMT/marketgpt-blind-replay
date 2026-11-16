"""Portfolio service.

Maintains per-session positions and emits a portfolio snapshot after
every state change. The service is the *only* writer of position and
snapshot rows — the broker calls into it, never bypasses it.

Snapshots are taken:
  * Once at session creation (initial snapshot at bar 0).
  * Once after every step (the new current bar's close becomes the mark).
  * Once after every fill the broker creates within a step.

The result is an equity curve indexed by bar_index that is read by the
risk service for drawdown computation.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.pnl import (
    PositionState,
    apply_trade,
    compute_equity,
    unrealized_pnl,
)
from src.core.risk_metrics import max_drawdown
from src.models.fill import FillORM
from src.models.market import OHLCVBarORM
from src.models.order import OrderORM
from src.models.portfolio import PortfolioSnapshotORM, PositionORM
from src.models.session import ReplaySessionORM
from src.schemas.portfolio import (
    PortfolioSnapshotRead,
    PositionRead,
)

# --- Public API ------------------------------------------------------------


def initialize_session(db: Session, session_id: str) -> PortfolioSnapshotRead:
    """Emit the initial snapshot for a fresh session (bar 0, cash = initial)."""
    session_orm = _require_session(db, session_id)
    snap = _persist_snapshot(
        db,
        session_id=session_id,
        bar_index=session_orm.current_bar_index,
        cash=session_orm.initial_cash,
        positions={},
        instruments_close={},
    )
    db.commit()
    return snap


def record_fill(db: Session, fill: FillORM) -> None:
    """Apply a fill to the session's position and cash.

    Called by the broker immediately after creating a Fill row. Does
    NOT emit a fresh snapshot — the caller (broker/replay engine)
    drives snapshotting after the bar is fully processed, so each bar
    yields exactly one snapshot row.
    """
    order = db.get(OrderORM, fill.order_id)
    if order is None:
        return  # defensive — orphan fill should never happen

    session_orm = _require_session(db, order.session_id)
    multiplier = session_orm.multiplier
    key = session_orm.instrument
    position = _load_or_init_position(db, session_orm, order, key, multiplier)
    state = _position_state(position)
    new_state = apply_trade(
        state, side=order.side, quantity=fill.fill_quantity, price=fill.fill_price
    )

    # Notional moves cash by quantity * price * multiplier (point value).
    notional = fill.fill_quantity * fill.fill_price * multiplier
    cash_delta = -notional if order.side == "buy" else notional
    if state.futures_style:
        cash_delta = new_state.realized_pnl - state.realized_pnl
    cash_delta -= fill.fee or 0.0

    # Apply cash to the latest snapshot's cash value — held in the
    # latest snapshot row. If there's no snapshot yet, fall back to
    # initial_cash. Cash is always read from the most recent snapshot.
    latest_cash = _latest_cash(db, order.session_id, session_orm.initial_cash)
    new_cash = latest_cash + cash_delta

    # Persist position changes.
    _save_position(position, new_state)
    # Carry the latest cash forward by stamping it into a new snapshot
    # at the current bar. (The bar-level snapshot in apply_step_snapshot
    # may overwrite this when called separately.)
    bar_index = session_orm.current_bar_index
    _upsert_snapshot(
        db,
        session_id=order.session_id,
        bar_index=bar_index,
        cash=new_cash,
        positions=_all_positions_states(db, order.session_id),
        instruments_close=_marks_at_bar(db, order.session_id, bar_index),
    )


def apply_step_snapshot(db: Session, session_id: str) -> None:
    """Snapshot the portfolio at the session's current bar.

    Called by the replay engine after each step. Marks every open
    position to the current bar's close.
    """
    session_orm = _require_session(db, session_id)
    bar_index = session_orm.current_bar_index
    cash = _latest_cash(db, session_id, session_orm.initial_cash)
    _upsert_snapshot(
        db,
        session_id=session_id,
        bar_index=bar_index,
        cash=cash,
        positions=_all_positions_states(db, session_id),
        instruments_close=_marks_at_bar(db, session_id, bar_index),
    )


def current_snapshot(db: Session, session_id: str) -> PortfolioSnapshotRead:
    """Return the most recent snapshot for ``session_id``."""
    snap = db.execute(
        select(PortfolioSnapshotORM)
        .where(PortfolioSnapshotORM.session_id == session_id)
        .order_by(PortfolioSnapshotORM.bar_index.desc())
        .limit(1)
    ).scalar_one_or_none()
    if snap is None:
        # Defensive: caller initialised but somehow no snapshot exists.
        return _empty_snapshot(db, session_id)
    return _to_read(db, snap)


def equity_curve(db: Session, session_id: str) -> list[PortfolioSnapshotRead]:
    """Return every snapshot for ``session_id`` ordered by bar_index.

    Historical snapshots are immutable: their cash, equity, realized and
    unrealized totals, and drawdown are persisted on the row. The equity
    curve only needs those aggregate fields, so this reads them directly.
    Per-position detail is served by ``current_snapshot`` for the live bar.
    """
    rows = (
        db.execute(
            select(PortfolioSnapshotORM)
            .where(PortfolioSnapshotORM.session_id == session_id)
            .order_by(PortfolioSnapshotORM.bar_index)
        )
        .scalars()
        .all()
    )
    return [_to_read_aggregate(r) for r in rows]


# --- Internals -------------------------------------------------------------


def _require_session(db: Session, session_id: str) -> ReplaySessionORM:
    orm = db.get(ReplaySessionORM, session_id)
    if orm is None:
        raise LookupError(f"session not found: {session_id}")
    return orm


def _load_or_init_position(
    db: Session,
    session_orm: ReplaySessionORM,
    order: OrderORM,
    key: str,
    multiplier: float,
) -> PositionORM:
    pos = db.execute(
        select(PositionORM)
        .where(PositionORM.session_id == session_orm.session_id)
        .where(PositionORM.contract_key == key)
    ).scalar_one_or_none()
    if pos is not None:
        return pos
    pos = PositionORM(
        session_id=session_orm.session_id,
        instrument=session_orm.instrument,
        contract_key=key,
        instrument_kind=order.instrument_kind,
        side="flat",
        quantity=0,
        avg_entry_price=0.0,
        realized_pnl=0.0,
        multiplier=multiplier,
    )
    db.add(pos)
    db.flush()
    return pos


def _position_state(pos: PositionORM) -> PositionState:
    return PositionState(
        instrument=pos.contract_key or pos.instrument,
        side=pos.side,  # type: ignore[arg-type]
        quantity=pos.quantity,
        avg_entry_price=pos.avg_entry_price,
        realized_pnl=pos.realized_pnl,
        multiplier=getattr(pos, "multiplier", 1.0) or 1.0,
        futures_style=pos.instrument_kind == "underlying"
        and pos.instrument.endswith("=F"),
    )


def _save_position(pos: PositionORM, state: PositionState) -> None:
    pos.side = state.side
    pos.quantity = state.quantity
    pos.avg_entry_price = state.avg_entry_price
    pos.realized_pnl = state.realized_pnl


def _all_positions_states(db: Session, session_id: str) -> dict[str, PositionState]:
    rows = (
        db.execute(select(PositionORM).where(PositionORM.session_id == session_id))
        .scalars()
        .all()
    )
    return {(p.contract_key or p.instrument): _position_state(p) for p in rows}


def _latest_cash(db: Session, session_id: str, fallback: float) -> float:
    snap = db.execute(
        select(PortfolioSnapshotORM.cash)
        .where(PortfolioSnapshotORM.session_id == session_id)
        .order_by(PortfolioSnapshotORM.bar_index.desc())
        .limit(1)
    ).scalar_one_or_none()
    return snap if snap is not None else fallback


def _marks_at_bar(db: Session, session_id: str, bar_index: int) -> dict[str, float]:
    """Map the session instrument to its close at the requested bar."""
    session_orm = _require_session(db, session_id)
    bars = session_bars(db, session_orm)
    if not (0 <= bar_index < len(bars)):
        return {}
    bar = bars[bar_index]
    return {session_orm.instrument: bar.close}


def session_bars(db: Session, session_orm: ReplaySessionORM) -> list[OHLCVBarORM]:
    """Ordered OHLCV bars spanning the session's window."""
    return list(
        db.execute(
            select(OHLCVBarORM)
            .where(OHLCVBarORM.instrument == session_orm.instrument)
            .where(OHLCVBarORM.date >= session_orm.start_date)
            .where(OHLCVBarORM.date <= session_orm.end_date)
            .order_by(OHLCVBarORM.date)
        ).scalars()
    )


def _upsert_snapshot(
    db: Session,
    *,
    session_id: str,
    bar_index: int,
    cash: float,
    positions: dict[str, PositionState],
    instruments_close: dict[str, float],
) -> PortfolioSnapshotRead:
    """Insert or update the snapshot row at (session, bar_index)."""
    existing = db.execute(
        select(PortfolioSnapshotORM)
        .where(PortfolioSnapshotORM.session_id == session_id)
        .where(PortfolioSnapshotORM.bar_index == bar_index)
    ).scalar_one_or_none()

    equity = compute_equity(cash, positions, instruments_close)
    realized_total = sum(p.realized_pnl for p in positions.values())
    unrealized_total = sum(
        unrealized_pnl(p, instruments_close.get(inst, p.avg_entry_price))
        for inst, p in positions.items()
    )
    dd_pct = _drawdown_at(db, session_id, equity, bar_index=bar_index)

    if existing is None:
        return _persist_snapshot(
            db,
            session_id=session_id,
            bar_index=bar_index,
            cash=cash,
            positions=positions,
            instruments_close=instruments_close,
            realized_total=realized_total,
            unrealized_total=unrealized_total,
            equity=equity,
            drawdown_pct=dd_pct,
        )

    existing.cash = cash
    existing.equity = equity
    existing.total_realized_pnl = realized_total
    existing.total_unrealized_pnl = unrealized_total
    existing.drawdown_pct = dd_pct
    db.flush()
    return _to_read(db, existing)


def _persist_snapshot(
    db: Session,
    *,
    session_id: str,
    bar_index: int,
    cash: float,
    positions: dict[str, PositionState],
    instruments_close: dict[str, float],
    realized_total: float | None = None,
    unrealized_total: float | None = None,
    equity: float | None = None,
    drawdown_pct: float | None = None,
) -> PortfolioSnapshotRead:
    if equity is None:
        equity = compute_equity(cash, positions, instruments_close)
    if realized_total is None:
        realized_total = sum(p.realized_pnl for p in positions.values())
    if unrealized_total is None:
        unrealized_total = sum(
            unrealized_pnl(p, instruments_close.get(inst, p.avg_entry_price))
            for inst, p in positions.items()
        )
    if drawdown_pct is None:
        drawdown_pct = _drawdown_at(db, session_id, equity, bar_index=bar_index)

    orm = PortfolioSnapshotORM(
        session_id=session_id,
        bar_index=bar_index,
        cash=cash,
        equity=equity,
        total_realized_pnl=realized_total,
        total_unrealized_pnl=unrealized_total,
        drawdown_pct=drawdown_pct,
        created_at=datetime.now(timezone.utc),
    )
    db.add(orm)
    db.flush()
    return _to_read(db, orm)


def _drawdown_at(
    db: Session, session_id: str, current_equity: float, bar_index: int | None = None
) -> float:
    """Drawdown at the current bar — uses persisted prior snapshots plus the new one.

    When ``bar_index`` is provided, any pre-existing snapshot at that
    bar is excluded from the "prior" series so an update doesn't
    double-count the bar being updated.
    """
    query = (
        select(PortfolioSnapshotORM.equity)
        .where(PortfolioSnapshotORM.session_id == session_id)
        .order_by(PortfolioSnapshotORM.bar_index)
    )
    if bar_index is not None:
        query = query.where(PortfolioSnapshotORM.bar_index != bar_index)
    prior = db.execute(query).scalars().all()
    series = list(prior) + [current_equity]
    return max_drawdown(series) * 100.0  # report as percentage


def _empty_snapshot(db: Session, session_id: str) -> PortfolioSnapshotRead:
    session_orm = _require_session(db, session_id)
    return PortfolioSnapshotRead(
        session_id=session_id,
        bar_index=session_orm.current_bar_index,
        cash=session_orm.initial_cash,
        equity=session_orm.initial_cash,
        positions=[],
        total_realized_pnl=0.0,
        total_unrealized_pnl=0.0,
        drawdown_pct=0.0,
    )


def _to_read_aggregate(snap: PortfolioSnapshotORM) -> PortfolioSnapshotRead:
    """Read a snapshot from its persisted aggregate fields only.

    Used for the equity curve, where per-position detail is not displayed.
    Performs no extra DB reads.
    """
    return PortfolioSnapshotRead(
        session_id=snap.session_id,
        bar_index=snap.bar_index,
        cash=snap.cash,
        equity=snap.equity,
        positions=[],
        total_realized_pnl=snap.total_realized_pnl,
        total_unrealized_pnl=snap.total_unrealized_pnl,
        drawdown_pct=snap.drawdown_pct,
    )


def _to_read(db: Session, snap: PortfolioSnapshotORM) -> PortfolioSnapshotRead:
    positions = (
        db.execute(select(PositionORM).where(PositionORM.session_id == snap.session_id))
        .scalars()
        .all()
    )
    marks = _marks_at_bar(db, snap.session_id, snap.bar_index)
    return PortfolioSnapshotRead(
        session_id=snap.session_id,
        bar_index=snap.bar_index,
        cash=snap.cash,
        total_fees=_total_fees(db, snap.session_id, snap.bar_index),
        equity=snap.equity,
        positions=[
            PositionRead(
                instrument=p.contract_key or p.instrument,
                side=p.side,  # type: ignore[arg-type]
                quantity=p.quantity,
                avg_entry_price=p.avg_entry_price,
                unrealized_pnl=unrealized_pnl(
                    _position_state(p),
                    marks.get(p.contract_key or p.instrument, p.avg_entry_price),
                ),
                realized_pnl=p.realized_pnl,
                multiplier=getattr(p, "multiplier", 1.0) or 1.0,
                instrument_kind=p.instrument_kind,  # type: ignore[arg-type]
            )
            for p in positions
            if p.side != "flat" or p.realized_pnl != 0.0
        ],
        total_realized_pnl=snap.total_realized_pnl,
        total_unrealized_pnl=snap.total_unrealized_pnl,
        drawdown_pct=snap.drawdown_pct,
    )


def _total_fees(db: Session, session_id: str, bar_index: int) -> float:
    from sqlalchemy import func

    return db.scalar(
        select(func.coalesce(func.sum(FillORM.fee), 0.0))
        .join(OrderORM, OrderORM.order_id == FillORM.order_id)
        .where(OrderORM.session_id == session_id, FillORM.fill_bar <= bar_index)
    )
