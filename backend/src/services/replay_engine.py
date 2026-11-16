"""Replay engine.

The replay engine is the sole owner of "current bar index" state for a
session (§5 invariant 6). Every read or write of that pointer goes
through this module so no parallel writer can violate the invariants.

Exceptions are dedicated types so routes can map them to specific HTTP
status codes:

  * ``SessionNotFoundError``      → 404
  * ``SessionCompletedError``     → 409 (illegal transition)
  * ``JumpDirectionError``        → 422 (forward jump rejected)
"""

import logging
from datetime import date, datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.market import OHLCVBarORM
from src.models.session import ReplaySessionORM
from src.schemas.market import OHLCVBar
from src.schemas.replay import ReplaySessionRead

logger = logging.getLogger(__name__)


class SessionNotFoundError(LookupError):
    """A session_id does not exist."""


class SessionCompletedError(RuntimeError):
    """An operation was attempted on a completed session."""


class JumpDirectionError(ValueError):
    """A forward jump was attempted (would reveal unseen future bars)."""


# --- CRUD -------------------------------------------------------------------


def create_session(
    db: Session,
    instrument: str,
    start_date: date,
    end_date: date,
    initial_cash: float = 100_000.0,
    multiplier: float | None = None,
    fee_per_unit: float = 0.0,
    spread_bps: float = 0.0,
) -> ReplaySessionRead:
    """Create a new replay session, validating the date range and bar coverage."""
    if end_date < start_date:
        raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date})")
    from src.services.data_loader import has_quarantined_wti
    from src.services.instrument_catalog import contract_multiplier

    if instrument == "CL=F" and has_quarantined_wti(db):
        raise ValueError(
            "CL=F data duplicates SPY and is quarantined. Import verified WTI data into a clean database; existing data is preserved."
        )
    expected = contract_multiplier(instrument)
    if multiplier is None:
        multiplier = expected or 1.0
    if expected is not None and multiplier != expected:
        raise ValueError(f"{instrument} requires multiplier {expected}")
    if multiplier <= 0:
        raise ValueError(f"multiplier must be > 0 (got {multiplier})")

    bar_count = _count_bars(db, instrument, start_date, end_date)
    if bar_count == 0:
        raise ValueError(
            f"no bars for instrument {instrument!r} between {start_date} and {end_date}"
        )

    orm = ReplaySessionORM(
        session_id=str(uuid4()),
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        current_bar_index=0,
        status="active",
        initial_cash=initial_cash,
        multiplier=multiplier,
        fee_per_unit=fee_per_unit,
        spread_bps=spread_bps,
        created_at=datetime.now(timezone.utc),
    )
    db.add(orm)
    db.commit()

    # Emit the initial portfolio snapshot at bar 0. Lazy import to
    # avoid a circular dependency at module load time.
    from src.services.portfolio_service import initialize_session

    initialize_session(db, orm.session_id)

    logger.info(
        "session created: id=%s instrument=%s bars=%d cash=%.2f",
        orm.session_id,
        instrument,
        bar_count,
        initial_cash,
    )
    return _to_read(orm, bar_count)


def get_session(db: Session, session_id: str) -> ReplaySessionRead:
    orm = _load_orm(db, session_id)
    return _to_read(orm, _count_bars(db, orm.instrument, orm.start_date, orm.end_date))


def list_sessions(db: Session) -> list[ReplaySessionRead]:
    rows = (
        db.execute(
            select(ReplaySessionORM).order_by(ReplaySessionORM.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [
        _to_read(row, _count_bars(db, row.instrument, row.start_date, row.end_date))
        for row in rows
    ]


# --- Transitions ------------------------------------------------------------


def step_session(db: Session, session_id: str, n: int = 1) -> ReplaySessionRead:
    """Advance ``current_bar_index`` by ``n``, evaluating fills at every bar.

    A multi-bar step is equivalent to N successive 1-bar steps: each
    intermediate bar gets a chance to fill pending orders. Skipping
    intermediate bars would silently lose fills that should have
    happened (e.g. a stop triggered on bar N+1 even though the user
    requested two steps to N+2).

    Clamping rule: if the requested distance reaches or exceeds the
    last bar, the pointer stops at the last bar and the session
    transitions to ``completed`` (no error).
    """
    if n < 1:
        raise ValueError(f"step n must be >= 1 (got {n})")

    orm = _load_orm(db, session_id)
    if orm.simulation_version < 2:
        raise ValueError(
            "Legacy simulation: accounting and execution rules changed. Create a new session; old results are not comparable."
        )
    if orm.status == "completed":
        raise SessionCompletedError(f"session {session_id} is already completed")

    total_bars = _count_bars(db, orm.instrument, orm.start_date, orm.end_date)
    last_index = total_bars - 1

    # Lazy import — broker → replay_engine for state reads is fine, but
    # replay_engine importing broker at module load would form a cycle.
    from src.services.broker import process_pending_orders
    from src.services.portfolio_service import apply_step_snapshot
    from src.services.risk_service import should_auto_pause

    steps_taken = 0
    while steps_taken < n and orm.current_bar_index < last_index:
        # One bar = one transaction. The index advance, any fills, and the
        # bar snapshot commit together so a mid-bar failure rolls the whole
        # bar back (no advanced pointer without its fills/snapshot).
        try:
            orm.current_bar_index += 1
            process_pending_orders(db, session_id, commit=False)
            apply_step_snapshot(db, session_id)
            db.commit()
        except Exception:
            db.rollback()
            db.refresh(orm)
            raise
        db.refresh(orm)

        if should_auto_pause(db, session_id):
            orm.status = "paused"
            db.commit()
            logger.warning(
                "session auto-paused on drawdown breach: id=%s bar=%d",
                session_id,
                orm.current_bar_index,
            )
            return _to_read(orm, total_bars)
        steps_taken += 1

    if orm.current_bar_index >= last_index:
        orm.status = "completed"
        _cancel_pending(db, session_id)
        db.commit()

    return _to_read(orm, total_bars)


def jump_to_bar(db: Session, session_id: str, target_index: int) -> ReplaySessionRead:
    """Jump the pointer to ``target_index``.

    Only backward jumps (or staying put) are permitted — moving forward
    would expose unseen future bars to the UI (§5).
    """
    if target_index < 0:
        raise ValueError(f"target_index must be >= 0 (got {target_index})")

    orm = _load_orm(db, session_id)
    if target_index > orm.current_bar_index:
        raise JumpDirectionError(
            f"forward jump not allowed: target={target_index} > "
            f"current={orm.current_bar_index}"
        )

    if target_index != orm.current_bar_index:
        raise JumpDirectionError(
            "Rewinding would retain future knowledge and portfolio state. "
            "Create a new session to repeat; use the chart to review history."
        )
    orm.current_bar_index = target_index
    # A backward jump resumes the session if it had completed.
    # Completed sessions remain completed, including a no-op jump.
    db.commit()
    return _to_read(
        orm,
        _count_bars(db, orm.instrument, orm.start_date, orm.end_date),
    )


def mark_completed(db: Session, session_id: str) -> ReplaySessionRead:
    orm = _load_orm(db, session_id)
    orm.status = "completed"
    _cancel_pending(db, session_id)
    db.commit()
    return _to_read(
        orm,
        _count_bars(db, orm.instrument, orm.start_date, orm.end_date),
    )


# --- Visible bars (anti-lookahead read path) -------------------------------


def get_visible_bars(db: Session, session_id: str) -> list[OHLCVBar]:
    """Return bars 0..current_bar_index (inclusive) for ``session_id``.

    THIS is the §5 invariant-1 enforcement point. The query is bounded
    by the session's own date range AND by ``current_bar_index``, so no
    future bar can leak to a caller.
    """
    orm = _load_orm(db, session_id)
    bars = _ordered_bars(db, orm.instrument, orm.start_date, orm.end_date)
    visible = bars[: orm.current_bar_index + 1]
    return [
        OHLCVBar(
            date=b.date,
            open=b.open,
            high=b.high,
            low=b.low,
            close=b.close,
            volume=b.volume,
            adjusted_close=b.adjusted_close,
        )
        for b in visible
    ]


# --- Internals --------------------------------------------------------------


def _load_orm(db: Session, session_id: str) -> ReplaySessionORM:
    orm = db.get(ReplaySessionORM, session_id)
    if orm is None:
        raise SessionNotFoundError(f"session not found: {session_id}")
    return orm


def _count_bars(db: Session, instrument: str, start_date: date, end_date: date) -> int:
    """Count bars within the session's instrument and date window."""
    from sqlalchemy import func

    return db.execute(
        select(func.count(OHLCVBarORM.id))
        .where(OHLCVBarORM.instrument == instrument)
        .where(OHLCVBarORM.date >= start_date)
        .where(OHLCVBarORM.date <= end_date)
    ).scalar_one()


def _ordered_bars(
    db: Session, instrument: str, start_date: date, end_date: date
) -> list[OHLCVBarORM]:
    return list(
        db.execute(
            select(OHLCVBarORM)
            .where(OHLCVBarORM.instrument == instrument)
            .where(OHLCVBarORM.date >= start_date)
            .where(OHLCVBarORM.date <= end_date)
            .order_by(OHLCVBarORM.date)
        ).scalars()
    )


def _to_read(orm: ReplaySessionORM, total_bars: int) -> ReplaySessionRead:
    return ReplaySessionRead(
        session_id=orm.session_id,
        simulation_version=orm.simulation_version,
        instrument=orm.instrument,
        start_date=orm.start_date,
        end_date=orm.end_date,
        current_bar_index=orm.current_bar_index,
        status=orm.status,  # type: ignore[arg-type]
        initial_cash=orm.initial_cash,
        created_at=orm.created_at,
        total_bars=total_bars,
        multiplier=orm.multiplier,
        fee_per_unit=orm.fee_per_unit,
        spread_bps=orm.spread_bps,
    )


def _cancel_pending(db: Session, session_id: str) -> None:
    from src.models.order import OrderORM

    for order in db.scalars(
        select(OrderORM).where(
            OrderORM.session_id == session_id, OrderORM.status == "pending"
        )
    ):
        order.status = "cancelled"
