"""Tests for the replay engine — Phase 2.

The replay engine is the sole owner of "current bar index" state (§5
invariant 6). These tests cover:

  * Session creation with a default + custom initial cash.
  * Bounds: cannot create a session for an unknown instrument or with
    no bars in the requested date range.
  * Step forward by N (default 1); cannot step past the last bar.
  * Jump: only backward, only within `[0..current]`; forward is rejected.
  * Completion transition.

Anti-lookahead invariants live in test_no_lookahead.py.
"""

from datetime import date

import pytest
from sqlalchemy.orm import Session

from src.models.market import OHLCVBarORM
from src.models.session import ReplaySessionORM
from src.services import portfolio_service
from src.services.bar_queries import bar_at
from src.services.replay_engine import (
    JumpDirectionError,
    SessionCompletedError,
    SessionNotFoundError,
    create_session,
    get_session,
    jump_to_bar,
    list_sessions,
    mark_completed,
    step_session,
)


@pytest.fixture
def seeded_db(db_session: Session) -> Session:
    """Seed 10 SPY bars (2023-01-03 .. 2023-01-12, weekdays only).

    Returns the same session, so tests can interact with the DB directly.
    """
    days = [3, 4, 5, 6, 9, 10, 11, 12, 13, 16]  # 10 weekdays in 2023-01
    for offset, day in enumerate(days):
        db_session.add(
            OHLCVBarORM(
                instrument="SPY",
                date=date(2023, 1, day),
                open=100.0 + offset,
                high=105.0 + offset,
                low=98.0 + offset,
                close=102.0 + offset,
                volume=1_000_000,
                adjusted_close=102.0 + offset,
            )
        )
    db_session.commit()
    return db_session


# --- Session creation ------------------------------------------------------

def test_initial_snapshot_failure_rolls_back_session(seeded_db, monkeypatch):
    original = portfolio_service.initialize_session

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("snapshot unavailable")

    monkeypatch.setattr(portfolio_service, "initialize_session", fail)
    with pytest.raises(RuntimeError, match="snapshot unavailable"):
        create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 16))
    assert list_sessions(seeded_db) == []


def test_single_bar_query_respects_session_window(seeded_db):
    session = create_session(seeded_db, "SPY", date(2023, 1, 4), date(2023, 1, 10))
    orm = seeded_db.get(ReplaySessionORM, session.session_id)
    assert bar_at(seeded_db, orm, 0).date == date(2023, 1, 4)
    assert bar_at(seeded_db, orm, 4).date == date(2023, 1, 10)
    assert bar_at(seeded_db, orm, 5) is None
    assert bar_at(seeded_db, orm, -1) is None


def test_create_session_with_defaults(seeded_db: Session) -> None:
    session = create_session(
        seeded_db,
        instrument="SPY",
        start_date=date(2023, 1, 3),
        end_date=date(2023, 1, 12),
    )
    assert session.session_id
    assert session.instrument == "SPY"
    assert session.current_bar_index == 0
    assert session.status == "active"
    assert session.initial_cash == 100_000.0  # documented default


def test_create_session_with_custom_initial_cash(seeded_db: Session) -> None:
    session = create_session(
        seeded_db,
        instrument="SPY",
        start_date=date(2023, 1, 3),
        end_date=date(2023, 1, 12),
        initial_cash=50_000.0,
    )
    assert session.initial_cash == 50_000.0


def test_create_session_unknown_instrument(seeded_db: Session) -> None:
    with pytest.raises(ValueError) as exc:
        create_session(
            seeded_db,
            instrument="UNKNOWN",
            start_date=date(2023, 1, 3),
            end_date=date(2023, 1, 12),
        )
    assert "no bars" in str(exc.value).lower() or "unknown" in str(exc.value).lower()


def test_create_session_empty_date_range(seeded_db: Session) -> None:
    """No bars between start and end → reject."""
    with pytest.raises(ValueError):
        create_session(
            seeded_db,
            instrument="SPY",
            start_date=date(2024, 6, 1),
            end_date=date(2024, 6, 30),
        )


def test_create_session_inverted_date_range(seeded_db: Session) -> None:
    """end < start → reject."""
    with pytest.raises(ValueError):
        create_session(
            seeded_db,
            instrument="SPY",
            start_date=date(2023, 1, 12),
            end_date=date(2023, 1, 3),
        )


def test_listing_and_retrieval(seeded_db: Session) -> None:
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 12))
    fetched = get_session(seeded_db, s.session_id)
    assert fetched.session_id == s.session_id
    assert any(item.session_id == s.session_id for item in list_sessions(seeded_db))


def test_get_unknown_session_raises(seeded_db: Session) -> None:
    with pytest.raises(SessionNotFoundError):
        get_session(seeded_db, "does-not-exist")


# --- Stepping --------------------------------------------------------------


def test_step_advances_one_bar_by_default(seeded_db: Session) -> None:
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 12))
    after = step_session(seeded_db, s.session_id)
    assert after.current_bar_index == 1


def test_step_advances_n_bars(seeded_db: Session) -> None:
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 12))
    after = step_session(seeded_db, s.session_id, n=3)
    assert after.current_bar_index == 3


def test_step_cannot_pass_last_bar(seeded_db: Session) -> None:
    """Stepping past the last bar clamps to the last bar AND marks completed."""
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 16))
    # 10 bars → last index is 9. Try to leap past.
    after = step_session(seeded_db, s.session_id, n=100)
    assert after.current_bar_index == 9
    assert after.status == "completed"


def test_step_rejects_zero_or_negative(seeded_db: Session) -> None:
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 12))
    with pytest.raises(ValueError):
        step_session(seeded_db, s.session_id, n=0)
    with pytest.raises(ValueError):
        step_session(seeded_db, s.session_id, n=-1)


def test_step_on_completed_session_rejected(seeded_db: Session) -> None:
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 12))
    step_session(seeded_db, s.session_id, n=100)  # completes it
    with pytest.raises(SessionCompletedError):
        step_session(seeded_db, s.session_id)


def test_step_is_atomic_on_mid_bar_failure(
    seeded_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after the index advance rolls the whole bar back.

    The pointer must not move and no snapshot may exist for the unreached
    bar — the bar's index advance, fills and snapshot commit together.
    """
    from sqlalchemy import func, select

    from src.models.portfolio import PortfolioSnapshotORM
    from src.services import portfolio_service

    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 12))

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated mid-step failure")

    monkeypatch.setattr(portfolio_service, "apply_step_snapshot", _boom)

    with pytest.raises(RuntimeError, match="simulated mid-step failure"):
        step_session(seeded_db, s.session_id)

    # Pointer rolled back to bar 0.
    reloaded = get_session(seeded_db, s.session_id)
    assert reloaded.current_bar_index == 0

    # No snapshot leaked for bar 1.
    snap_count = seeded_db.execute(
        select(func.count())
        .select_from(PortfolioSnapshotORM)
        .where(PortfolioSnapshotORM.session_id == s.session_id)
        .where(PortfolioSnapshotORM.bar_index == 1)
    ).scalar_one()
    assert snap_count == 0


# --- Jumping ---------------------------------------------------------------


def test_jump_backward_allowed(seeded_db: Session) -> None:
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 12))
    step_session(seeded_db, s.session_id, n=5)
    with pytest.raises(JumpDirectionError, match="Rewinding"):
        jump_to_bar(seeded_db, s.session_id, target_index=2)


def test_jump_to_current_allowed(seeded_db: Session) -> None:
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 12))
    step_session(seeded_db, s.session_id, n=3)
    after = jump_to_bar(seeded_db, s.session_id, target_index=3)
    assert after.current_bar_index == 3


def test_jump_forward_rejected(seeded_db: Session) -> None:
    """Forward jump is rejected — would expose unseen future bars (§5)."""
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 12))
    step_session(seeded_db, s.session_id, n=3)
    with pytest.raises(JumpDirectionError):
        jump_to_bar(seeded_db, s.session_id, target_index=5)


def test_jump_negative_rejected(seeded_db: Session) -> None:
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 12))
    with pytest.raises(ValueError):
        jump_to_bar(seeded_db, s.session_id, target_index=-1)


# --- Completion ------------------------------------------------------------


def test_mark_completed(seeded_db: Session) -> None:
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 12))
    after = mark_completed(seeded_db, s.session_id)
    assert after.status == "completed"


def test_completed_session_step_blocked(seeded_db: Session) -> None:
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 12))
    mark_completed(seeded_db, s.session_id)
    with pytest.raises(SessionCompletedError):
        step_session(seeded_db, s.session_id)
