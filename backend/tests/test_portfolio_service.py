"""Tests for portfolio_service — Phase 4b.

The service applies fills (computed by the broker) to a session's
positions, updates cash, and persists portfolio snapshots that
collectively form the equity curve.
"""

from datetime import date

import pytest
from sqlalchemy.orm import Session

from src.models.market import OHLCVBarORM
from src.models.portfolio import PortfolioSnapshotORM
from src.schemas.order import CreateOrderRequest
from src.services.order_service import submit_order
from src.services.portfolio_service import (
    current_snapshot,
    equity_curve,
)
from src.services.replay_engine import create_session, step_session


@pytest.fixture
def seeded(db_session: Session) -> Session:
    """10 bars: a smooth uptrend so test math is easy."""
    rows = [
        (3, 100, 102, 99, 101),
        (4, 101, 104, 100, 103),
        (5, 103, 106, 102, 105),
        (6, 105, 108, 104, 107),
        (9, 107, 110, 106, 109),
        (10, 109, 112, 108, 111),
        (11, 111, 114, 110, 113),
        (12, 113, 116, 112, 115),
        (13, 115, 118, 114, 117),
        (16, 117, 120, 116, 119),
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


# --- Initial state --------------------------------------------------------


def test_initial_snapshot_after_session_creation(seeded: Session) -> None:
    """Creating a session emits an initial snapshot at bar 0."""
    sid = _sid(seeded)
    snap = current_snapshot(seeded, sid)
    assert snap.cash == 100_000.0
    assert snap.equity == 100_000.0
    assert snap.positions == []
    assert snap.total_realized_pnl == 0.0
    assert snap.drawdown_pct == 0.0


# --- After a market buy --------------------------------------------------


def test_snapshot_after_market_buy_reduces_cash(seeded: Session) -> None:
    sid = _sid(seeded)
    submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="market",
            quantity=10,
            rationale="entry",
        ),
    )
    step_session(seeded, sid, n=1)  # execute queued order at next opening
    snap = current_snapshot(seeded, sid)
    # Bar 0 open = 100. 10 shares * 100 = 1000 spent.
    assert snap.cash == pytest.approx(98_990.0)
    assert len(snap.positions) == 1
    pos = snap.positions[0]
    assert pos.side == "long"
    assert pos.quantity == 10
    assert pos.avg_entry_price == 101.0


def test_equity_uses_close_price_as_mark(seeded: Session) -> None:
    """After a buy, equity at bar 0 marks the position to bar 0's close."""
    sid = _sid(seeded)
    submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="market",
            quantity=10,
            rationale="entry",
        ),
    )
    step_session(seeded, sid, n=1)  # execute queued order at next opening
    snap = current_snapshot(seeded, sid)
    # Bar 0 close = 101. Unrealized = 10 * (101-100) = 10.
    # Equity = cash (99,000) + unrealized (10) = 99,010.
    assert snap.equity == pytest.approx(100_020.0)
    assert snap.total_unrealized_pnl == pytest.approx(20.0)


# --- Across multiple steps ----------------------------------------------


def test_equity_curve_lengthens_with_each_step(seeded: Session) -> None:
    sid = _sid(seeded)
    submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="market",
            quantity=10,
            rationale="entry",
        ),
    )
    step_session(seeded, sid, n=3)

    curve = equity_curve(seeded, sid)
    # Expect snapshots at bars 0, 1, 2, 3 (4 entries).
    assert len(curve) == 4
    assert [s.bar_index for s in curve] == [0, 1, 2, 3]


def test_equity_rises_with_market(seeded: Session) -> None:
    """A long position in a rising market shows monotonically rising equity."""
    sid = _sid(seeded)
    submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="market",
            quantity=10,
            rationale="entry",
        ),
    )
    step_session(seeded, sid, n=5)

    curve = equity_curve(seeded, sid)
    equities = [s.equity for s in curve]
    # Monotonic non-decreasing (close prices in seed are strictly rising).
    for previous, current in zip(equities, equities[1:], strict=False):
        assert current >= previous


# --- Realized P&L on a round trip ----------------------------------------


def test_realized_pnl_after_round_trip(seeded: Session) -> None:
    sid = _sid(seeded)
    # Buy 10 at bar 0 open (=100).
    submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="buy",
            order_type="market",
            quantity=10,
            rationale="entry",
        ),
    )
    step_session(seeded, sid, n=1)  # execute queued order at next opening
    # Step to bar 2 and sell 10 at the open (=103).
    step_session(seeded, sid, n=2)
    submit_order(
        seeded,
        CreateOrderRequest(
            session_id=sid,
            side="sell",
            order_type="market",
            quantity=10,
            rationale="exit",
        ),
    )
    step_session(seeded, sid, n=1)  # execute queued order at next opening

    snap = current_snapshot(seeded, sid)
    # Realized = 10 * (103 - 100) = 30. Cash back: started 100k, paid 1000, got 1030 → 100,030.
    assert snap.total_realized_pnl == pytest.approx(60.0)
    assert snap.cash == pytest.approx(100_060.0)
    # No open positions → equity == cash.
    assert snap.equity == pytest.approx(100_060.0)
    assert snap.positions == [] or all(p.side == "flat" for p in snap.positions)


# --- Snapshots are persisted, not recomputed -----------------------------


def test_snapshots_persist_to_db(seeded: Session) -> None:
    sid = _sid(seeded)
    step_session(seeded, sid, n=2)
    persisted = seeded.query(PortfolioSnapshotORM).filter_by(session_id=sid).all()
    assert len(persisted) >= 3  # initial + 2 steps
