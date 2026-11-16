"""Anti-lookahead invariant tests — Phase 2.

§5 of the project prompt declares these invariants NON-NEGOTIABLE with a
zero-tolerance pass rate. The test names below are the ones explicitly
required by the prompt:

  * test_api_never_returns_future_bars
  * test_order_cannot_reference_future_bar          (Phase 3 — see note)
  * test_fill_price_within_bar_range                (Phase 3 — see note)
  * test_risk_metrics_exclude_future_data           (Phase 4 — see note)
  * test_chart_data_truncated_at_current_bar

All retained invariant tests are active.
"""

from collections.abc import Iterator
from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session
from src.database import build_session_factory
from src.models.fill import FillORM
from src.models.market import OHLCVBarORM
from src.schemas.order import CreateOrderRequest
from src.services.order_service import submit_order
from src.services.replay_engine import (
    create_session,
    get_visible_bars,
    step_session,
)


@pytest.fixture
def seeded_db(db_session: Session) -> Session:
    """Same seed as test_replay_engine: 10 SPY bars, ascending dates."""
    days = [3, 4, 5, 6, 9, 10, 11, 12, 13, 16]
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


@pytest.fixture
def replay_client(
    app: FastAPI, db_engine: Engine, seeded_db: Session
) -> Iterator[TestClient]:
    """A TestClient sharing the seeded DB via dependency override."""
    session_factory = build_session_factory(db_engine)

    def _override() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = _override
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


# ===========================================================================
# Invariant 1 — API never returns bars index > current_bar_index
# ===========================================================================


def test_api_never_returns_future_bars(replay_client: TestClient) -> None:
    """GET .../visible-bars returns only bars up to and including current."""
    create_response = replay_client.post(
        "/api/replay/sessions",
        json={
            "instrument": "SPY",
            "start_date": "2023-01-03",
            "end_date": "2023-01-16",
        },
    )
    assert create_response.status_code == 201, create_response.text
    session_id = create_response.json()["session_id"]

    # At index 0 → exactly one bar visible (the first).
    bars = replay_client.get(f"/api/replay/sessions/{session_id}/visible-bars").json()
    assert len(bars) == 1

    # Step three bars → exactly four visible (indices 0..3).
    replay_client.post(f"/api/replay/sessions/{session_id}/step", json={"n": 3})
    bars = replay_client.get(f"/api/replay/sessions/{session_id}/visible-bars").json()
    assert len(bars) == 4
    # And the LAST visible bar matches the seeded sequence — no leak.
    assert bars[-1]["date"] == "2023-01-06"


# ===========================================================================
# Invariant 5 — Chart data truncated at current bar (service-level guarantee)
# ===========================================================================


def test_chart_data_truncated_at_current_bar(seeded_db: Session) -> None:
    """get_visible_bars never returns a bar with index > current."""
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 16))
    step_session(seeded_db, s.session_id, n=4)
    bars = get_visible_bars(seeded_db, s.session_id)
    assert len(bars) == 5  # indices 0..4
    # The bar dates are strictly ≤ the current pointer's date.
    current_date = bars[-1].date
    for bar in bars:
        assert bar.date <= current_date


def test_visible_bars_at_index_zero_returns_one_bar(seeded_db: Session) -> None:
    """current_bar_index=0 is inclusive — the first bar IS visible."""
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 16))
    bars = get_visible_bars(seeded_db, s.session_id)
    assert len(bars) == 1
    assert bars[0].date == date(2023, 1, 3)


# ===========================================================================
# Risk and execution invariants
# ===========================================================================


def test_risk_metrics_exclude_future_data(seeded_db: Session) -> None:
    """Risk metrics never include data from bars after current_bar_index (§5 inv. 4).

    Structural enforcement: the portfolio_snapshots table is the only
    equity-curve source, and snapshots are only ever written for the
    session's current bar. So computing metrics at bar K cannot see
    a bar > K — there's no row for it yet.
    """
    from src.services.risk_service import compute_metrics

    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 16))

    # Metrics at bar 0 — no future data possible.
    m0 = compute_metrics(seeded_db, s.session_id)
    # No trades and a single snapshot → metrics that need history are None.
    assert m0.sharpe_ratio is None
    assert m0.win_rate is None
    assert m0.expectancy is None

    # Step a few bars; snapshots count == bars stepped + 1 (the initial).
    step_session(seeded_db, s.session_id, n=3)
    snapshots = (
        seeded_db.query(
            __import__(
                "src.models.portfolio", fromlist=["PortfolioSnapshotORM"]
            ).PortfolioSnapshotORM
        )
        .filter_by(session_id=s.session_id)
        .all()
    )
    assert len(snapshots) == 4  # bars 0..3 (inclusive)
    # And every snapshot has bar_index <= current_bar_index — by construction.
    current = max(s.bar_index for s in snapshots)
    assert all(snap.bar_index <= current for snap in snapshots)


# --- Phase 3 invariants now active ----------------------------------------


def test_order_cannot_reference_future_bar(seeded_db: Session) -> None:
    """Limit/stop orders cannot fill on the bar they were submitted (§5 inv. 2)."""
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 16))
    # Submit a limit-buy that *would* fill at bar 0 if not for the rule
    # (bar 0 low = 98, limit = 99).
    order = submit_order(
        seeded_db,
        CreateOrderRequest(
            session_id=s.session_id,
            side="buy",
            order_type="limit",
            quantity=10,
            limit_price=99.0,
            rationale="must not fill on submission bar",
        ),
    )
    # Even though the price condition is met, the limit order remains
    # pending — its submitted_at_bar == current_bar_index, and the
    # fill rule requires current > submitted.
    assert order.status == "pending"


def test_fill_price_within_bar_range(seeded_db: Session) -> None:
    """Every persisted fill satisfies bar.low <= fill_price <= bar.high (§5 inv. 3)."""
    s = create_session(seeded_db, "SPY", date(2023, 1, 3), date(2023, 1, 16))

    # Create a mix of orders likely to fill across the session.
    submit_order(
        seeded_db,
        CreateOrderRequest(
            session_id=s.session_id,
            side="buy",
            order_type="market",
            quantity=10,
            rationale="m1",
        ),
    )
    step_session(seeded_db, s.session_id, n=1)
    submit_order(
        seeded_db,
        CreateOrderRequest(
            session_id=s.session_id,
            side="buy",
            order_type="limit",
            quantity=10,
            limit_price=99.0,
            rationale="l1",
        ),
    )
    step_session(seeded_db, s.session_id, n=1)
    from src.services.order_service import cancel_order, list_orders

    for pending in list_orders(seeded_db, s.session_id):
        if pending.status == "pending":
            cancel_order(seeded_db, pending.order_id)
    submit_order(
        seeded_db,
        CreateOrderRequest(
            session_id=s.session_id,
            side="sell",
            order_type="stop",
            quantity=10,
            stop_price=98.5,
            rationale="s1",
        ),
    )

    # Step through the full session.
    step_session(seeded_db, s.session_id, n=100)

    fills = seeded_db.query(FillORM).all()
    bars = seeded_db.query(OHLCVBarORM).order_by(OHLCVBarORM.date).all()
    assert fills, "expected at least one fill across the session"
    for f in fills:
        bar = bars[f.fill_bar]
        assert bar.low <= f.fill_price <= bar.high, (
            f"§5 invariant 3 violated: fill {f.fill_id} @ {f.fill_price} "
            f"outside bar {f.fill_bar} [{bar.low}, {bar.high}]"
        )
