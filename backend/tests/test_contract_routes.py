"""HTTP contract tests across the retained routers.

One place that proves every router is wired and honours its contract: a
happy path and the primary error path for each. Complements the deeper
per-domain suites; this is the breadth check that guards the API surface.
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
from src.models.market import OHLCVBarORM


@pytest.fixture
def api(app: FastAPI, db_engine: Engine) -> Iterator[TestClient]:
    """A TestClient bound to a per-test in-memory DB seeded with 10 SPY bars."""
    factory = build_session_factory(db_engine)

    with factory() as seed:
        for offset, day in enumerate([3, 4, 5, 6, 9, 10, 11, 12, 13, 16]):
            seed.add(
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
        seed.commit()

    def _override() -> Iterator[Session]:
        with factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = _override
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def session_id(api: TestClient) -> str:
    """Create an active session over the seeded SPY bars."""
    resp = api.post(
        "/api/replay/sessions",
        json={
            "instrument": "SPY",
            "start_date": "2023-01-03",
            "end_date": "2023-01-16",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


# --- 1. health -------------------------------------------------------------


def test_health_contract(api: TestClient) -> None:
    assert api.get("/api/health").status_code == 200


def test_explicit_frontend_origin_preserves_defaults_and_rejects_others(monkeypatch):
    from src.config import get_settings
    from src.main import create_app

    monkeypatch.setenv("MRTL_FRONTEND_ORIGIN", "http://localhost:3100")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            for origin in (
                "http://localhost:3100",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ):
                response = client.options(
                    "/api/orders",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    },
                )
                assert response.status_code == 200
                assert response.headers["access-control-allow-origin"] == origin
            denied = client.options(
                "/api/orders",
                headers={
                    "Origin": "http://localhost:3999",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert denied.status_code == 400
            assert "access-control-allow-origin" not in denied.headers
    finally:
        get_settings.cache_clear()


# --- 2. market -------------------------------------------------------------


def test_market_contract(api: TestClient) -> None:
    assert api.get("/api/market/instruments").status_code == 200


# --- 3. replay -------------------------------------------------------------


def test_replay_contract(api: TestClient, session_id: str) -> None:
    assert api.get(f"/api/replay/sessions/{session_id}").status_code == 200
    assert (
        api.post(f"/api/replay/sessions/{session_id}/step", json={"n": 1}).status_code
        == 200
    )
    assert api.get(f"/api/replay/sessions/{session_id}/visible-bars").status_code == 200
    # Forward jump is rejected (anti-lookahead).
    forward = api.post(
        f"/api/replay/sessions/{session_id}/jump", json={"target_index": 99}
    )
    assert forward.status_code == 422
    assert api.get("/api/replay/sessions/does-not-exist").status_code == 404


# --- 4. orders -------------------------------------------------------------


def test_orders_contract(api: TestClient, session_id: str) -> None:
    order = api.post(
        "/api/orders",
        json={
            "session_id": session_id,
            "side": "buy",
            "order_type": "market",
            "quantity": 10,
            "rationale": "contract test",
            "instrument_kind": "underlying",
        },
    )
    assert order.status_code == 201, order.text
    assert api.get(f"/api/orders?session_id={session_id}").status_code == 200
    missing = api.delete("/api/orders/does-not-exist")
    assert missing.status_code == 404


def test_option_order_is_rejected_without_creating_an_order(
    api: TestClient, session_id: str
) -> None:
    response = api.post(
        "/api/orders",
        json={
            "session_id": session_id,
            "side": "buy",
            "order_type": "market",
            "quantity": 1,
            "rationale": "Unsupported instrument must not execute as underlying",
            "instrument_kind": "option",
            "option_type": "call",
            "strike": 100,
            "expiry": "2024-01-01",
        },
    )
    assert response.status_code == 422
    assert api.get(f"/api/orders?session_id={session_id}").json() == []


# --- 5. portfolio ----------------------------------------------------------


def test_portfolio_contract(api: TestClient, session_id: str) -> None:
    assert api.get(f"/api/portfolio/{session_id}").status_code == 200
    assert api.get(f"/api/portfolio/{session_id}/history").status_code == 200
    assert api.get("/api/portfolio/does-not-exist").status_code == 404


# --- 7. risk ---------------------------------------------------------------


def test_risk_contract(api: TestClient, session_id: str) -> None:
    assert api.get(f"/api/risk/{session_id}").status_code == 200
    assert api.get(f"/api/risk/{session_id}/limits").status_code == 200
    put = api.put(f"/api/risk/{session_id}/limits", json={"max_drawdown_pct": 25.0})
    assert put.status_code == 200
    assert put.json()["max_drawdown_pct"] == 25.0


