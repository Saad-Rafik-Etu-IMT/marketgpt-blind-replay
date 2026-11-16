"""Tests for /api/market routes — Phase 1, Module 5.

The market routes wire validator + loader behind HTTP:

  * POST /api/market/upload  — multipart CSV, validates, stores
  * GET  /api/market/instruments — lists what's loaded
  * GET  /api/market/bars/{instrument} — all bars (inspection only)

These tests use the live FastAPI ``client`` fixture with a per-test
in-memory DB injected via the ``get_db_session`` dependency override.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session
from src.database import build_session_factory

_HEADER = "date,open,high,low,close,volume,adjusted_close\n"
_GOOD_CSV = (
    _HEADER
    + "2023-01-03,100,105,98,102,1000000,102\n"
    + "2023-01-04,102,108,101,107,1200000,107\n"
    + "2023-01-05,107,109,104,105,900000,105\n"
)


@pytest.fixture
def market_client(app: FastAPI, db_engine: Engine) -> Iterator[TestClient]:
    """A TestClient whose DB dependency is bound to the per-test engine."""
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


def _upload(client: TestClient, instrument: str, csv_text: str) -> Any:
    """POST a CSV body as multipart/form-data."""
    return client.post(
        "/api/market/upload",
        params={"instrument": instrument},
        files={"file": (f"{instrument}.csv", csv_text, "text/csv")},
    )


# --- Upload happy path ------------------------------------------------------


def test_upload_accepts_valid_csv(market_client: TestClient) -> None:
    response = _upload(market_client, "SPY", _GOOD_CSV)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["instrument"] == "SPY"
    assert body["loaded_count"] == 3
    assert body["errors"] == []


def test_upload_rejects_oversize_file(
    market_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file larger than the cap is refused with 413 and no bars loaded."""
    from src.api import routes_market

    monkeypatch.setattr(routes_market, "MAX_UPLOAD_BYTES", 64)
    big_csv = _GOOD_CSV + "2023-01-06,1,2,0,1,1,1\n" * 100  # > 64 bytes
    response = _upload(market_client, "SPY", big_csv)
    assert response.status_code == 413, response.text
    assert response.json()["error_code"] == "FILE_TOO_LARGE"
    # Nothing was persisted.
    instruments = market_client.get("/api/market/instruments").json()
    assert instruments == []


def test_upload_persists_bars(market_client: TestClient) -> None:
    _upload(market_client, "SPY", _GOOD_CSV)
    response = market_client.get("/api/market/instruments")
    assert response.status_code == 200
    instruments = response.json()
    assert any(
        item["instrument"] == "SPY" and item["bar_count"] == 3 for item in instruments
    )


def test_instruments_include_display_name(market_client: TestClient) -> None:
    """The instruments endpoint serves a human-readable name per symbol."""
    _upload(market_client, "SPY", _GOOD_CSV)
    rows = market_client.get("/api/market/instruments").json()
    spy = next(r for r in rows if r["instrument"] == "SPY")
    assert spy["name"] == "S&P 500 ETF"


# --- Upload validation failures --------------------------------------------


def test_upload_rejects_invalid_csv_with_typed_errors(
    market_client: TestClient,
) -> None:
    """An OHLC integrity violation is reported per-row with field + message."""
    bad_csv = _HEADER + "2023-01-03,100,95,98,102,1000000,102\n"  # high < close
    response = _upload(market_client, "SPY", bad_csv)
    assert response.status_code == 422
    body = response.json()
    assert body["loaded_count"] == 0
    assert body["errors"], "errors must be populated"
    first_error = body["errors"][0]
    assert {"row_number", "field", "message"} <= set(first_error.keys())


def test_upload_rejects_when_no_file(market_client: TestClient) -> None:
    response = market_client.post("/api/market/upload", params={"instrument": "SPY"})
    assert response.status_code == 422  # FastAPI: missing required field


def test_upload_rejects_when_instrument_missing(market_client: TestClient) -> None:
    response = market_client.post(
        "/api/market/upload",
        files={"file": ("x.csv", _GOOD_CSV, "text/csv")},
    )
    assert response.status_code == 422


def test_upload_refuses_duplicate_dates(market_client: TestClient) -> None:
    """A second upload that conflicts is rejected with 409 — nothing persisted."""
    assert _upload(market_client, "SPY", _GOOD_CSV).status_code == 201

    overlapping = (
        _HEADER
        + "2023-01-04,200,210,195,205,500000,205\n"  # collides
        + "2023-01-06,205,215,200,210,600000,210\n"
    )
    response = _upload(market_client, "SPY", overlapping)
    assert response.status_code == 409
    body = response.json()
    assert body["error_code"] == "DUPLICATE_DATES"
    assert "2023-01-04" in body["detail"]

    # And the non-conflicting bar (day 6) must not have been written.
    created = market_client.post(
        "/api/replay/sessions",
        json={
            "instrument": "SPY",
            "start_date": "2023-01-03",
            "end_date": "2023-01-06",
        },
    )
    assert created.status_code == 201, created.text
    sid = created.json()["session_id"]
    stepped = market_client.post(f"/api/replay/sessions/{sid}/step", json={"n": 100})
    assert stepped.status_code == 200, stepped.text
    bars = market_client.get(f"/api/replay/sessions/{sid}/visible-bars").json()
    assert len(bars) == 3
    assert all(b["date"] != "2023-01-06" for b in bars)


# --- Instruments listing ----------------------------------------------------


def test_instruments_empty_by_default(market_client: TestClient) -> None:
    response = market_client.get("/api/market/instruments")
    assert response.status_code == 200
    assert response.json() == []


def test_instruments_summary_shape(market_client: TestClient) -> None:
    _upload(market_client, "SPY", _GOOD_CSV)
    instruments = market_client.get("/api/market/instruments").json()
    assert len(instruments) == 1
    item = instruments[0]
    assert item["instrument"] == "SPY"
    assert item["bar_count"] == 3
    assert item["first_date"] == "2023-01-03"
    assert item["last_date"] == "2023-01-05"


# --- Removed unrestricted history ------------------------------------------


def test_full_history_endpoint_is_not_exposed(market_client: TestClient) -> None:
    """Even a loaded instrument cannot bypass session-visible history."""
    assert _upload(market_client, "SPY", _GOOD_CSV).status_code == 201
    response = market_client.get("/api/market/bars/SPY")
    assert response.status_code == 404
    assert "/api/market/bars/{instrument}" not in market_client.get(
        "/openapi.json"
    ).json()["paths"]
