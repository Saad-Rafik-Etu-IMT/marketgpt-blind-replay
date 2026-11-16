"""Smoke tests for the API surface — Phase 0.

These tests verify that the FastAPI application boots and that the health
endpoint honours its documented contract. They are written *before* the
implementation exists (TDD: red first, then green).
"""

from fastapi.testclient import TestClient


def test_health_endpoint_returns_200(client: TestClient) -> None:
    """GET /api/health responds with HTTP 200."""
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_endpoint_returns_expected_payload(client: TestClient) -> None:
    """GET /api/health returns the documented status/version contract."""
    response = client.get("/api/health")
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"


def test_health_response_shape(client: TestClient) -> None:
    """The health payload contains exactly the documented keys — no more."""
    response = client.get("/api/health")
    body = response.json()
    assert set(body.keys()) == {"status", "version"}
