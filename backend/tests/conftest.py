"""Shared pytest fixtures for the backend test suite.

Phase 0 provided the application and test-client fixtures. Phase 1
Module 3 adds the schema-aware ``db_session`` fixture: each test gets a
brand-new in-memory SQLite with all ORM tables created, and the session
is closed (database discarded) when the test ends.
"""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.database import Base, build_engine, build_session_factory
from src.main import create_app
from src.models import fill as _fill_models  # noqa: F401

# Importing models here so Base.metadata knows about every ORM table
# before create_all runs. Each new ORM module added later must be
# imported below (or transitively from src.models.__init__).
from src.models import market as _market_models  # noqa: F401
from src.models import order as _order_models  # noqa: F401
from src.models import portfolio as _portfolio_models  # noqa: F401
from src.models import risk as _risk_models  # noqa: F401
from src.models import session as _session_models  # noqa: F401


@pytest.fixture
def app() -> FastAPI:
    """A fresh FastAPI application instance, isolated per test."""
    return create_app()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """A TestClient bound to a fresh application instance."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_db_url() -> str:
    """In-memory SQLite URL for tests."""
    return "sqlite:///:memory:"


@pytest.fixture
def db_engine(test_db_url: str) -> Iterator[Engine]:
    """A fresh in-memory engine with all ORM tables created."""
    engine = build_engine(test_db_url)
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Iterator[Session]:
    """A fresh DB session bound to the per-test engine."""
    session_factory = build_session_factory(db_engine)
    with session_factory() as session:
        yield session
