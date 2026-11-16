"""Shared FastAPI dependencies.

Phase 1 exposes one dependency: ``get_db_session``, a per-request
SQLAlchemy session bound to the configured engine. Tests override this
to inject an in-memory engine.
"""

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from threading import Lock

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from src.database import build_engine, build_session_factory
from src.models import fill as _fill_models  # noqa: F401

# Import every ORM module so the mappers are registered when the app uses
# them. The schema itself is owned by Alembic migrations (see alembic/),
# not by ``Base.metadata.create_all``. New ORM modules must be added here.
from src.models import market as _market_models  # noqa: F401
from src.models import order as _order_models  # noqa: F401
from src.models import portfolio as _portfolio_models  # noqa: F401
from src.models import risk as _risk_models  # noqa: F401
from src.models import session as _session_models  # noqa: F401

# backend/src/api/dependencies.py -> backend/
_BACKEND_DIR = Path(__file__).resolve().parents[2]


_migration_lock = Lock()


def _run_migrations(engine: Engine, database_url: str) -> None:
    # First browser render sends parallel requests; lru_cache alone does not
    # serialize cache misses. Alembic configuration is process-global.
    with _migration_lock:
        _migrate_locked(engine, database_url)


def _migrate_locked(engine: Engine, database_url: str) -> None:
    """Upgrade a fresh or already-versioned target database to head."""
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(_BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(_BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(cfg, "head")


@lru_cache(maxsize=1)
def _get_engine() -> Engine:
    """Build (once) and return the process-wide engine.

    Alembic migrations are applied to head on first use so a fresh
    checkout boots without a manual migration step. Source databases and
    unstamped existing schemas are not adopted.
    """
    settings = get_settings()
    engine = build_engine(settings.database_url)
    _run_migrations(engine, settings.database_url)
    return engine


@lru_cache(maxsize=1)
def _get_session_factory() -> sessionmaker[Session]:
    """Build (once) the process-wide session factory.

    Rebuilding the factory on every request is pure overhead on a hot
    path hit several times a second; the engine is already cached, so the
    factory can be too.
    """
    return build_session_factory(_get_engine())


def get_db_session() -> Iterator[Session]:
    """Yield a SQLAlchemy session for the duration of one request."""
    session_factory = _get_session_factory()
    with session_factory() as session:
        yield session
