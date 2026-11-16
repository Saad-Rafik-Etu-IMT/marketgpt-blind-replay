"""SQLAlchemy 2.0 engine, session factory, and declarative ``Base``.

V1 is single-user and step-driven, so the engine is sync. File-backed
SQLite engines enable WAL mode (better concurrency between the API and
any future inspection tool); in-memory SQLite does not support WAL and
is left in its default journal mode.

The DB directory is auto-created if missing so a fresh checkout boots
cleanly without manual setup.
"""

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    """Declarative base for every ORM model in the project."""


def build_engine(url: str) -> Engine:
    """Create a SQLAlchemy engine for ``url`` with sane defaults.

    For SQLite the engine is configured with ``check_same_thread=False``
    (FastAPI may dispatch requests on multiple threads) and, for
    file-backed SQLite, WAL journaling is enabled on every connection.
    """
    connect_args: dict[str, object] = {}
    is_sqlite = url.startswith("sqlite")
    is_memory = ":memory:" in url

    if is_sqlite:
        connect_args["check_same_thread"] = False

    if is_sqlite and not is_memory:
        _ensure_sqlite_path_exists(url)

    if is_sqlite and is_memory:
        engine = create_engine(
            url,
            connect_args=connect_args,
            future=True,
            poolclass=StaticPool,
        )
    else:
        engine = create_engine(url, connect_args=connect_args, future=True)

    if is_sqlite and not is_memory:
        _enable_wal_on_connect(engine)

    return engine


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a session factory bound to ``engine``.

    ``expire_on_commit=False`` keeps ORM objects usable after commit,
    which simplifies the route → service → response handoff.
    """
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def _ensure_sqlite_path_exists(url: str) -> None:
    """Create the parent directory for a file-backed SQLite URL if missing."""
    # SQLAlchemy SQLite URLs use either ``sqlite:///relative/path`` or
    # ``sqlite:////absolute/path``. Both forms collapse to the path after
    # the three-slash prefix.
    path_part = url.split("sqlite:///", 1)[1]
    if path_part and path_part != ":memory:":
        Path(path_part).parent.mkdir(parents=True, exist_ok=True)


def _enable_wal_on_connect(engine: Engine) -> None:
    """Enable WAL journaling on every new SQLite connection."""

    @event.listens_for(engine, "connect")
    def _set_wal(dbapi_connection: object, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()
