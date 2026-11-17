"""Fresh target migration and repeat-startup regression."""

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text

from src.api.dependencies import _BACKEND_DIR, _run_migrations
from src.database import build_engine


def test_fresh_db_is_migrated_from_base(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'fresh.sqlite'}"
    engine = build_engine(url)
    try:
        _run_migrations(engine, url)
        cfg = Config(str(_BACKEND_DIR / "alembic.ini"))
        cfg.set_main_option("script_location", str(_BACKEND_DIR / "alembic"))
        head = ScriptDirectory.from_config(cfg).get_current_head()
        with engine.begin() as conn:
            assert conn.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one() == head
            assert set(inspect(conn).get_table_names()) == {
                "alembic_version", "ohlcv_bars", "replay_sessions", "orders",
                "fills", "positions", "portfolio_snapshots", "risk_limits",
            }
            conn.execute(text(
                "INSERT INTO ohlcv_bars "
                "(instrument,date,open,high,low,close,volume) "
                "VALUES ('SYNTH','2023-01-03',100,105,98,102,1000)"
            ))
        _run_migrations(engine, url)
        with engine.connect() as conn:
            assert conn.execute(
                text("SELECT count(*) FROM ohlcv_bars")
            ).scalar_one() == 1
    finally:
        engine.dispose()


def test_migration_creates_missing_sqlite_directory(tmp_path: Path) -> None:
    """A fresh checkout has no data/db/ folder; `alembic upgrade head` must still work."""
    from alembic import command

    target = tmp_path / "missing" / "db" / "fresh.sqlite"
    cfg = Config(str(_BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(_BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{target}")

    command.upgrade(cfg, "head")

    engine = build_engine(f"sqlite:///{target}")
    try:
        with engine.connect() as conn:
            assert "orders" in inspect(conn).get_table_names()
    finally:
        engine.dispose()
