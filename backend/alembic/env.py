"""Alembic migration environment.

The target metadata is the project's declarative ``Base.metadata`` with
every ORM module imported so all tables are registered. The database URL
comes from the application settings, keeping migrations and the running
app pointed at the same database.
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from src.config import get_settings
from src.database import Base
from src.models import fill as _fill  # noqa: F401

# Import every ORM module so Base.metadata is fully populated before
# autogenerate / migrations run. New ORM modules must be added here.
from src.models import market as _market  # noqa: F401
from src.models import order as _order  # noqa: F401
from src.models import portfolio as _portfolio  # noqa: F401
from src.models import risk as _risk  # noqa: F401
from src.models import session as _session  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Point Alembic at the app's configured database URL, unless a caller has
# already set one on the config (e.g. the app bootstrap targeting a specific
# database, or a test targeting a temp file). Only fall back to settings.
if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL, no DBAPI connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # SQLite needs batch mode for ALTER
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
