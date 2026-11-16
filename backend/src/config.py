"""Runtime settings — paths, database URL, defaults.

Settings are sourced from environment variables when present and fall
back to repository-relative defaults. The defaults are tuned for V1's
local-only constraint: the SQLite file lives under ``backend/data/db/``.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: backend/src/config.py -> backend/src -> backend
BACKEND_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BACKEND_DIR / "data"
DB_DIR: Path = DATA_DIR / "db"
DEFAULT_DB_PATH: Path = DB_DIR / "trading_lab.sqlite"


class Settings(BaseSettings):
    """Application settings.

    Override any field via the ``MRTL_`` environment-variable prefix
    (e.g. ``MRTL_DATABASE_URL=sqlite:///./alt.sqlite``).
    """

    model_config = SettingsConfigDict(env_prefix="MRTL_", extra="ignore")

    database_url: str = Field(
        default=f"sqlite:///{DEFAULT_DB_PATH}",
        description="SQLAlchemy URL. Default is the local SQLite file.",
    )
    data_dir: Path = Field(default=DATA_DIR, description="Root data directory.")
    db_dir: Path = Field(default=DB_DIR, description="SQLite file directory.")
    frontend_origin: str | None = Field(
        default=None,
        description="Optional additional frontend origin for an explicit local port.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor used as a FastAPI dependency."""
    return Settings()
