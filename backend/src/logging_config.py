"""Logging configuration for the backend.

A single ``configure_logging`` entry point sets up structured-ish console
logging for the whole process. It is idempotent so repeated calls (e.g.
one per app factory invocation in tests) do not stack handlers.

Domain code obtains loggers via ``logging.getLogger(__name__)`` and logs
events (session created, fills, auto-pause, rejections); the request/error
middleware in ``main.py`` logs the HTTP layer and unhandled exceptions.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s :: %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once for the process.

    The level comes from the ``LOG_LEVEL`` environment variable (default
    ``INFO``) unless an explicit ``level`` is passed. Safe to call more
    than once — subsequent calls are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format=_DEFAULT_FORMAT,
    )
    _CONFIGURED = True
