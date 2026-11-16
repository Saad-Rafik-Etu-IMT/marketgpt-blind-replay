"""Central error handling for the API layer.

A single ``AppError`` exception plus one FastAPI handler replaces the
per-router ``_error``/``_error_response`` helpers that each rebuilt the
same flat error body. Routes now ``raise AppError(...)`` with the
context-specific status code and error code; the handler renders the
canonical ``ErrorResponse`` shape (``{detail, error_code, timestamp}``)
that the tests and the frontend consume.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """The canonical flat error body returned by every failing endpoint."""

    detail: str
    error_code: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AppError(Exception):
    """A handled application error carrying its HTTP status and error code.

    Raised by routes (and, where convenient, services) instead of building
    a ``JSONResponse`` inline. The registered handler turns it into an
    ``ErrorResponse`` body with the given ``status_code``.
    """

    def __init__(self, status_code: int, error_code: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail


def _render(status_code: int, error_code: str, detail: str) -> JSONResponse:
    body = ErrorResponse(detail=detail, error_code=error_code)
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def register_error_handlers(app: FastAPI) -> None:
    """Register the application's exception handlers on ``app``."""

    @app.exception_handler(AppError)
    async def _handle_app_error(  # pyright: ignore[reportUnusedFunction]
        _request: Request, exc: AppError
    ) -> JSONResponse:
        return _render(exc.status_code, exc.error_code, exc.detail)
