"""Health-check endpoint.

A liveness probe used by the frontend, the Makefile smoke step, and CI to
confirm the backend is up and to surface its version.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from src import __version__

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Response model for the health endpoint."""

    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Report service liveness and the running API version."""
    return HealthResponse(status="ok", version=__version__)
