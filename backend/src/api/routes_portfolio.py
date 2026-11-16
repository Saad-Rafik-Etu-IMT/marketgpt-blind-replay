"""HTTP routes for portfolio snapshots."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session
from src.api.errors import AppError
from src.schemas.portfolio import PortfolioSnapshotRead
from src.services.portfolio_service import (
    current_snapshot,
    equity_curve,
)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/{session_id}", response_model=PortfolioSnapshotRead)
def get_current(
    session_id: str, session: Session = Depends(get_db_session)
) -> PortfolioSnapshotRead:
    try:
        return current_snapshot(session, session_id)
    except LookupError as exc:
        raise AppError(
            status.HTTP_404_NOT_FOUND, "SESSION_NOT_FOUND", str(exc)
        ) from exc


@router.get("/{session_id}/history", response_model=list[PortfolioSnapshotRead])
def get_history(
    session_id: str, session: Session = Depends(get_db_session)
) -> list[PortfolioSnapshotRead]:
    return equity_curve(session, session_id)
