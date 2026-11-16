"""HTTP routes for risk metrics and limits."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session
from src.schemas.risk import RiskLimitsRead, RiskLimitsUpdate, RiskMetricsRead
from src.services.risk_service import (
    compute_metrics,
    get_limits,
    update_limits,
)

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("/{session_id}", response_model=RiskMetricsRead)
def get_metrics(
    session_id: str, session: Session = Depends(get_db_session)
) -> RiskMetricsRead:
    return compute_metrics(session, session_id)


@router.get("/{session_id}/limits", response_model=RiskLimitsRead)
def read_limits(
    session_id: str, session: Session = Depends(get_db_session)
) -> RiskLimitsRead:
    return get_limits(session, session_id)


@router.put("/{session_id}/limits", response_model=RiskLimitsRead)
def put_limits(
    session_id: str,
    body: RiskLimitsUpdate,
    session: Session = Depends(get_db_session),
) -> RiskLimitsRead:
    return update_limits(session, session_id, body)
