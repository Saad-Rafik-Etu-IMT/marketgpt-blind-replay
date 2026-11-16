"""HTTP routes for replay sessions.

  * POST   /api/replay/sessions
  * GET    /api/replay/sessions
  * GET    /api/replay/sessions/{id}
  * POST   /api/replay/sessions/{id}/step
  * POST   /api/replay/sessions/{id}/jump
  * GET    /api/replay/sessions/{id}/visible-bars   (§5 invariant 1 enforcement)
  * POST   /api/replay/sessions/{id}/complete

Error bodies use the flat ``{error_code, detail, timestamp}`` shape
established in Phase 1.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session
from src.api.errors import AppError
from src.schemas.market import OHLCVBar
from src.schemas.replay import (
    CreateSessionRequest,
    JumpRequest,
    ReplaySessionRead,
    StepRequest,
)
from src.services.replay_engine import (
    JumpDirectionError,
    SessionCompletedError,
    SessionNotFoundError,
    create_session,
    get_session,
    get_visible_bars,
    jump_to_bar,
    list_sessions,
    mark_completed,
    step_session,
)

router = APIRouter(prefix="/replay", tags=["replay"])


# --- CRUD ------------------------------------------------------------------


@router.post(
    "/sessions",
    response_model=ReplaySessionRead,
    status_code=status.HTTP_201_CREATED,
)
def post_session(
    body: CreateSessionRequest,
    session: Session = Depends(get_db_session),
) -> ReplaySessionRead:
    try:
        return create_session(
            session,
            instrument=body.instrument,
            start_date=body.start_date,
            end_date=body.end_date,
            initial_cash=body.initial_cash,
            multiplier=body.multiplier,
            fee_per_unit=body.fee_per_unit,
            spread_bps=body.spread_bps,
        )
    except ValueError as exc:
        raise AppError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "INVALID_SESSION_REQUEST",
            str(exc),
        ) from exc


@router.get("/sessions", response_model=list[ReplaySessionRead])
def get_sessions(session: Session = Depends(get_db_session)) -> list[ReplaySessionRead]:
    return list_sessions(session)


@router.get("/sessions/{session_id}", response_model=ReplaySessionRead)
def get_one_session(
    session_id: str, session: Session = Depends(get_db_session)
) -> ReplaySessionRead:
    try:
        return get_session(session, session_id)
    except SessionNotFoundError as exc:
        raise AppError(
            status.HTTP_404_NOT_FOUND, "SESSION_NOT_FOUND", str(exc)
        ) from exc


# --- Transitions -----------------------------------------------------------


@router.post("/sessions/{session_id}/step", response_model=ReplaySessionRead)
def post_step(
    session_id: str,
    body: StepRequest | None = None,
    session: Session = Depends(get_db_session),
) -> ReplaySessionRead:
    payload = body or StepRequest()
    try:
        return step_session(session, session_id, n=payload.n)
    except SessionNotFoundError as exc:
        raise AppError(
            status.HTTP_404_NOT_FOUND, "SESSION_NOT_FOUND", str(exc)
        ) from exc
    except SessionCompletedError as exc:
        raise AppError(status.HTTP_409_CONFLICT, "SESSION_COMPLETED", str(exc)) from exc
    except ValueError as exc:
        raise AppError(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "INVALID_STEP", str(exc)
        ) from exc


@router.post("/sessions/{session_id}/jump", response_model=ReplaySessionRead)
def post_jump(
    session_id: str,
    body: JumpRequest,
    session: Session = Depends(get_db_session),
) -> ReplaySessionRead:
    try:
        return jump_to_bar(session, session_id, target_index=body.target_index)
    except SessionNotFoundError as exc:
        raise AppError(
            status.HTTP_404_NOT_FOUND, "SESSION_NOT_FOUND", str(exc)
        ) from exc
    except JumpDirectionError as exc:
        raise AppError(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "FORWARD_JUMP_REJECTED", str(exc)
        ) from exc
    except ValueError as exc:
        raise AppError(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "INVALID_JUMP", str(exc)
        ) from exc


@router.post("/sessions/{session_id}/complete", response_model=ReplaySessionRead)
def post_complete(
    session_id: str, session: Session = Depends(get_db_session)
) -> ReplaySessionRead:
    try:
        return mark_completed(session, session_id)
    except SessionNotFoundError as exc:
        raise AppError(
            status.HTTP_404_NOT_FOUND, "SESSION_NOT_FOUND", str(exc)
        ) from exc


# --- Visible bars (§5 invariant 1) ----------------------------------------


@router.get("/sessions/{session_id}/visible-bars", response_model=list[OHLCVBar])
def get_visible(
    session_id: str, session: Session = Depends(get_db_session)
) -> list[OHLCVBar]:
    try:
        return get_visible_bars(session, session_id)
    except SessionNotFoundError as exc:
        raise AppError(
            status.HTTP_404_NOT_FOUND, "SESSION_NOT_FOUND", str(exc)
        ) from exc
