"""Market-data HTTP routes.

  * POST /api/market/upload  — multipart CSV upload, validated and stored.
  * GET  /api/market/instruments — summary of every instrument in the DB.
Full-history retrieval is deliberately not exposed. Replay clients use
the session-scoped ``visible-bars`` endpoint.

Note on error bodies: error responses are built with ``JSONResponse``
rather than ``HTTPException`` so the response body is a *flat* object
(``{"error_code": "...", "detail": "...", ...}``) instead of FastAPI's
default ``{"detail": ...}`` wrapping. A flat shape is what every
consumer of the API contract expects.
"""

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import JSONResponse, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session
from src.api.errors import AppError
from src.models.market import OHLCVBarORM
from src.schemas.market import (
    InstrumentSummary,
    UploadResponse,
    ValidationReport,
)
from src.services.data_loader import load_bars
from src.services.data_validator import validate_csv_content
from src.services.instrument_catalog import contract_multiplier, display_name

router = APIRouter(prefix="/market", tags=["market"])

# CSV uploads are read fully into memory before validation, so cap the
# size to keep a single request from exhausting the process. Daily OHLCV
# for decades of one instrument is well under this.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB


async def _read_capped(file: UploadFile, limit: int) -> bytes | None:
    """Read ``file`` up to ``limit`` bytes; return None if it exceeds the cap.

    Reads in chunks so an over-size file is rejected without ever holding
    more than ``limit`` (+ one chunk) in memory.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_market_data(
    instrument: str = Query(..., min_length=1, max_length=32),
    file: UploadFile = File(...),
    session: Session = Depends(get_db_session),
) -> Response:
    """Validate a CSV upload and persist its bars atomically."""
    raw_bytes = await _read_capped(file, MAX_UPLOAD_BYTES)
    if raw_bytes is None:
        raise AppError(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "FILE_TOO_LARGE",
            f"file exceeds the maximum upload size of {MAX_UPLOAD_BYTES} bytes",
        )
    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AppError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "INVALID_ENCODING",
            f"file is not valid UTF-8: {exc}",
        ) from exc

    report = validate_csv_content(content)
    if not report.is_valid:
        return _validation_failure_body(instrument, report)

    result = load_bars(session, instrument, report)
    if not result.success:
        # is_valid above + non-empty bars => only the conflict refusal
        # remains. Surface as 409.
        raise AppError(
            status.HTTP_409_CONFLICT,
            "DUPLICATE_DATES",
            result.error or "load refused",
        )

    body = UploadResponse(
        instrument=instrument,
        loaded_count=result.loaded_count,
        total_rows=report.total_rows,
        errors=[],
    )
    return JSONResponse(
        status_code=status.HTTP_201_CREATED, content=body.model_dump(mode="json")
    )


@router.get("/instruments", response_model=list[InstrumentSummary])
def list_instruments(
    session: Session = Depends(get_db_session),
) -> list[InstrumentSummary]:
    """List every loaded instrument with its bar count and date range."""
    rows = session.execute(
        select(
            OHLCVBarORM.instrument,
            func.count(OHLCVBarORM.id),
            func.min(OHLCVBarORM.date),
            func.max(OHLCVBarORM.date),
        )
        .group_by(OHLCVBarORM.instrument)
        .order_by(OHLCVBarORM.instrument)
    ).all()
    return [
        InstrumentSummary(
            instrument=instrument,
            name=display_name(instrument),
            multiplier=contract_multiplier(instrument),
            bar_count=bar_count,
            first_date=first_date,
            last_date=last_date,
        )
        for (instrument, bar_count, first_date, last_date) in rows
    ]


# --- Error helpers ---------------------------------------------------------


def _validation_failure_body(instrument: str, report: ValidationReport) -> JSONResponse:
    """422 body carrying typed per-row errors at the TOP level."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "instrument": instrument,
            "loaded_count": 0,
            "total_rows": report.total_rows,
            "errors": [e.model_dump(mode="json") for e in report.errors],
            "error_code": "VALIDATION_FAILED",
        },
    )
