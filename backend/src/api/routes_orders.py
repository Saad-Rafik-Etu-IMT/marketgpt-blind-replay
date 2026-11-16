"""HTTP routes for orders.

* POST   /api/orders                  — submit an order
* GET    /api/orders?session_id=...   — list orders for a session
* DELETE /api/orders/{order_id}       — cancel a pending order
"""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session
from src.api.errors import AppError
from src.schemas.order import CreateOrderRequest, OrderRead
from src.services.order_service import (
    OrderRejectedError,
    cancel_order,
    list_orders,
    submit_order,
)

router = APIRouter(prefix="/orders", tags=["orders"])


@router.post(
    "",
    response_model=OrderRead,
    status_code=status.HTTP_201_CREATED,
)
def post_order(
    body: CreateOrderRequest,
    session: Session = Depends(get_db_session),
) -> OrderRead:
    try:
        return submit_order(session, body)
    except OrderRejectedError as exc:
        raise AppError(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "ORDER_REJECTED", str(exc)
        ) from exc


@router.get("", response_model=list[OrderRead])
def get_orders(
    session_id: str = Query(...),
    session: Session = Depends(get_db_session),
) -> list[OrderRead]:
    return list_orders(session, session_id)


@router.delete("/{order_id}")
def delete_order(
    order_id: str, session: Session = Depends(get_db_session)
) -> OrderRead:
    try:
        return cancel_order(session, order_id)
    except LookupError as exc:
        raise AppError(status.HTTP_404_NOT_FOUND, "ORDER_NOT_FOUND", str(exc)) from exc
    except OrderRejectedError as exc:
        raise AppError(
            status.HTTP_409_CONFLICT, "ORDER_NOT_CANCELLABLE", str(exc)
        ) from exc
