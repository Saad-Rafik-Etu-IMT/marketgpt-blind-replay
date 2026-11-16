"""FastAPI application factory for the Market Replay Trading Lab backend.

``create_app`` is a factory (not a module-level singleton only) so the
test suite can build fully isolated application instances. A module-level
``app`` is also exported for ``uvicorn src.main:app``.
"""

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.api.errors import register_error_handlers
from src.api.routes_health import router as health_router
from src.api.routes_market import router as market_router
from src.api.routes_orders import router as orders_router
from src.api.routes_portfolio import router as portfolio_router
from src.api.routes_replay import router as replay_router
from src.api.routes_risk import router as risk_router
from src.config import get_settings
from src.logging_config import configure_logging

# The Next.js dev server. No wildcard: V1 is local-only by design.
ALLOWED_ORIGINS: list[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

logger = logging.getLogger("market_replay.request")


def create_app() -> FastAPI:
    """Build and configure a FastAPI application instance."""
    configure_logging()

    app = FastAPI(
        title="MarketGPT Blind Replay",
        description="Local-only educational replay benchmark foundation.",
        version=__version__,
    )

    register_error_handlers(app)

    allowed_origins = ALLOWED_ORIGINS.copy()
    frontend_origin = get_settings().frontend_origin
    if frontend_origin and frontend_origin not in allowed_origins:
        allowed_origins.append(frontend_origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Compress larger JSON payloads (e.g. the equity-curve history) so the
    # polled read endpoints stay light on the wire.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def log_requests(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Log every request's outcome and turn unhandled errors into 500s.

        A crashing handler is logged with a traceback and returned as a
        flat JSON error rather than leaking a bare stack trace, so a failed
        step/order is always traceable in the logs.
        """
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "unhandled error %s %s (%.1fms)",
                request.method,
                request.url.path,
                elapsed_ms,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error_code": "INTERNAL_ERROR",
                    "detail": "an internal error occurred",
                },
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # Every resource group is registered here under the /api prefix.
    app.include_router(health_router, prefix="/api")
    app.include_router(market_router, prefix="/api")
    app.include_router(replay_router, prefix="/api")
    app.include_router(orders_router, prefix="/api")
    app.include_router(portfolio_router, prefix="/api")
    app.include_router(risk_router, prefix="/api")

    return app


app = create_app()
