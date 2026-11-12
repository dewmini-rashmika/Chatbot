"""
FastAPI Application Entry Point.

Production configuration includes:
- CORS for React frontend
- Middleware stack (logging, PII scrubbing, rate limiting)
- API routers (versioned under /api/v1)
- Lifespan context for startup/shutdown (DB init, graph init)
- Health check endpoint
"""

import sys
import asyncio
import platform
import contextlib
import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from app.core.config import settings
from app.core.logging_config import setup_logging, setup_langsmith
from app.core.middleware import RequestLoggingMiddleware, PIIScrubbingMiddleware
from app.agents.graph import init_graph
from app.db.database import engine, Base
from app.api.v1.endpoints import auth, chat, conversations

load_dotenv()
setup_logging()
logger = structlog.get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup (DB init, graph init) and shutdown (cleanup).
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    setup_langsmith()

    # Create all DB tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_ready")

    # Build the LangGraph agent graph
    await init_graph()
    logger.info("langgraph_initialized")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await engine.dispose()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="Production-grade Music Knowledge & Discovery Agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── Rate limiting ─────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, lambda req, exc: exc)

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",   # Vite dev server
            "http://localhost:3000",   # CRA dev server
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Custom ASGI middleware ────────────────────────────────────────────────
    app.add_middleware(PIIScrubbingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # ── API routers ───────────────────────────────────────────────────────────
    prefix = "/api/v1"
    app.include_router(auth.router,          prefix=prefix)
    app.include_router(chat.router,          prefix=prefix)
    app.include_router(conversations.router, prefix=prefix)

    @app.get("/health", tags=["health"])
    async def health_check():
        return {
            "status":  "ok",
            "app":     settings.app_name,
            "env":     settings.app_env,
            "models": {
                "supervisor": settings.supervisor_model,
                "rag":        settings.rag_model,
                "reviewer":   settings.reviewer_model,
                "memory":     settings.memory_model,
                "sql":        settings.sql_model,
            },
        }

    return app


app = create_app()

if __name__ == "__main__":
    # Windows needs SelectorEventLoop for asyncpg compatibility
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=(settings.app_env == "development"),
    )
