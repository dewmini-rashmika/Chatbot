"""
FastAPI Application Entry Point.

Production configuration includes:
- CORS for React frontend
- Middleware stack (logging, PII scrubbing, rate limiting)
- API routers (versioned under /api/v1)
- Lifespan context for startup/shutdown (DB init, graph init)
- Health check endpoint
"""
import structlog
import sys
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Ensure .env is loaded into os.environ so tools can access API keys (like TAVILY_API_KEY)
load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agents.graph import init_graph
from app.api.v1.endpoints import auth, chat, conversations
from app.core.config import settings
from app.core.logging_config import setup_logging, setup_langsmith
from app.core.middleware import PIIScrubbingMiddleware, RequestLoggingMiddleware
from app.db.database import engine
from app.models.models import Base

setup_logging()
setup_langsmith()
logger = structlog.get_logger(__name__)

# Rate limiter — 60 requests/minute per IP (generous for dev, tighten for prod)
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


from contextlib import asynccontextmanager, AsyncExitStack

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup (DB init, graph init) and shutdown (cleanup).
    """
    async with AsyncExitStack() as stack:
        # ── STARTUP ────────────────────────────────────────────────────────────────
        logger.info("startup_begin", app=settings.app_name, env=settings.app_env)
    
        # Create all DB tables (idempotent in dev; use Alembic for prod migrations)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_tables_ready")
    
        # Initialize the LangGraph multi-agent system
        await init_graph(settings.sync_database_url, stack)
        logger.info("langgraph_initialized")
    
        logger.info("startup_complete", host=settings.app_host, port=settings.app_port)
        yield
    
        # ── SHUTDOWN ───────────────────────────────────────────────────────────────
        await engine.dispose()
        logger.info("shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Production-grade Music Knowledge & Discovery Agent",
        docs_url="/api/docs" if settings.app_env == "development" else None,
        redoc_url="/api/redoc" if settings.app_env == "development" else None,
        lifespan=lifespan,
    )

    # ── Rate Limiting ──────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── CORS ───────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",  # Vite dev server
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Custom Middlewares (order matters — outermost runs first) ──────────────
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(PIIScrubbingMiddleware)

    # ── API Routers ────────────────────────────────────────────────────────────
    prefix = "/api/v1"
    app.include_router(auth.router, prefix=prefix)
    app.include_router(conversations.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)

    # ── Health Check ───────────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "healthy",
            "app": settings.app_name,
            "env": settings.app_env,
            "version": "1.0.0",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,  # MUST BE FALSE ON WINDOWS to ensure SelectorEventLoop is used!
        log_level="info",
    )
