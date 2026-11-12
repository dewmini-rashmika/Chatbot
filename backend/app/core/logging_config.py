"""
Structured logging setup using structlog + stdlib logging.
Writes JSON logs to both stdout and rotating log files.
"""

import os
import sys
import logging
import logging.handlers
import structlog
from datetime import datetime
from app.core.config import settings


def setup_logging() -> None:
    """Configure structlog with JSON rendering for production, pretty for dev."""
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    app_log_path  = os.path.join(log_dir, "app.log")
    eval_log_path = os.path.join(log_dir, "evaluations.log")

    # ── stdlib root logger ────────────────────────────────────────────────────
    app_handler = logging.handlers.TimedRotatingFileHandler(
        app_log_path, when="midnight", backupCount=7, encoding="utf-8"
    )
    app_handler.suffix = "%Y%m%d"

    console_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[app_handler, console_handler],
    )

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── Evaluation logger (separate file, no rotation) ────────────────────────
    eval_logger = logging.getLogger("evaluations")
    eval_handler = logging.FileHandler(eval_log_path, encoding="utf-8")
    eval_logger.addHandler(eval_handler)
    eval_logger.propagate = False

    # ── structlog processors ──────────────────────────────────────────────────
    processors: list = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.app_env == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def setup_langsmith() -> None:
    """
    Activate LangSmith free-tier tracing by setting the required environment variables.
    LangGraph / LangChain picks these up automatically — no code changes to the graph needed.

    Set LANGCHAIN_API_KEY in your .env to enable.
    If the key is blank or the placeholder, tracing is silently disabled.
    """
    log = structlog.get_logger(__name__)

    placeholder = "your-langsmith-api-key-here"
    api_key = settings.langchain_api_key

    if not api_key or api_key == placeholder:
        log.info("LangSmith tracing disabled — set LANGCHAIN_API_KEY in .env to enable.")
        return

    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGCHAIN_API_KEY"]      = api_key
    os.environ["LANGCHAIN_PROJECT"]      = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"]     = settings.langchain_endpoint
    log.info("LangSmith tracing enabled -> project='%s' endpoint='%s'",
             settings.langchain_project, settings.langchain_endpoint)


def log_evaluation(
    query: str,
    response: str,
    sources: list[str],
    latency_ms: float,
    thread_id: str = "",
) -> None:
    """Helper function to log structured evaluations."""
    eval_logger = logging.getLogger("evaluations")
    eval_logger.info(
        str({
            "ts":         datetime.utcnow().isoformat(),
            "thread_id":  thread_id,
            "query":      query[:200],
            "response":   response[:500],
            "sources":    sources,
            "latency_ms": round(latency_ms, 2),
        })
    )
