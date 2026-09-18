import logging
from logging.handlers import TimedRotatingFileHandler
import os
import structlog
import sys
from datetime import datetime

def setup_logging():
    # Create logs directory
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 1. Setup standard Python logging for daily rotation
    # App log
    app_log_path = os.path.join(log_dir, "app.log")
    app_handler = TimedRotatingFileHandler(app_log_path, when="midnight", interval=1, backupCount=30)
    app_handler.suffix = "%Y-%m-%d"
    
    # Eval log (for QA and evaluations)
    eval_log_path = os.path.join(log_dir, "evaluations.log")
    eval_handler = TimedRotatingFileHandler(eval_log_path, when="midnight", interval=1, backupCount=30)
    eval_handler.suffix = "%Y-%m-%d"

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[app_handler, console_handler],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Separate logger specifically for evaluations
    eval_logger = logging.getLogger("evaluations")
    eval_logger.setLevel(logging.INFO)
    eval_logger.propagate = False # Don't double log to app
    eval_logger.addHandler(eval_handler)

    # 2. Configure structlog to bridge to standard logging
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer() # JSON format for the file
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def log_evaluation(thread_id: str, query: str, response: str, sources: list, latency_ms: float = 0.0):
    """
    Helper function to log structured evaluations.
    """
    eval_logger = structlog.get_logger("evaluations")
    eval_logger.info("evaluation_record",
                     thread_id=thread_id,
                     query=query,
                     response=response,
                     sources=sources,
                     latency_ms=latency_ms,
                     timestamp=datetime.utcnow().isoformat())
