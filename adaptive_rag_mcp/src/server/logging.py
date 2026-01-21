"""Structured logging configuration using structlog."""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import Processor

from src.config import get_settings


def setup_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Shared processors for all outputs
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
    ]
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Create formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer() if settings.log_level == "DEBUG" 
            else structlog.processors.JSONRenderer(),
        ],
    )
    
    # Apply to root handler
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger instance with the given name."""
    return structlog.get_logger(name)


def bind_logger_context(**kwargs: Any) -> None:
    """Bind variables to the logger context for all subsequent logs in this thread/task."""
    structlog.contextvars.bind_contextvars(**kwargs)


def log_tool_call(
    tool_name: str,
    request_id: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any] | None = None,
    error: str | None = None,
    duration_ms: float | None = None,
) -> None:
    """Log a tool call with structured data."""
    logger = get_logger("tool_call")
    
    log_data = {
        "tool_name": tool_name,
        "request_id": request_id,
        "input_keys": list(input_data.keys()),
    }
    
    if duration_ms is not None:
        log_data["duration_ms"] = round(duration_ms, 2)
    
    if error:
        log_data["error"] = error
        logger.error("tool_call_failed", **log_data)
    elif output_data is not None:
        log_data["output_keys"] = list(output_data.keys())
        logger.info("tool_call_success", **log_data)
    else:
        logger.info("tool_call_started", **log_data)
