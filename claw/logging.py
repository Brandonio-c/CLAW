"""Structured logging configuration for CLAW."""

import logging
import sys
from typing import Optional
from pathlib import Path

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import config


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    enable_rich: bool = True,
    trace_file: Optional[str] = None
) -> None:
    """Set up structured logging for CLAW.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format type (json, text)
        enable_rich: Enable rich console output
        trace_file: Optional trace file path
    """
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if format_type == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    log_level = getattr(logging, level.upper())
    
    # Create console handler
    if enable_rich:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            markup=True
        )
    else:
        console_handler = logging.StreamHandler(sys.stderr)
    
    console_handler.setLevel(log_level)
    
    # Create formatter
    if format_type == "json":
        formatter = logging.Formatter(
            '%(message)s'  # structlog handles formatting
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Add file handler if trace file specified
    if trace_file:
        trace_path = Path(trace_file)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(trace_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Initialize logging with config
setup_logging(
    level=config.logging.level,
    format_type=config.logging.format,
    enable_rich=config.logging.enable_rich,
    trace_file=config.logging.trace_file
)
