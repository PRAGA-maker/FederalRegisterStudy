"""
Centralized logging configuration for the Federal Register Study pipeline.

This module provides:
- Consistent log formatting across all pipeline modules
- Year-prefixed log messages for multi-year runs
- Support for verbose/quiet modes
- Integration with tqdm progress bars

Example:
    >>> from stratification_scripts.logging_utils import get_logger, setup_logging
    >>> 
    >>> # Setup logging at module level
    >>> setup_logging(verbose=True, year=2024)
    >>> logger = get_logger(__name__)
    >>> 
    >>> logger.info("Starting pipeline")
    [2024-12-30 10:30:45] [2024] INFO: Starting pipeline
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


# Module-level state for current year context
_current_year: Optional[int] = None


class YearContextFilter(logging.Filter):
    """
    Logging filter that adds year context to log records.
    
    This filter adds a 'year' attribute to each log record, which can be
    used in the log format string.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add year context to log record."""
        record.year = _current_year or "----"
        return True


class TqdmLoggingHandler(logging.StreamHandler):
    """
    Logging handler that writes through tqdm to avoid progress bar corruption.
    
    When tqdm progress bars are active, this handler uses tqdm.write()
    to output log messages without breaking the progress display.
    """
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record, using tqdm.write if available."""
        try:
            msg = self.format(record)
            try:
                from tqdm import tqdm
                tqdm.write(msg, file=self.stream)
            except ImportError:
                self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def set_year_context(year: Optional[int]) -> None:
    """
    Set the current year context for log messages.
    
    Args:
        year: Year to include in log messages, or None to clear.
    """
    global _current_year
    _current_year = year


def get_year_context() -> Optional[int]:
    """
    Get the current year context.
    
    Returns:
        Current year context or None if not set.
    """
    return _current_year


def setup_logging(
    level: int = logging.INFO,
    verbose: bool = False,
    quiet: bool = False,
    year: Optional[int] = None,
) -> None:
    """
    Configure logging for the pipeline with consistent formatting.
    
    This should be called once at the start of pipeline execution.
    It configures the root logger and sets up appropriate formatting.
    
    Args:
        level: Base logging level (default: INFO)
        verbose: If True, set level to DEBUG
        quiet: If True, set level to WARNING
        year: Year context to include in log messages
    
    Side Effects:
        Configures the root logger
        Sets module-level year context
    """
    # Set year context
    if year is not None:
        set_year_context(year)
    
    # Determine effective level
    if verbose:
        effective_level = logging.DEBUG
    elif quiet:
        effective_level = logging.WARNING
    else:
        effective_level = level
    
    # Create formatter with year context
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(year)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Create handler
    handler = TqdmLoggingHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(YearContextFilter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)
    
    # Remove existing handlers to avoid duplicates
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)
    
    root_logger.addHandler(handler)
    
    # Also configure our package logger
    package_logger = logging.getLogger("stratification_scripts")
    package_logger.setLevel(effective_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Configured logger instance.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)
    
    # Ensure the year filter is attached if not already
    has_filter = any(isinstance(f, YearContextFilter) for f in logger.filters)
    if not has_filter:
        logger.addFilter(YearContextFilter())
    
    return logger


def log_banner(logger: logging.Logger, title: str, char: str = "=", width: int = 60) -> None:
    """
    Log a banner/section header.
    
    Args:
        logger: Logger to use
        title: Title text for the banner
        char: Character to use for the border (default: "=")
        width: Width of the banner (default: 60)
    
    Example:
        >>> log_banner(logger, "STAGE 1: FETCHING DOCUMENTS")
        ============================================================
        STAGE 1: FETCHING DOCUMENTS
        ============================================================
    """
    border = char * width
    logger.info(border)
    logger.info(title)
    logger.info(border)


def log_step(logger: logging.Logger, step: int, total: int, description: str) -> None:
    """
    Log a pipeline step with progress indicator.
    
    Args:
        logger: Logger to use
        step: Current step number
        total: Total number of steps
        description: Description of the step
    
    Example:
        >>> log_step(logger, 1, 4, "Fetching documents")
        Step 1/4: Fetching documents...
    """
    logger.info(f"Step {step}/{total}: {description}...")


def log_completion(logger: logging.Logger, message: str) -> None:
    """
    Log a completion message with checkmark.
    
    Args:
        logger: Logger to use
        message: Completion message
    
    Example:
        >>> log_completion(logger, "Year 2024 complete")
        ✓ Year 2024 complete
    """
    logger.info(f"[DONE] {message}")


def log_error(logger: logging.Logger, message: str, exc: Optional[Exception] = None) -> None:
    """
    Log an error message with optional exception details.
    
    Args:
        logger: Logger to use
        message: Error message
        exc: Optional exception to include
    
    Example:
        >>> log_error(logger, "API call failed", exc)
        ERROR: API call failed - ConnectionError: timeout
    """
    if exc:
        logger.error(f"{message} - {type(exc).__name__}: {exc}")
    else:
        logger.error(message)


def log_warning(logger: logging.Logger, message: str) -> None:
    """
    Log a warning message.
    
    Args:
        logger: Logger to use
        message: Warning message
    
    Example:
        >>> log_warning(logger, "Low merge rate detected")
        ⚠ Low merge rate detected
    """
    logger.warning(f"⚠ {message}")

