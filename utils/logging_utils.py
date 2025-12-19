"""Logging utilities for the VERDICT project.

This module provides centralised logging configuration for consistent
logging across all modules in the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    include_timestamp: bool = True,
) -> logging.Logger:
    """
    Set up logger with console and optional file handlers.
    
    Creates a logger with consistent formatting across the project.
    Supports both console output and file logging for experiment tracking.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
        log_file: Optional path to log file. If provided, logs will be
            written to this file in addition to console output.
        level: Logging level (default: logging.INFO).
        include_timestamp: Whether to include timestamp in log file name.
    
    Returns:
        Configured logger instance.
    
    Example:
        >>> logger = setup_logger(__name__, Path("logs/train.log"))
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create formatter with consistent format
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler for stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log file path provided
    if log_file is not None:
        # Add timestamp to filename if requested
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_file.parent / f"{log_file.stem}_{timestamp}{log_file.suffix}"
        
        # Ensure parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a basic one.
    
    This is a convenience function for getting loggers in modules
    without full configuration.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
    
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class LoggerContext:
    """Context manager for temporarily changing log level."""
    
    def __init__(self, logger: logging.Logger, level: int) -> None:
        """
        Initialise logger context.
        
        Args:
            logger: Logger to modify.
            level: Temporary logging level.
        """
        self.logger = logger
        self.new_level = level
        self.original_level = logger.level
    
    def __enter__(self) -> logging.Logger:
        """Enter context and set new log level."""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore original log level."""
        self.logger.setLevel(self.original_level)
