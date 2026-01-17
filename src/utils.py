# src/utils.py
"""
Utility functions for logging configuration.

This module provides logging setup functionality for ML pipeline scripts.
All logs are written to timestamped files and displayed in console.
"""
import logging
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_dir: str = "logs", level=logging.INFO):
    """
    Configure and return a logger that writes to both file and console.
    
    Args:
        name: Logger name (typically __name__)
        log_dir: Directory to store log files
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{name}_{timestamp}.log"
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
