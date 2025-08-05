"""
Logging utilities for the high-risk AI healthcare project.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from .config import config

def setup_logger(
    name: str = "high_risk_ai_healthcare",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    
    Returns:
        Configured logger instance
    """
    # Get logging level from config or use default
    log_level = level or config.get("logging.level", "INFO")
    log_format = config.get("logging.format", 
                           "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is None:
        log_dir = Path(config.get("paths.logs", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "high_risk_ai_healthcare") -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger

# Create default logger
logger = get_logger() 