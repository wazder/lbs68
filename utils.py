"""
Utility functions and helpers for the Luggage Analysis System
"""

import os
import sys
import logging
import warnings
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from contextlib import contextmanager
import functools
import torch


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured logging for the application.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional file to write logs to
        format_string: Custom format string for log messages
    
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger('luggage_analysis')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log all levels to file
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Suppress third-party warnings unless in debug mode
    if level.upper() != 'DEBUG':
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
    
    return logger


def validate_image_file(file_path: Union[str, Path]) -> bool:
    """
    Validate if a file is a valid image file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            return False
            
        # Check file size (should be > 0 and < 50MB)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb == 0 or size_mb > 50:
            return False
            
        # Check extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        if file_path.suffix.lower() not in valid_extensions:
            return False
            
        # Try to open with PIL (basic validation)
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()  # Verify it's a valid image
            
        return True
        
    except Exception:
        return False


def validate_directory(dir_path: Union[str, Path], create_if_missing: bool = False) -> bool:
    """
    Validate if a directory path is valid.
    
    Args:
        dir_path: Path to the directory
        create_if_missing: Create directory if it doesn't exist
        
    Returns:
        True if valid directory, False otherwise
    """
    try:
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            if create_if_missing:
                dir_path.mkdir(parents=True, exist_ok=True)
                return True
            else:
                return False
                
        return dir_path.is_dir()
        
    except Exception:
        return False


def get_image_files(directory: Union[str, Path]) -> List[Path]:
    """
    Get all valid image files from a directory.
    
    Args:
        directory: Path to search for images
        
    Returns:
        List of valid image file paths
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for file_path in directory.iterdir():
        if (file_path.is_file() and 
            file_path.suffix.lower() in image_extensions and 
            validate_image_file(file_path)):
            image_files.append(file_path)
    
    return sorted(image_files)


@contextmanager
def memory_cleanup():
    """
    Context manager for memory cleanup after operations.
    """
    import gc
    try:
        yield
    finally:
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True):
    """
    Decorator to retry function calls on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger('luggage_analysis')
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    wait_time = delay * (2 ** attempt) if exponential_backoff else delay
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
            
        return wrapper
    return decorator


def safe_file_operation(operation_func, file_path: Union[str, Path], *args, **kwargs):
    """
    Safely perform file operations with error handling.
    
    Args:
        operation_func: Function to perform on the file
        file_path: Path to the file
        *args, **kwargs: Additional arguments for the operation
        
    Returns:
        Result of operation or None if failed
    """
    logger = logging.getLogger('luggage_analysis')
    
    try:
        file_path = Path(file_path)
        
        # Basic validation
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return None
            
        return operation_func(file_path, *args, **kwargs)
        
    except PermissionError:
        logger.error(f"Permission denied accessing file: {file_path}")
        return None
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 1:
        return f"{seconds:.2f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total = total_items
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.logger = logging.getLogger('luggage_analysis')
        
    def update(self, increment: int = 1, item_description: str = ""):
        """Update progress."""
        self.current += increment
        elapsed = time.time() - self.start_time
        
        if self.total > 0:
            percentage = (self.current / self.total) * 100
            if elapsed > 0:
                rate = self.current / elapsed
                eta = (self.total - self.current) / rate if rate > 0 else 0
                self.logger.info(
                    f"{self.description}: {self.current}/{self.total} "
                    f"({percentage:.1f}%) - ETA: {format_duration(eta)} - {item_description}"
                )
            else:
                self.logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - {item_description}")
        else:
            self.logger.info(f"{self.description}: {self.current} items processed - {item_description}")
    
    def finish(self):
        """Mark progress as finished."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        self.logger.info(
            f"{self.description} completed: {self.current} items in {format_duration(elapsed)} "
            f"({rate:.1f} items/sec)"
        )


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are available.
    
    Returns:
        Dictionary with dependency availability status
    """
    dependencies = {
        'torch': False,
        'transformers': False,
        'segment_anything': False,
        'cv2': False,
        'sklearn': False,
        'numpy': False,
        'PIL': False,
        'faiss': False
    }
    
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    try:
        from transformers import CLIPModel
        dependencies['transformers'] = True
    except ImportError:
        pass
    
    try:
        from segment_anything import SamPredictor
        dependencies['segment_anything'] = True
    except ImportError:
        pass
    
    try:
        import cv2
        dependencies['cv2'] = True
    except ImportError:
        pass
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        dependencies['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    try:
        from PIL import Image
        dependencies['PIL'] = True
    except ImportError:
        pass
    
    try:
        import faiss
        dependencies['faiss'] = True
    except ImportError:
        pass
    
    return dependencies