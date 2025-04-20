import logging
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("../logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Prevent adding handlers multiple times
        logger.setLevel(logging.DEBUG)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # File handler - logs everything
        log_file = logs_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler - logs info and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger 