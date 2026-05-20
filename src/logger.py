import logging
import os
import sys

def get_agent_logger(log_file_path: str = None) -> logging.Logger:
    """
    Creates and configures a logger suitable for agentic tracking.
    Outputs to both stdout and a designated log file.
    """
    logger = logging.getLogger("agentic_cad_pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    # Formatting standard
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(module)s:%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(log_file_path)), exist_ok=True)
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
