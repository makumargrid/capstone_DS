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
    
    # Formatting standard
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(module)s:%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add console handler if not already present
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    if not has_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Add file handler if requested and not already present for this path
    if log_file_path:
        # Check if file handler for this path already exists
        abs_path = os.path.abspath(log_file_path)
        has_file_handler = False
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                if os.path.abspath(h.baseFilename) == abs_path:
                    has_file_handler = True
                    break
        if not has_file_handler:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            fh = logging.FileHandler(abs_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
