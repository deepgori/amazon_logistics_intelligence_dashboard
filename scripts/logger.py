# scripts/logger.py

import logging
import os

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers: # Prevent adding handlers multiple times
        logger.setLevel(os.getenv('LOG_LEVEL', 'INFO').upper()) # Use ENV VAR for level

        _base_dir_logger = os.path.dirname(os.path.abspath(__file__)) 
        _base_dir_logger = os.path.dirname(_base_dir_logger) 
        _log_file = os.path.join(_base_dir_logger, 'app.log')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(os.getenv('LOG_LEVEL', 'INFO').upper())
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        os.makedirs(os.path.dirname(_log_file), exist_ok=True) # Ensure log directory exists
        fh = logging.FileHandler(_log_file)
        fh.setLevel(os.getenv('LOG_LEVEL', 'INFO').upper())
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger