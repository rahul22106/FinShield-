import logging
import os
import sys
from datetime import datetime

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler with UTF-8 encoding (fixes Windows emoji error)
        console_handler = logging.StreamHandler(
            stream=open(sys.stdout.fileno(), 
                       mode='w', 
                       encoding='utf-8', 
                       buffering=1)
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler with UTF-8 encoding
        log_file = os.path.join(
            LOG_DIR,
            f"finshield_{datetime.today().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger