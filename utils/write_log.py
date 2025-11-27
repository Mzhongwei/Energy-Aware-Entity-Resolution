import logging
import os
from logging.handlers import RotatingFileHandler


def write_log(log_path, dir_name, file_name):
    path = f"{log_path}/{dir_name}"
    os.makedirs(path, exist_ok=True)
   
    # log configuration
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # create handler
        handler = RotatingFileHandler(f"{path}/{file_name}.log", maxBytes=5 * 1024 * 1024, backupCount=10)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger
