import logging
import os

def get_file_logger(log_path: str, name: str = "exp"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 清空旧 handler，避免重复写入
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger