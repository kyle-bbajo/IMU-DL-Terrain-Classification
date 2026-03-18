# -*- coding: utf-8 -*-
"""src/utils/logger.py — 로거"""
from __future__ import annotations
import time, logging, sys
from pathlib import Path


def get_logger(name: str = "imu", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


_logger = get_logger()
def log(msg: str) -> None: _logger.info(msg)
