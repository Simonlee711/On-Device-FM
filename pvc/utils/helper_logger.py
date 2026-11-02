"""Logging utilities for downstream evaluation scripts."""

import logging
import os


def create_logger(
    name,
    log_file,
    level=logging.DEBUG,
    format='%(message)s',
    overwrite=True
):
    """Create a logger that writes to ``log_file``.

    Args:
        name: Logger name.
        log_file: Destination file path.
        level: Logging level for the logger and its file handler.
        format: Log record format string.
        overwrite: If True, replace any existing log file; otherwise append.

    Returns:
        logging.Logger: Configured logger that does not propagate to the root logger.
    """
    # Remove existing handlers to avoid duplicate log lines on re-creation.
    logger = logging.getLogger(name)
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
            h.close()

    logger.setLevel(level)

    if overwrite and os.path.exists(log_file):
        os.remove(log_file)

    mode = "w" if overwrite else "a"
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(level)

    formatter = logging.Formatter(format)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
