import logging

from rich.logging import RichHandler


def setup_logger(name: str, level=logging.NOTSET) -> logging.Logger:
    """Sets up a consistent logger with Rich.

    Args:
        name (str): The name of the logger.
        level (optional): Logging level, e.g., logging.INFO or "INFO".

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs if root logger is configured

    if not logger.handlers:
        FORMAT = "[white]%(name)-40s[/]    %(message)s"
        DATEFMT = "[%X]"

        handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
        formatter = logging.Formatter(fmt=FORMAT, datefmt=DATEFMT)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
