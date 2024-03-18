import logging


def set_logger():
    logger = logging.getLogger(__file__)

    logger.setLevel("DEBUG")
    handler = logging.StreamHandler()
    log_format = "%(asctime)s %(levelname)s -- %(message)s"
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


LOGGER = set_logger()

__al__ = ['LOGGER']
