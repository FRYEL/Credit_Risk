import logging
# import pathlib
#
# from dotenv import load_dotenv


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

# set_logger()
#
# dotenv_path = pathlib.Path(__file__) / 'env_local.env'
# if dotenv_path.exists():
#     load_dotenv(dotenv_path=dotenv_path)
# else:
#     dotenv_path = pathlib.Path(__file__) / 'env.env'
#     load_dotenv(dotenv_path=dotenv_path)
