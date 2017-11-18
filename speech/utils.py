import logging


def get_logger(name):
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  handler = logging.StreamHandler()
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                                "%(funcName)s - %(levelname)s ] %(message)s")
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  return logger

