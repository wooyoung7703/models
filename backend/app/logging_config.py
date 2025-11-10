import logging
from logging import Logger
from pythonjsonlogger import jsonlogger
from typing import Optional
from .core.config import settings


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Inject common contextual fields
        record.symbol = settings.SYMBOL
        record.exchange_type = settings.EXCHANGE_TYPE
        record.interval = settings.INTERVAL
        record.service = settings.SERVICE_NAME
        record.environment = settings.ENVIRONMENT
        return True


def init_logging() -> Logger:
    root = logging.getLogger()
    # Clear existing handlers (avoid duplicate logs if reloaded)
    for h in list(root.handlers):
        root.removeHandler(h)

    level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    root.setLevel(level)

    handler = logging.StreamHandler()
    if settings.LOG_FORMAT == "json":
        fmt = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(symbol)s %(exchange_type)s %(interval)s %(service)s %(environment)s %(filename)s %(lineno)d"
        )
        handler.setFormatter(fmt)
    else:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s | symbol=%(symbol)s ex=%(exchange_type)s interval=%(interval)s service=%(service)s env=%(environment)s"
        )
        handler.setFormatter(formatter)

    handler.addFilter(ContextFilter())
    root.addHandler(handler)

    # Reduce noisy third-party loggers if needed
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    root.debug("Structured logging initialized")
    return root
