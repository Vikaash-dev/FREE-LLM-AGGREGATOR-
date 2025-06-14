import logging
import sys
import structlog
from .settings import settings

def setup_logging():
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if sys.stderr.isatty():
        # Pretty printing for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(),
        ]
    else:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging to use structlog's processors
    # This ensures that logs from other libraries are also processed by structlog
    root_logger = logging.getLogger()
    # root_logger.handlers.clear() # Avoid clearing handlers if uvicorn or other libs add theirs

    # Check if handlers are already configured by uvicorn or another process
    if not root_logger.hasHandlers() or all(isinstance(h, logging.NullHandler) for h in root_logger.handlers):
        # If no effective handlers are present, add our default.
        # This avoids duplicate logs if uvicorn/FastAPI already set up a handler.
        root_logger.handlers.clear() # Clear any NullHandlers or other defaults if we are taking over.
        handler = logging.StreamHandler(sys.stdout) # Or sys.stderr
        # The default formatter for structlog's stdlib integration is usually sufficient.
        # If specific formatting is needed for stdlib logs processed by structlog:
        # formatter = structlog.stdlib.ProcessorFormatter.wrap_for_formatter(
        #     structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer()
        # )
        # handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.setLevel(log_level)

    # Capture warnings
    logging.captureWarnings(True)

    # Get a logger instance AFTER configuration
    logger = structlog.get_logger("initialization")
    logger.info("Logging configured", log_level=settings.LOG_LEVEL, environment="dev" if sys.stderr.isatty() else "prod")
