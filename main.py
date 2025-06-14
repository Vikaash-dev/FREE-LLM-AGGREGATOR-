#!/usr/bin/env python3
"""
Main entry point for the LLM API Aggregator.
"""

import argparse
import asyncio
# import logging # Removed standard logging
import sys
import structlog # Added structlog if any specific logging is needed here
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Logging is now configured by src.config.logging_config via src.api.server
from src.api.server import run_server

# It's generally good practice for main entry points to also get a logger,
# even if just for a few messages. setup_logging() in server.py should cover this.
logger = structlog.get_logger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="LLM API Aggregator")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    # --log-level argument might be less relevant here if uvicorn/FastAPI handles level via settings,
    # but can be kept if direct uvicorn log level override is desired.
    # The application's structlog level is set from settings in logging_config.py.
    parser.add_argument("--log-level", default=None, help="Override server log level (e.g., info, debug)")
    
    args = parser.parse_args()
    
    # Logging is initialized when src.api.server is imported, which calls setup_logging().
    # No explicit setup_logging(args.log_level) call needed here anymore.
    # The log level for uvicorn itself can be passed to run_server if needed,
    # or configured through environment variables that uvicorn/FastAPI respects.
    
    # Run server
    logger.info("Starting LLM API Aggregator via main.py", host=args.host, port=args.port, reload=args.reload, log_level_override=args.log_level)
    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()