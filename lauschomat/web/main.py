"""
Main entry point for the web visualization service.
"""
import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

import uvicorn

from lauschomat.common.config import Config, load_config
from lauschomat.web.server import WebServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Lauschomat Web Visualization")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Configure file logging
    log_dir = Path(config.app.data_root) / "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "web.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # Create web server
    try:
        server = WebServer(config)
    except Exception as e:
        logger.error(f"Failed to create web server: {e}")
        sys.exit(1)

    # Run server directly (not in a thread)
    try:
        logger.info(f"Starting web server at http://{config.web.bind_host}:{config.web.port}")
        uvicorn.run(
            server.app,
            host=config.web.bind_host,
            port=config.web.port,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error running web server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
