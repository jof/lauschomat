"""
Main entry point for the audio capture service.
"""
import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from lauschomat.capture.audio import AudioCapture, list_audio_devices
from lauschomat.capture.recorder import AudioRecorder
from lauschomat.capture.squelch import create_squelch_detector
from lauschomat.capture.transfer import FileTransferManager
from lauschomat.common.config import Config, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


class CaptureService:
    """Main service for audio capture, squelch detection, and recording."""

    def __init__(self, config: Config):
        """Initialize the capture service."""
        self.config = config

        # Ensure required configurations are present
        if not config.audio:
            raise ValueError("Audio configuration is required")
        if not config.squelch:
            raise ValueError("Squelch configuration is required")
        if not config.recording:
            raise ValueError("Recording configuration is required")

        # Create components
        self.audio_capture = AudioCapture(config.audio)
        self.squelch_detector = create_squelch_detector(config.squelch)
        self.recorder = AudioRecorder(config)

        # Create file transfer manager if configured
        self.transfer_manager = None
        if config.transfer:
            self.transfer_manager = FileTransferManager(config)

        # Set up callbacks
        self.squelch_detector.on_open(self._on_squelch_open)
        self.squelch_detector.on_close(self._on_squelch_close)

        # Override audio capture frame handler
        self.audio_capture.on_frame = self._on_audio_frame

        # State
        self.running = False

    def _on_audio_frame(self, frame):
        """Handle incoming audio frames."""
        # Process with squelch detector
        state, energy_db = self.squelch_detector.process(frame)

        # Always pass to recorder for either pre-roll buffering or recording
        self.recorder.process_frame(frame)

    def _on_squelch_open(self):
        """Handle squelch open event."""
        self.recorder.start_recording()

    def _on_squelch_close(self):
        """Handle squelch close event."""
        recorded_file = self.recorder.stop_recording()

        # Queue file for transfer if enabled
        if recorded_file and self.transfer_manager:
            self.transfer_manager.queue_file(recorded_file)
            self.transfer_manager.queue_file(recorded_file.with_suffix(".meta.json"))

    def start(self):
        """Start the capture service."""
        if self.running:
            return

        logger.info("Starting capture service")

        # Start audio capture
        self.audio_capture.start()

        # Start transfer manager if configured
        if self.transfer_manager:
            self.transfer_manager.start()

        self.running = True
        logger.info("Capture service started")

    def stop(self):
        """Stop the capture service."""
        if not self.running:
            return

        logger.info("Stopping capture service")

        # Stop audio capture
        self.audio_capture.stop()

        # Stop recording if active
        if self.recorder.is_recording:
            self.recorder.stop_recording()

        # Stop transfer manager if running
        if self.transfer_manager:
            self.transfer_manager.stop()

        self.running = False
        logger.info("Capture service stopped")


def list_devices_command():
    """List available audio devices and exit."""
    devices = list_audio_devices()
    print("Available audio devices:")
    for i, device in enumerate(devices):
        direction = []
        if device.get('max_input_channels', 0) > 0:
            direction.append("IN")
        if device.get('max_output_channels', 0) > 0:
            direction.append("OUT")

        direction_str = "+".join(direction)
        print(f"{i}: {device['name']} [{direction_str}] - {device.get('hostapi', 'unknown')}")
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Lauschomat Audio Capture Service")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices_command()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Configure file logging
    log_dir = Path(config.app.data_root) / "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "ingest.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # Create and start service
    service = CaptureService(config)

    # Handle signals
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down")
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        service.start()

        # Keep running until interrupted
        while service.running:
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error in capture service: {e}")
        service.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
