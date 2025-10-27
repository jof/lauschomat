"""
Main entry point for the development mode.

This module combines the capture and transcription services into a single process
for easier development and testing.
"""
import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from lauschomat.capture.audio import AudioCapture
from lauschomat.capture.recorder import AudioRecorder
from lauschomat.capture.squelch import create_squelch_detector
from lauschomat.common.config import Config, load_config
from lauschomat.transcribe.model import create_transcription_model
from lauschomat.web.server import WebServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level, will be overridden by command line args
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


class DevService:
    """Combined service for development mode."""
    
    def __init__(self, config: Config):
        """Initialize the development service."""
        self.config = config
        
        # Ensure required configurations are present
        if not config.audio:
            raise ValueError("Audio configuration is required")
        if not config.squelch:
            raise ValueError("Squelch configuration is required")
        if not config.recording:
            raise ValueError("Recording configuration is required")
        if not config.transcription:
            raise ValueError("Transcription configuration is required")
        
        # Create data directories
        self.data_root = Path(config.app.data_root)
        os.makedirs(self.data_root, exist_ok=True)
        os.makedirs(self.data_root / "recordings", exist_ok=True)
        os.makedirs(self.data_root / "indices", exist_ok=True)
        os.makedirs(self.data_root / "logs", exist_ok=True)
        
        # Create components
        self.audio_capture = AudioCapture(config.audio)
        self.squelch_detector = create_squelch_detector(config.squelch)
        self.recorder = AudioRecorder(config)
        self.transcription_model = create_transcription_model(config.transcription)
        
        # Web server (optional)
        self.web_server = None
        if config.web and config.web.enabled:
            self.web_server = WebServer(config)
        
        # Set up callbacks
        self.squelch_detector.on_open(self._on_squelch_open)
        self.squelch_detector.on_close(self._on_squelch_close)
        
        # Override audio capture frame handler
        self.audio_capture.on_frame = self._on_audio_frame
        
        # State
        self.running = False
        self.pending_transcriptions = []
        self.transcription_thread = None
    
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
        
        # Queue for transcription if available
        if recorded_file:
            self.pending_transcriptions.append(recorded_file)
    
    def _transcription_worker(self):
        """Background worker for transcription."""
        while self.running:
            try:
                # Check for pending transcriptions
                if self.pending_transcriptions:
                    audio_file = self.pending_transcriptions.pop(0)
                    meta_file = audio_file.with_suffix(".meta.json")
                    
                    if audio_file.exists() and meta_file.exists():
                        logger.info(f"Transcribing {audio_file}")
                        
                        # Load metadata
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Transcribe
                        result = self.transcription_model.transcribe(audio_file)
                        
                        if result:
                            # Write transcription result
                            transcript_file = audio_file.with_suffix(".transcript.json")
                            with open(transcript_file, 'w') as f:
                                json.dump(result, f, indent=2)
                            
                            logger.info(f"Transcription complete: {result.get('text', '')}")
                        else:
                            logger.error(f"Transcription failed for {audio_file}")
                
                # Sleep briefly
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in transcription worker: {e}")
                time.sleep(1.0)  # Avoid tight loop on error
    
    # Index file functionality removed - using direct file access instead
    
    def start(self):
        """Start the development service."""
        if self.running:
            return
        
        logger.info("Starting development service")
        
        # Initialize transcription model
        self.transcription_model.initialize()
        
        # Start audio capture
        self.audio_capture.start()
        
        # Start transcription thread
        self.running = True
        self.transcription_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self.transcription_thread.start()
        
        # Start web server if enabled
        if self.web_server:
            self.web_server.start()
        
        logger.info("Development service started")
    
    def stop(self):
        """Stop the development service."""
        if not self.running:
            return
        
        logger.info("Stopping development service")
        
        # Stop audio capture
        self.audio_capture.stop()
        
        # Stop recording if active
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        
        # Stop transcription thread
        self.running = False
        if self.transcription_thread:
            self.transcription_thread.join(timeout=2.0)
        
        # Clean up transcription model
        self.transcription_model.cleanup()
        
        # Stop web server if running
        if self.web_server:
            self.web_server.stop()
        
        logger.info("Development service stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Lauschomat Development Mode")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Set the logging level")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output (same as --log-level DEBUG)")
    args = parser.parse_args()
    
    if args.list_devices:
        from lauschomat.capture.audio import list_audio_devices
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
    
    # Set logging level
    log_level = args.log_level
    if args.verbose:
        log_level = "DEBUG"
    
    # Get the numeric logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}")
        sys.exit(1)
    
    # Set the root logger level
    logging.getLogger().setLevel(numeric_level)
    
    # Load configuration
    try:
        config = load_config(args.config)
        
        # Ensure we're in development mode
        config.app.mode = "development"
        
        # Set default paths for development if not specified
        if not config.app.data_root or config.app.data_root == "/var/lib/lauschomat":
            config.app.data_root = "./data"
        if not config.app.tmp_dir or config.app.tmp_dir == "/var/tmp/lauschomat":
            config.app.tmp_dir = "./tmp"
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Configure file logging
    log_dir = Path(config.app.data_root) / "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "dev.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Create and start service
    service = DevService(config)
    
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
        while True:
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error in development service: {e}")
        service.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
