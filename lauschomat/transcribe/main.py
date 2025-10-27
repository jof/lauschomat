"""
Main entry point for the transcription service.
"""
import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from lauschomat.common.config import Config, load_config
from lauschomat.transcribe.model import TranscriptionModel, create_transcription_model
from lauschomat.transcribe.watcher import FileWatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


class TranscriptionService:
    """Main service for audio transcription."""
    
    def __init__(self, config: Config):
        """Initialize the transcription service."""
        self.config = config
        
        # Ensure required configurations are present
        if not config.transcription:
            raise ValueError("Transcription configuration is required")
        
        # Set up paths
        self.data_root = Path(config.app.data_root)
        self.incoming_dir = Path(config.app.incoming_dir or self.data_root / "incoming")
        self.processed_dir = Path(config.app.processed_dir or self.data_root / "processed")
        
        # Create directories
        os.makedirs(self.incoming_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Create transcription model
        self.model = create_transcription_model(config.transcription)
        
        # Create file watcher
        self.watcher = FileWatcher(
            watch_dir=self.incoming_dir,
            patterns=["*.wav", "*.meta.json"],
            callback=self._on_new_file
        )
        
        # Set of files currently being processed
        self.in_progress: Set[Path] = set()
        
        # Set of files that have been processed
        self.processed: Set[Path] = set()
        
        # State
        self.running = False
    
    def _on_new_file(self, file_path: Path):
        """Handle new file event."""
        if file_path.suffix.lower() != ".wav":
            # Only process WAV files directly
            return
        
        if file_path in self.in_progress or file_path in self.processed:
            return
        
        # Check if metadata file exists
        meta_file = file_path.with_suffix(".meta.json")
        if not meta_file.exists():
            logger.warning(f"Metadata file not found for {file_path}")
            return
        
        # Process the file
        self._process_file(file_path, meta_file)
    
    def _process_file(self, audio_file: Path, meta_file: Path):
        """Process an audio file and its metadata."""
        try:
            # Mark as in progress
            self.in_progress.add(audio_file)
            
            # Load metadata
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Transcribe audio
            logger.info(f"Transcribing {audio_file}")
            result = self.model.transcribe(audio_file)
            
            if not result:
                logger.error(f"Transcription failed for {audio_file}")
                self.in_progress.remove(audio_file)
                return
            
            # Create output directory structure
            date_str = metadata.get("date", time.strftime("%Y-%m-%d"))
            output_dir = self.processed_dir / date_str
            os.makedirs(output_dir, exist_ok=True)
            
            # Move files to processed directory
            output_audio = output_dir / audio_file.name
            output_meta = output_dir / meta_file.name
            output_transcript = output_dir / audio_file.name.replace(".wav", ".transcript.json")
            
            # Write transcription result
            with open(output_transcript, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Copy audio and metadata files
            if not output_audio.exists():
                shutil.copy2(audio_file, output_audio)
            if not output_meta.exists():
                shutil.copy2(meta_file, output_meta)
            
            # Update index
            self._update_index(metadata, result, output_audio, output_transcript)
            
            # Mark as processed
            self.processed.add(audio_file)
            logger.info(f"Processed {audio_file}")
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
        finally:
            # Remove from in progress
            if audio_file in self.in_progress:
                self.in_progress.remove(audio_file)
    
    def _update_index(self, metadata: Dict, transcription: Dict, audio_path: Path, transcript_path: Path):
        """Update the index with transcription information."""
        date = metadata.get("date")
        if not date:
            return
        
        index_file = self.data_root / "indices" / f"transmissions.{date}.jsonl"
        
        # Create index entry
        index_entry = {
            "id": metadata["id"],
            "date": metadata["date"],
            "timestamp_utc": metadata["timestamp_utc"],
            "session_id": metadata.get("session_id", "unknown"),
            "audio_path": str(audio_path.relative_to(self.data_root)),
            "metadata_path": str(audio_path.with_suffix(".meta.json").relative_to(self.data_root)),
            "transcription_path": str(transcript_path.relative_to(self.data_root)),
            "duration_sec": metadata.get("duration_sec", 0),
            "rms_dbfs": metadata.get("rms_dbfs", 0),
            "peak_dbfs": metadata.get("peak_dbfs", 0),
            "device": metadata.get("device", "unknown"),
            "text": transcription.get("text", ""),
            "confidence": transcription.get("confidence", 0),
            "model": transcription.get("model", "unknown")
        }
        
        # Append to index file
        with open(index_file, 'a') as f:
            f.write(json.dumps(index_entry) + "\n")
    
    def start(self):
        """Start the transcription service."""
        if self.running:
            return
        
        logger.info("Starting transcription service")
        
        # Initialize model
        self.model.initialize()
        
        # Start file watcher
        self.watcher.start()
        
        self.running = True
        logger.info("Transcription service started")
    
    def stop(self):
        """Stop the transcription service."""
        if not self.running:
            return
        
        logger.info("Stopping transcription service")
        
        # Stop file watcher
        self.watcher.stop()
        
        # Clean up model
        self.model.cleanup()
        
        self.running = False
        logger.info("Transcription service stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Lauschomat Transcription Service")
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
    file_handler = logging.FileHandler(log_dir / "transcribe.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Create and start service
    service = TranscriptionService(config)
    
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
        logger.error(f"Error in transcription service: {e}")
        service.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
