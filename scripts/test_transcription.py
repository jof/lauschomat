#!/usr/bin/env python3
"""
Test script for transcription models.
This script allows you to test different transcription models on audio files.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lauschomat.common.config import TranscriptionConfig
from lauschomat.transcribe.model import create_transcription_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test_transcription")


def main():
    parser = argparse.ArgumentParser(description="Test transcription models")
    parser.add_argument("audio_file", type=str, help="Path to audio file to transcribe")
    parser.add_argument(
        "--engine", 
        type=str, 
        default="whisper", 
        choices=["whisper", "granite_speech", "nemo_parakeet_tdt", "dummy"],
        help="Transcription engine to use"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="openai/whisper-large-v3",
        help="Model name or path (e.g., openai/whisper-large-v3, IBM/granite-13b-speech-3.3)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="Device to run inference on (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--language", 
        type=str, 
        default="en-US",
        help="Language code"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 1
    
    # Create transcription config
    config = TranscriptionConfig(
        enabled=True,
        engine=args.engine,
        model_name=args.model,
        device=args.device,
        batch_size=1,
        language=args.language,
        diarization=False
    )
    
    # Create and initialize model
    logger.info(f"Creating {args.engine} model: {args.model}")
    model = create_transcription_model(config)
    
    logger.info("Initializing model...")
    if not model.initialize():
        logger.error("Failed to initialize model")
        return 1
    
    # Transcribe audio
    logger.info(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path)
    
    if result:
        logger.info("Transcription result:")
        logger.info(f"Text: {result['text']}")
        logger.info(f"Confidence: {result.get('confidence', 'N/A')}")
        logger.info(f"Model: {result.get('model', 'N/A')}")
        logger.info(f"Duration: {result.get('duration', 'N/A')} seconds")
        logger.info(f"Word count: {len(result.get('words', []))}")
    else:
        logger.error("Transcription failed")
        return 1
    
    # Clean up
    model.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
