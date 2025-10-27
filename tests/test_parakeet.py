#!/usr/bin/env python3
"""
Test script for Parakeet TDT model implementation.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add parent directory to path to import from lauschomat
sys.path.insert(0, str(Path(__file__).parent.parent))

from lauschomat.common.config import TranscriptionConfig
from lauschomat.transcribe.model import ParakeetTDTModel, DummyTranscriptionModel

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def test_parakeet(audio_file: str, model_name: str = "nvidia/parakeet-ctc-1.1b", device: str = "cuda:0"):
    """Test Parakeet TDT model on a single audio file."""
    # Create config
    config = TranscriptionConfig(
        enabled=True,
        engine="nemo_parakeet_tdt",
        model_name=model_name,
        device=device,
        batch_size=1,
        language="en-US",
        diarization=False
    )
    
    # Try to import nemo to check if it's available
    try:
        import nemo
        logger.info("NeMo is available, using Parakeet TDT model")
        model = ParakeetTDTModel(config)
    except ImportError:
        logger.warning("NeMo not available, falling back to dummy model")
        logger.warning("To use Parakeet TDT, install NeMo with: pip install nemo_toolkit[asr]")
        model = DummyTranscriptionModel(config)
    
    # Initialize model
    logger.info(f"Initializing model: {model.__class__.__name__}")
    success = model.initialize()
    if not success:
        logger.error("Failed to initialize model")
        return False
    
    # Transcribe audio
    logger.info(f"Transcribing audio file: {audio_file}")
    start_time = time.time()
    result = model.transcribe(audio_file)
    elapsed = time.time() - start_time
    
    if not result:
        logger.error("Transcription failed")
        return False
    
    # Print results
    logger.info(f"Transcription completed in {elapsed:.2f} seconds")
    logger.info(f"Text: {result['text']}")
    logger.info(f"Confidence: {result['confidence']}")
    logger.info(f"Model: {result['model']}")
    logger.info(f"Device: {result['runtime_device']}")
    logger.info(f"Latency: {result['latency_ms']} ms")
    
    # Print word timestamps
    logger.info("Word timestamps:")
    for word in result['words']:
        logger.info(f"  {word['w']}: {word['start']:.2f}s - {word['end']:.2f}s (conf: {word['conf']:.2f})")
    
    # Save result to file
    output_file = Path(audio_file).with_suffix(".transcript.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved transcription to {output_file}")
    
    # Clean up
    model.cleanup()
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Parakeet TDT model")
    parser.add_argument("audio_file", type=str, help="Path to audio file to transcribe")
    parser.add_argument("--model", type=str, default="nvidia/parakeet-ctc-1.1b", 
                        help="Model name or path to .nemo file")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Device to use (cuda:0, cuda:1, cpu, etc.)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        logger.error(f"Audio file not found: {args.audio_file}")
        return 1
    
    # Run test
    success = test_parakeet(args.audio_file, args.model, args.device)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
