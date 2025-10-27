#!/usr/bin/env python3
"""
Compare different transcription models on the same audio file.
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
logger = logging.getLogger("compare_models")


def transcribe_with_model(audio_path: str, engine: str, model_name: str, device: str, language: str) -> Optional[Dict]:
    """Transcribe audio with the specified model."""
    # Create transcription config
    config = TranscriptionConfig(
        enabled=True,
        engine=engine,
        model_name=model_name,
        device=device,
        batch_size=1,
        language=language,
        diarization=False
    )
    
    # Create and initialize model
    logger.info(f"Creating {engine} model: {model_name}")
    model = create_transcription_model(config)
    
    logger.info("Initializing model...")
    if not model.initialize():
        logger.error("Failed to initialize model")
        return None
    
    # Transcribe audio
    start_time = time.time()
    logger.info(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path)
    elapsed_time = time.time() - start_time
    
    if result:
        result["elapsed_time"] = elapsed_time
    
    # Clean up
    model.cleanup()
    return result


def compare_models(audio_path: str, models: List[Dict]) -> None:
    """Compare multiple models on the same audio file."""
    results = []
    
    for model_config in models:
        logger.info(f"Testing model: {model_config['engine']} - {model_config['model_name']}")
        result = transcribe_with_model(
            audio_path=audio_path,
            engine=model_config["engine"],
            model_name=model_config["model_name"],
            device=model_config["device"],
            language=model_config["language"]
        )
        
        if result:
            results.append({
                "engine": model_config["engine"],
                "model_name": model_config["model_name"],
                "text": result["text"],
                "elapsed_time": result["elapsed_time"],
                "word_count": len(result.get("words", []))
            })
    
    # Print comparison
    if results:
        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        
        for result in results:
            print(f"\nModel: {result['engine']} - {result['model_name']}")
            print(f"Time: {result['elapsed_time']:.2f} seconds")
            print(f"Word count: {result['word_count']}")
            print(f"Text: {result['text']}")
        
        print("\n" + "=" * 80)
    else:
        logger.error("No successful transcriptions to compare")


def main():
    parser = argparse.ArgumentParser(description="Compare transcription models")
    parser.add_argument("audio_file", type=str, help="Path to audio file to transcribe")
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
    
    # Define models to compare
    models = [
        {
            "engine": "whisper",
            "model_name": "openai/whisper-large-v3",
            "device": args.device,
            "language": args.language
        },
        {
            "engine": "granite_speech",
            "model_name": "ibm-granite/granite-speech-3.3-2b",
            "device": args.device,
            "language": args.language
        }
    ]
    
    # Compare models
    compare_models(str(audio_path), models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
