"""
Transcription model interface and implementations.
"""
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

import torch  # type: ignore
import soundfile as sf  # type: ignore

from lauschomat.common.config import TranscriptionConfig

logger = logging.getLogger(__name__)


class TranscriptionModel(ABC):
    """Abstract base class for transcription models."""
    
    def __init__(self, config: TranscriptionConfig):
        """Initialize transcription model."""
        self.config = config
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the model. Returns success status."""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: Union[str, Path]) -> Optional[Dict]:
        """Transcribe audio file. Returns transcription result or None on failure."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass


class DummyTranscriptionModel(TranscriptionModel):
    """Dummy transcription model for testing."""
    
    def __init__(self, config: TranscriptionConfig):
        """Initialize dummy model."""
        super().__init__(config)
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the model."""
        logger.info("Initializing dummy transcription model")
        self.initialized = True
        return True
    
    def transcribe(self, audio_path: Union[str, Path]) -> Optional[Dict]:
        """Transcribe audio file with dummy output."""
        if not self.initialized:
            logger.error("Model not initialized")
            return None
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None
        
        try:
            # Get audio duration
            with sf.SoundFile(audio_path) as f:
                duration_sec = len(f) / f.samplerate
            
            # Create dummy transcription
            result = {
                "model": "dummy_model",
                "version": "0.1.0",
                "text": f"This is a dummy transcription for {audio_path.name}",
                "confidence": 0.95,
                "words": [
                    {"w": "This", "start": 0.1, "end": 0.3, "conf": 0.95},
                    {"w": "is", "start": 0.4, "end": 0.5, "conf": 0.97},
                    {"w": "a", "start": 0.6, "end": 0.7, "conf": 0.99},
                    {"w": "dummy", "start": 0.8, "end": 1.2, "conf": 0.96},
                    {"w": "transcription", "start": 1.3, "end": 2.0, "conf": 0.94},
                ],
                "latency_ms": int(duration_sec * 100),  # Simulate processing time
                "runtime_device": "cpu"
            }
            
            # Simulate processing time
            time.sleep(min(0.5, duration_sec * 0.1))
            
            return result
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            return None
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up dummy transcription model")
        self.initialized = False


class ParakeetTDTModel(TranscriptionModel):
    """NVIDIA Parakeet TDT model wrapper."""
    
    def __init__(self, config: TranscriptionConfig):
        """Initialize Parakeet TDT model."""
        super().__init__(config)
        self.model = None
        self.initialized = False
        self.device = config.device
        self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.language = config.language
        self.sample_rate = 16000  # Parakeet TDT expects 16kHz audio
        
    def initialize(self) -> bool:
        """Initialize the model."""
        logger.info(f"Initializing Parakeet TDT model '{self.model_name}' on device {self.device}")
        
        try:
            # Import NeMo ASR module
            import torch
            import nemo.collections.asr as nemo_asr
            
            # Set device
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = 'cpu'
            
            # Load the Parakeet TDT model
            # The model_name should be one of the available pretrained models in NeMo
            # or a path to a .nemo file
            logger.info(f"Loading model {self.model_name} on {self.device}")
            try:
                # Try to load as a pretrained model first
                self.model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                    model_name=self.model_name,
                    map_location=torch.device(self.device)
                )
            except Exception as e:
                raise ValueError(f"Model {self.model_name} not found as pretrained: {e}")
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Get model metadata
            self.model_metadata = {
                "name": self.model_name,
                "version": getattr(self.model, "version", "unknown"),
                "type": self.model.__class__.__name__,
                "device": self.device,
                "language": self.language
            }
            
            logger.info(f"Parakeet TDT model loaded successfully: {self.model_metadata}")
            self.initialized = True
            return True
        except ImportError as e:
            logger.error(f"Failed to import NeMo: {e}. Make sure NeMo is installed with 'pip install nemo_toolkit[asr]'")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Parakeet TDT model: {e}")
            return False
    
    def _preprocess_audio(self, audio_path: Path) -> Optional[torch.Tensor]:
        """Preprocess audio file for Parakeet TDT model."""
        try:
            import torch
            import torchaudio
            
            # Load audio file
            logger.debug(f"Loading audio file: {audio_path}")
            
            # Use torchaudio for loading
            waveform, sample_rate = torchaudio.load(audio_path)
            logger.debug(f"Original waveform shape: {waveform.shape}, sample_rate: {sample_rate}")
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.debug(f"After mono conversion: {waveform.shape}")
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                logger.debug(f"Resampling from {sample_rate} to {self.sample_rate}")
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
                logger.debug(f"After resampling: {waveform.shape}")
            
            # Normalize audio (important for model performance)
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)  # Add small epsilon to avoid division by zero
            
            # CRITICAL: Ensure the shape is exactly [batch_size, time_steps] = [1, time]
            # The NeMo model expects shape [batch, time] without any channel dimension
            
            # Log original shape for debugging
            logger.debug(f"Processing waveform with shape: {waveform.shape}")
            
            # First, ensure we have a 2D tensor
            if waveform.dim() == 1:  # [time]
                waveform = waveform.unsqueeze(0)  # Now [1, time]
            elif waveform.dim() == 3:  # [batch, channels, time]
                # This is the problematic case - we need to remove the channel dimension
                waveform = waveform.squeeze(1)  # Remove channel dimension to get [batch, time]
                logger.debug(f"Removed channel dimension, new shape: {waveform.shape}")
            
            # Now we should have a 2D tensor [channels/batch, time]
            if waveform.dim() == 2:
                if waveform.shape[0] > 1:  # Multiple channels
                    # Average channels to get mono
                    waveform = torch.mean(waveform, dim=0, keepdim=True)  # Now [1, time]
                    logger.debug(f"Averaged channels, new shape: {waveform.shape}")
            else:
                # If we still don't have a 2D tensor, something is wrong
                logger.warning(f"Unexpected waveform dimensions: {waveform.shape}, forcing reshape...")
                # Force reshape to [1, time]
                waveform = waveform.reshape(1, -1)
            
            # Final verification
            if waveform.dim() != 2 or waveform.shape[0] != 1:
                logger.warning(f"Final shape check failed: {waveform.shape}, forcing correction...")
                # Last resort - force the correct shape
                if waveform.dim() == 2 and waveform.shape[0] > 1:
                    waveform = waveform[0].unsqueeze(0)  # Take first channel/batch
                else:
                    # Try to reshape completely
                    total_elements = waveform.numel()
                    waveform = waveform.reshape(1, total_elements)
            
            logger.debug(f"Final waveform shape: {waveform.shape}")
            return waveform
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _extract_word_timestamps(self, hypotheses) -> list:
        """Extract word-level timestamps from model output."""
        try:
            # This implementation depends on the specific model output format
            words = []
            
            # Log the type of hypotheses to help debug
            logger.debug(f"Hypotheses type: {type(hypotheses)}")
            if isinstance(hypotheses, list):
                logger.debug(f"First hypothesis type: {type(hypotheses[0])}")
                if hasattr(hypotheses[0], '__dict__'):
                    logger.debug(f"Hypothesis attributes: {hypotheses[0].__dict__.keys()}")
            
            # Handle different types of model outputs
            if isinstance(hypotheses, list) and hasattr(hypotheses[0], 'timestep'):
                # CTC model with timestep information
                logger.debug("Using timestep information from CTC model")
                text = str(hypotheses[0])
                words_list = text.split()
                
                # Extract timesteps if available
                if hasattr(hypotheses[0], 'timestep') and hypotheses[0].timestep:
                    timesteps = hypotheses[0].timestep
                    logger.debug(f"Found {len(timesteps)} timesteps")
                    
                    # Process timesteps into word timestamps
                    # This is model-specific and may need adjustment
                    word_idx = 0
                    current_word = ""
                    word_start = 0
                    
                    for i, (char, time) in enumerate(zip(text, timesteps)):
                        if char == " ":
                            if current_word and word_idx < len(words_list):
                                words.append({
                                    "w": current_word,
                                    "start": word_start * 0.02,  # Convert frame index to seconds (assuming 20ms frames)
                                    "end": time * 0.02,
                                    "conf": 0.9  # Default confidence
                                })
                                word_idx += 1
                                current_word = ""
                                word_start = time
                        else:
                            if not current_word:  # Start of a new word
                                word_start = time
                            current_word += char
                    
                    # Add the last word if any
                    if current_word and word_idx < len(words_list):
                        words.append({
                            "w": current_word,
                            "start": word_start * 0.02,
                            "end": timesteps[-1] * 0.02,
                            "conf": 0.9
                        })
                else:
                    # Fallback to estimation
                    logger.debug("No timesteps found, falling back to estimation")
                    return self._estimate_word_timestamps(text)
            elif hasattr(self.model, "decoding") and hasattr(self.model.decoding, "word_timestamps") and \
                 self.model.decoding.word_timestamps:
                # If the model provides word timestamps directly
                logger.debug("Using word timestamps from model")
                for word_info in hypotheses[0].word_timestamps:
                    words.append({
                        "w": word_info.word,
                        "start": word_info.start_time,
                        "end": word_info.end_time,
                        "conf": word_info.confidence if hasattr(word_info, "confidence") else 0.9
                    })
            else:
                # If no word timestamps, use estimation
                logger.debug("No word timestamps available, using estimation")
                text = hypotheses[0].text if hasattr(hypotheses[0], 'text') else str(hypotheses[0])
                return self._estimate_word_timestamps(text)
            
            return words
        except Exception as e:
            logger.error(f"Error extracting word timestamps: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _estimate_word_timestamps(self, text: str) -> list:
        """Estimate word timestamps based on character counts."""
        if not text:
            return []
            
        # Simple estimation based on character count
        words_list = text.split()
        total_chars = sum(len(word) for word in words_list)
        
        # Estimate duration based on average speaking rate
        # Assuming ~5 characters per second as a rough estimate
        estimated_duration = total_chars / 5.0
        
        # Distribute words evenly across the estimated duration
        words = []
        current_time = 0.1  # Start a bit after beginning
        for word in words_list:
            word_duration = (len(word) / total_chars) * estimated_duration
            words.append({
                "w": word,
                "start": round(current_time, 2),
                "end": round(current_time + word_duration, 2),
                "conf": 0.9  # Default confidence
            })
            current_time += word_duration + 0.1  # Add small gap between words
        
        return words
    
    def transcribe(self, audio_path: Union[str, Path]) -> Optional[Dict]:
        """Transcribe audio file using Parakeet TDT."""
        if not self.initialized:
            logger.error("Model not initialized")
            return None
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None
        
        # Wrap the entire transcription process in a try-except block
        try:
            import torch
            import nemo.collections.asr as nemo_asr
            transcription_start_time = time.time()
            
            # Preprocess audio
            waveform = self._preprocess_audio(audio_path)
            if waveform is None:
                return None
            
            # Get audio duration
            with sf.SoundFile(audio_path) as f:
                duration_sec = len(f) / f.samplerate
            
            # Move to appropriate device
            waveform = waveform.to(torch.device(self.device))
            
            # Ensure correct shape (batch, time) - remove any extra dimensions
            if len(waveform.shape) > 2:
                # If shape is (1, 1, time), reshape to (1, time)
                waveform = waveform.squeeze(1)
            logger.debug(f"Waveform shape after reshaping: {waveform.shape}")
            
            # Run inference
            logger.info(f"Running inference on {audio_path}")
            with torch.no_grad():
                # For CTC models, we need to provide the signal length
                signal_length = torch.tensor([waveform.shape[1]], device=waveform.device)
                logger.debug(f"Audio shape: {waveform.shape}, length: {signal_length}")
                
                # Use the appropriate method for the model type
                try:
                    # Different models have different APIs, try the appropriate method
                    # First, check if model is initialized
                    if self.model is None:
                        logger.error("Model is not initialized")
                        return None
                        
                    # Skip the transcribe method and use direct forward pass instead
                    # This is more reliable for handling shape issues
                    logger.debug("Skipping transcribe method and using direct forward pass")
                    
                    # Go straight to direct forward pass
                    logger.debug("Using direct forward pass")
                    
                    # Double-check waveform shape before forward pass
                    logger.debug(f"Shape before forward pass: {waveform.shape}")
                    if waveform.dim() != 2 or waveform.shape[0] != 1:
                        logger.warning(f"Incorrect shape for forward pass: {waveform.shape}, fixing...")
                        # If 3D tensor, remove the channel dimension
                        if waveform.dim() == 3:
                            waveform = waveform.squeeze(1)
                            logger.debug(f"Removed channel dimension: {waveform.shape}")
                            
                        # If still not 2D with batch size 1, reshape
                        if waveform.dim() != 2 or waveform.shape[0] != 1:
                            # Force reshape to [1, time]
                            total_elements = waveform.numel()
                            waveform = waveform.reshape(1, total_elements)
                            logger.debug(f"Forced reshape to: {waveform.shape}")
                    
                    # Final verification
                    if waveform.dim() != 2 or waveform.shape[0] != 1:
                        logger.error(f"Failed to achieve correct shape: {waveform.shape}")
                        return None
                        
                    # Calculate signal length after shape is fixed
                    signal_length = torch.tensor([waveform.shape[1]], device=waveform.device)
                    logger.debug(f"Signal length: {signal_length}, waveform shape: {waveform.shape}")
                    
                    # Forward pass - only if model is not None
                    if self.model is None:
                        logger.error("Cannot perform direct forward pass: model is None")
                        return None
                        
                    # Use try-except to catch any errors during forward pass and decoding
                    try:
                        with torch.inference_mode():
                            logits, logits_len, greedy_predictions = self.model(
                                input_signal=waveform,
                                input_signal_length=signal_length
                            )
                        
                        # Convert to text using the model's tokenizer or decoder
                        # Different models have different attributes for decoding
                        if hasattr(self.model, 'wer') and hasattr(self.model.wer, 'ctc_decoder_predictions_tensor'):
                            try:
                                # Try with predictions parameter
                                hypotheses = self.model.wer.ctc_decoder_predictions_tensor(
                                    logits, logits_len, predictions=greedy_predictions
                                )
                            except TypeError:
                                # If that fails, try without predictions parameter
                                hypotheses = self.model.wer.ctc_decoder_predictions_tensor(
                                    logits, logits_len
                                )
                        elif hasattr(self.model, 'decoding') and hasattr(self.model.decoding, 'ctc_decoder_predictions_tensor'):
                            try:
                                # Try without predictions parameter
                                hypotheses = self.model.decoding.ctc_decoder_predictions_tensor(
                                    logits, logits_len
                                )
                            except TypeError as e:
                                logger.warning(f"Error with standard decoding: {e}, trying alternative")
                                # Try with predictions parameter as a fallback
                                hypotheses = self.model.decoding.ctc_decoder_predictions_tensor(
                                    logits, logits_len, predictions=greedy_predictions
                                )
                        else:
                            # Last resort - just use the greedy predictions directly
                            # This won't have proper text, but at least it won't crash
                            logger.warning("No decoder found, using raw predictions")
                            hypotheses = [greedy_predictions[0].cpu().numpy()]
                    except Exception as e:
                        logger.error(f"Error in forward pass or decoding: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        return None
                    
                    logger.debug(f"Transcription output type: {type(hypotheses)}")
                    logger.debug(f"Transcription result: {hypotheses}")
                except Exception as e:
                    logger.error(f"Transcription failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return None
            
            # Return the result from direct forward pass
            return self._process_transcription_result(hypotheses, audio_path, duration_sec, transcription_start_time)
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _process_transcription_result(self, hypotheses, audio_path: Path, duration_sec: float, 
                                  start_time: Optional[float] = None) -> Optional[Dict]:
        """Process transcription result into a standardized format.
        
        Args:
            hypotheses: Raw transcription output from the model
            audio_path: Path to the audio file
            duration_sec: Duration of the audio in seconds
            start_time: Optional start time for calculating elapsed time
            
        Returns:
            Standardized transcription result dictionary or None on failure
        """
        # Check if we have valid results
        if not hypotheses or len(hypotheses) == 0:
            logger.warning(f"No transcription result for {audio_path}")
            return None
        
        # Extract text
        text = hypotheses[0].text if hasattr(hypotheses[0], 'text') else str(hypotheses[0])
        
        # Calculate confidence if available
        confidence = getattr(hypotheses[0], 'confidence', 0.9) if hasattr(hypotheses[0], 'confidence') else 0.9
        
        # Extract word timestamps if available
        words = self._extract_word_timestamps(hypotheses)
        
        # Prepare result
        result = {
            "text": text.strip(),
            "confidence": confidence,
            "words": words,
            "duration": duration_sec,
            "model": self.model_name,
            "language": self.language
        }
        
        elapsed_time = time.time() - start_time if start_time else 0
        logger.info(f"Transcription completed in {elapsed_time:.2f}s: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        return result
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up Parakeet TDT model")
        if self.model is not None:
            # Release CUDA memory if using GPU
            if self.device.startswith('cuda'):
                import torch
                torch.cuda.empty_cache()
        self.model = None
        self.initialized = False


def create_transcription_model(config: TranscriptionConfig) -> TranscriptionModel:
    """Factory function to create a transcription model based on configuration."""
    engine = config.engine.lower()
    
    if engine == "nemo_parakeet_tdt":
        # Check if we can import nemo
        try:
            import nemo
            logger.info("NeMo is available, using Parakeet TDT model")
            return ParakeetTDTModel(config)
        except ImportError:
            logger.warning("NeMo not available, falling back to dummy model")
            logger.warning("To use Parakeet TDT, install NeMo with: pip install nemo_toolkit[asr]")
            return DummyTranscriptionModel(config)
    elif engine == "dummy":
        logger.info("Using dummy transcription model")
        return DummyTranscriptionModel(config)
    else:
        logger.warning(f"Unknown transcription engine '{engine}', falling back to dummy model")
        return DummyTranscriptionModel(config)
