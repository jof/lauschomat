"""
Audio capture and device management.
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd

from lauschomat.common.config import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class AudioFrame:
    """A frame of audio data with timestamp."""
    data: np.ndarray
    timestamp: float  # Monotonic timestamp
    sample_rate: int
    channels: int
    frame_number: int = 0


class AudioCapture:
    """Captures audio from a device using PulseAudio."""

    def __init__(self, config: AudioConfig):
        """Initialize audio capture with the given configuration."""
        self.config = config
        self.device_name = config.device_name
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        self.format_str = config.format
        self.gain_db = config.gain_db

        # Determine numpy dtype from format string
        if self.format_str == "S16_LE":
            self.dtype = np.int16
        elif self.format_str == "FLOAT32_LE":
            self.dtype = np.float32
        else:
            raise ValueError(f"Unsupported audio format: {self.format_str}")

        self.stream: Optional[sd.InputStream] = None
        self.frame_queue = queue.Queue(maxsize=100)  # Buffer up to 100 frames
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_count = 0

        # Check if device exists
        self._check_device()

    def _check_device(self) -> None:
        """Check if the configured device exists."""
        devices = sd.query_devices()
        device_names = [d['name'] for d in devices]

        # Handle device names with hardware IDs (like 'Device Name (hw:X,Y)')
        # Extract just the name part for comparison
        device_name_only = self.device_name.split('(')[0].strip() if '(' in self.device_name else self.device_name

        # Check if the device exists by name
        if self.device_name != "default":
            # First try exact match
            if self.device_name not in device_names:
                # Then try matching just the name part
                matching_devices = [d for d in device_names if device_name_only in d]
                if matching_devices:
                    self.device_name = matching_devices[0]
                    logger.info(f"Using device '{self.device_name}'")
                else:
                    logger.warning(f"Device '{self.device_name}' not found. Available devices: {device_names}")
                    logger.warning("Falling back to default device")
                    self.device_name = "default"

    def start(self) -> None:
        """Start capturing audio."""
        if self.running:
            return

        # Calculate blocksize based on frame_ms if needed
        # For now, use a reasonable default
        blocksize = int(self.sample_rate * 0.02)  # 20ms frames

        try:
            self.stream = sd.InputStream(
                device=self.device_name,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=blocksize,
                callback=self._audio_callback
            )
            self.stream.start()
            self.running = True
            self.thread = threading.Thread(target=self._process_frames, daemon=True)
            self.thread.start()
            logger.info(f"Started audio capture on device '{self.device_name}'")
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            raise

    def stop(self) -> None:
        """Stop capturing audio."""
        if not self.running:
            return

        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

        logger.info("Stopped audio capture")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags) -> None:
        """Callback for the sounddevice InputStream."""
        if status:
            logger.warning(f"Audio callback status: {status}")

        # Apply gain if needed
        if self.gain_db != 0:
            gain_factor = 10 ** (self.gain_db / 20.0)
            indata = indata * gain_factor

        # Get current timestamp - time_info is a C structure in sounddevice, not a dict
        # Use monotonic time as fallback
        timestamp = time.monotonic()

        # Create frame and put in queue
        frame = AudioFrame(
            data=indata.copy(),  # Copy to ensure we own the data
            timestamp=timestamp,
            sample_rate=self.sample_rate,
            channels=self.channels,
            frame_number=self.frame_count
        )
        self.frame_count += 1

        try:
            self.frame_queue.put(frame, block=False)
        except queue.Full:
            logger.warning("Audio frame queue is full, dropping frame")

    def _process_frames(self) -> None:
        """Process frames from the queue."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                self.on_frame(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio frame: {e}")

    def on_frame(self, frame: AudioFrame) -> None:
        """Process a single audio frame. Override in subclasses."""
        pass  # Default implementation does nothing

    def get_frame(self, timeout: float = 1.0) -> Optional[AudioFrame]:
        """Get a single frame from the queue."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class AudioFrameProcessor:
    """Base class for audio frame processors."""

    def __init__(self, config: AudioConfig):
        self.config = config

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Process an audio frame. Override in subclasses."""
        return frame


def list_audio_devices() -> List[dict]:
    """List all available audio devices."""
    return sd.query_devices()


def get_default_device() -> Tuple[str, str]:
    """Get the default input and output device names."""
    try:
        devices = sd.query_devices()
        default_input = devices[sd.default.device[0]]['name']
        default_output = devices[sd.default.device[1]]['name']
        return default_input, default_output
    except Exception as e:
        logger.error(f"Error getting default devices: {e}")
        return "default", "default"
