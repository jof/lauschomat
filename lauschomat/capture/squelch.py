"""
Squelch detection for radio transmissions.
"""
import logging
import time
from enum import Enum
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np
from collections import deque

from lauschomat.capture.audio import AudioFrame
from lauschomat.common.config import SquelchConfig

logger = logging.getLogger(__name__)


class SquelchState(str, Enum):
    """Squelch state enum."""
    CLOSED = "closed"
    OPEN = "open"


class SquelchDetector:
    """Base class for squelch detection."""

    def __init__(self, config: SquelchConfig):
        """Initialize squelch detector with the given configuration."""
        self.config = config
        self.state = SquelchState.CLOSED
        self.open_since: Optional[float] = None
        self.hang_until: Optional[float] = None
        self.last_frame_time: Optional[float] = None
        self.consecutive_open_frames = 0
        self.frame_ms = config.frame_ms
        self.min_open_frames = max(1, config.min_open_ms // config.frame_ms)
        self.hang_frames = max(1, config.hang_ms // config.frame_ms)
        self.max_transmission_frames = max(1, config.max_transmission_ms // config.frame_ms)
        self.frame_count = 0
        self.on_open_callbacks: List[Callable[[], None]] = []
        self.on_close_callbacks: List[Callable[[], None]] = []

        # For periodic logging
        self.last_log_time = time.monotonic()
        self.log_interval = 2.0  # Log audio levels every 2 seconds

    def process(self, frame: AudioFrame) -> Tuple[SquelchState, float]:
        """Process an audio frame and determine squelch state."""
        self.frame_count += 1
        now = time.monotonic()
        self.last_frame_time = now

        # Calculate energy level for this frame
        energy_db = self._calculate_energy(frame)

        # Periodic logging of audio levels
        if now - self.last_log_time >= self.log_interval:
            threshold_indicator = ""
            if energy_db > self.config.threshold_open:
                threshold_indicator = " (ABOVE THRESHOLD)"
            elif energy_db > self.config.threshold_close:
                threshold_indicator = " (above close threshold)"

            # Create a visual meter
            meter_min = -70  # Minimum dB to show on meter
            meter_max = -20  # Maximum dB to show on meter
            meter_width = 30  # Width of the meter in characters

            # Clamp energy level to meter range
            meter_value = max(min(energy_db, meter_max), meter_min)
            # Scale to 0-1 range
            meter_scaled = (meter_value - meter_min) / (meter_max - meter_min)
            # Convert to meter position
            meter_pos = int(meter_scaled * meter_width)

            # Create the meter string
            meter = "[" + "#" * meter_pos + "-" * (meter_width - meter_pos) + "]"

            logger.info(f"Audio level: {energy_db:.1f} dBFS {meter}{threshold_indicator}")
            self.last_log_time = now

        # Apply squelch logic
        if self.state == SquelchState.CLOSED:
            if energy_db > self.config.threshold_open:
                self.consecutive_open_frames += 1
                if self.consecutive_open_frames >= self.min_open_frames:
                    self._open_squelch(now)
            else:
                self.consecutive_open_frames = 0
        else:  # OPEN state
            # Check for max transmission length
            if self.open_since and (now - self.open_since) * 1000 > self.config.max_transmission_ms:
                logger.info("Closing squelch due to max transmission time")
                self._close_squelch(now, reason="max_transmission_time")
                return self.state, energy_db

            # Check for signal drop
            if energy_db < self.config.threshold_close:
                if self.hang_until is None:
                    self.hang_until = now + (self.config.hang_ms / 1000.0)
                    logger.debug(f"Signal dropped below threshold ({energy_db:.1f} dBFS < {self.config.threshold_close} dBFS), starting hang timer for {self.config.hang_ms}ms")
                elif now > self.hang_until:
                    self._close_squelch(now, reason="hang_timeout")
                else:
                    remaining_ms = int((self.hang_until - now) * 1000)
                    logger.debug(f"Signal still below threshold, hang timer: {remaining_ms}ms remaining")
            else:
                # Signal is back above close threshold, reset hang timer
                if self.hang_until is not None:
                    logger.debug(f"Signal back above threshold ({energy_db:.1f} dBFS >= {self.config.threshold_close} dBFS), resetting hang timer")
                self.hang_until = None

        return self.state, energy_db

    def _calculate_energy(self, frame: AudioFrame) -> float:
        """Calculate energy level in dBFS for the frame."""
        # Convert to float32 for calculation if needed
        data = frame.data.astype(np.float32) if frame.data.dtype != np.float32 else frame.data

        # Handle different dtypes for proper scaling to -1.0 to 1.0 range
        if frame.data.dtype == np.int16:
            data = data / 32768.0

        # Calculate RMS
        rms = np.sqrt(np.mean(np.square(data)))

        # Convert to dBFS
        if rms > 0:
            dbfs = 20 * np.log10(rms)
        else:
            dbfs = -120.0  # Arbitrary low value for silence

        return dbfs

    def _open_squelch(self, timestamp: float) -> None:
        """Open the squelch."""
        if self.state == SquelchState.OPEN:
            return

        self.state = SquelchState.OPEN
        self.open_since = timestamp
        self.hang_until = None
        logger.info(f"Squelch OPEN at {timestamp:.3f}")

        # Call registered callbacks
        for callback in self.on_open_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in squelch open callback: {e}")

    def _close_squelch(self, timestamp: float, reason: str = "hang_timeout") -> None:
        """Close the squelch."""
        if self.state == SquelchState.CLOSED:
            return

        duration = timestamp - self.open_since if self.open_since else 0
        self.state = SquelchState.CLOSED
        self.open_since = None
        self.hang_until = None
        self.consecutive_open_frames = 0
        logger.info(f"Squelch CLOSED at {timestamp:.3f} (duration: {duration:.3f}s, reason: {reason})")

        # Call registered callbacks
        for callback in self.on_close_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in squelch close callback: {e}")

    def on_open(self, callback: Callable[[], None]) -> None:
        """Register a callback for squelch open events."""
        self.on_open_callbacks.append(callback)

    def on_close(self, callback: Callable[[], None]) -> None:
        """Register a callback for squelch close events."""
        self.on_close_callbacks.append(callback)

    def reset(self) -> None:
        """Reset the squelch detector state."""
        self.state = SquelchState.CLOSED
        self.open_since = None
        self.hang_until = None
        self.consecutive_open_frames = 0


class EnergySquelchDetector(SquelchDetector):
    """Energy-based squelch detector with hysteresis."""

    def __init__(self, config: SquelchConfig):
        """Initialize energy-based squelch detector."""
        super().__init__(config)
        # Additional smoothing for energy levels
        self.energy_history: Deque[float] = deque(maxlen=5)

    def _calculate_energy(self, frame: AudioFrame) -> float:
        """Calculate smoothed energy level in dBFS."""
        # Get raw energy from parent method
        raw_energy = super()._calculate_energy(frame)

        # Add to history and calculate smoothed value
        self.energy_history.append(raw_energy)
        smoothed = sum(self.energy_history) / len(self.energy_history)

        return smoothed


def create_squelch_detector(config: SquelchConfig) -> SquelchDetector:
    """Factory function to create a squelch detector based on configuration."""
    method = config.method.lower()

    if method == "energy_hysteresis":
        return EnergySquelchDetector(config)
    # Add other detector types here as needed
    else:
        logger.warning(f"Unknown squelch method '{method}', falling back to energy_hysteresis")
        return EnergySquelchDetector(config)
