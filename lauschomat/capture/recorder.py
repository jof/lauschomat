"""
Audio recording functionality.
"""
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np  # type: ignore
import soundfile as sf  # type: ignore

from lauschomat.common.config import Config, RecordingConfig
from lauschomat.capture.audio import AudioFrame

logger = logging.getLogger(__name__)


class PreRollBuffer:
    """Circular buffer for storing pre-roll audio data."""

    def __init__(self, ms: int, sample_rate: int, channels: int, dtype=np.float32):
        """Initialize pre-roll buffer.

        Args:
            ms: Buffer size in milliseconds
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            dtype: Numpy data type for audio samples
        """
        self.ms = ms
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype

        # Calculate buffer size in frames
        self.frames = int((ms / 1000.0) * sample_rate)

        # Create buffer
        self.buffer = np.zeros((self.frames, channels), dtype=dtype)
        self.position = 0
        self.is_full = False

    def push(self, frame: AudioFrame) -> None:
        """Push a frame into the buffer."""
        # Convert dtype if needed
        data = frame.data.astype(self.dtype) if frame.data.dtype != self.dtype else frame.data

        # Handle frame size vs buffer size
        frame_samples = data.shape[0]

        if frame_samples >= self.frames:
            # Frame is larger than buffer, just take the most recent portion
            self.buffer[:] = data[-self.frames:]
            self.position = 0
            self.is_full = True
        else:
            # Frame fits in buffer
            space_left = self.frames - self.position

            if frame_samples <= space_left:
                # Frame fits in remaining space
                self.buffer[self.position:self.position+frame_samples] = data
                self.position += frame_samples

                # Wrap around if we reached the end
                if self.position >= self.frames:
                    self.position = 0
                    self.is_full = True
            else:
                # Frame needs to wrap around
                first_part = space_left
                second_part = frame_samples - space_left

                # Copy first part to end of buffer
                self.buffer[self.position:] = data[:first_part]

                # Copy second part to beginning of buffer
                self.buffer[:second_part] = data[first_part:]

                self.position = second_part
                self.is_full = True

    def dump(self) -> np.ndarray:
        """Dump the buffer contents in chronological order."""
        if not self.is_full:
            # Buffer not full yet, just return the valid portion
            return self.buffer[:self.position]

        # Buffer is full, need to reorder to get chronological order
        result = np.empty_like(self.buffer)
        result[:self.frames-self.position] = self.buffer[self.position:]
        result[self.frames-self.position:] = self.buffer[:self.position]
        return result

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.fill(0)
        self.position = 0
        self.is_full = False


class AudioRecorder:
    """Records audio to file when squelch is open."""

    def __init__(self, config: Config):
        """Initialize recorder with configuration."""
        self.config = config
        self.recording_config = config.recording
        if not self.recording_config:
            raise ValueError("Recording configuration is required")

        self.data_root = Path(config.app.data_root)
        self.is_recording = False
        self.current_file: Optional[Path] = None
        self.current_writer: Optional[sf.SoundFile] = None
        self.current_metadata: Dict = {}
        self.session_id = self._generate_session_id()

        # Pre-roll buffer
        self.sample_rate = 16000  # Default, will be updated from first frame
        self.channels = 1  # Default, will be updated from first frame
        self.pre_roll = PreRollBuffer(
            ms=self.recording_config.include_pre_roll_ms,
            sample_rate=self.sample_rate,
            channels=self.channels
        )

        # Ensure directories exist
        self._ensure_directories()

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session-{uuid.uuid4().hex[:8]}"

    def _ensure_directories(self) -> None:
        """Ensure necessary directories exist."""
        today = datetime.now().strftime("%Y-%m-%d")
        recordings_dir = self.data_root / "recordings" / today
        os.makedirs(recordings_dir, exist_ok=True)

        indices_dir = self.data_root / "indices"
        os.makedirs(indices_dir, exist_ok=True)

        logs_dir = self.data_root / "logs"
        os.makedirs(logs_dir, exist_ok=True)

    def _get_filename(self, timestamp: float) -> Tuple[Path, str]:
        """Generate a filename for a recording based on the template."""
        dt = datetime.fromtimestamp(timestamp)
        date_str = dt.strftime("%Y-%m-%d")
        timestamp_str = dt.strftime("%Y%m%dT%H%M%S")

        # Find next sequence number
        seq = 1
        while True:
            if self.recording_config is None:
                raise ValueError("Recording config is not set")

            filename = self.recording_config.filename_template.format(
                date=date_str,
                session_id=self.session_id,
                timestamp=timestamp_str,
                seq=f"{seq:04d}",
                ext=self.recording_config.container
            )

            full_path = self.data_root / "recordings" / filename

            # Create parent directory if it doesn't exist
            os.makedirs(full_path.parent, exist_ok=True)

            # Check if file exists
            if not full_path.exists():
                return full_path, f"{timestamp_str}_{seq:04d}"

            seq += 1

    def process_frame(self, frame: AudioFrame) -> None:
        """Process an audio frame."""
        # Update sample rate and channels if needed
        if frame.sample_rate != self.sample_rate or frame.channels != self.channels:
            self.sample_rate = frame.sample_rate
            self.channels = frame.channels
            if self.recording_config is None:
                raise ValueError("Recording config is not set")

            self.pre_roll = PreRollBuffer(
                ms=self.recording_config.include_pre_roll_ms,
                sample_rate=self.sample_rate,
                channels=self.channels,
                dtype=frame.data.dtype
            )

        if not self.is_recording:
            # Add to pre-roll buffer
            self.pre_roll.push(frame)
        else:
            # Write to file
            if self.current_writer:
                self.current_writer.write(frame.data)

    def start_recording(self) -> None:
        """Start recording."""
        if self.is_recording:
            return

        timestamp = time.time()
        self.current_file, recording_id = self._get_filename(timestamp)

        # Prepare metadata
        self.current_metadata = {
            "id": recording_id,
            "date": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d"),
            "timestamp_utc": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "session_id": self.session_id,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "started_at": timestamp,
            "device": self.config.audio.device_name if self.config.audio else "unknown"
        }

        logger.info(f"Starting recording to {self.current_file}")

        try:
            # Open sound file for writing
            if self.current_file is None or self.recording_config is None:
                raise ValueError("Current file or recording config is not set")

            self.current_writer = sf.SoundFile(
                self.current_file,
                mode='w',
                samplerate=self.sample_rate,
                channels=self.channels,
                format=self.recording_config.codec
            )

            # Write pre-roll buffer
            pre_roll_data = self.pre_roll.dump()
            if len(pre_roll_data) > 0:
                self.current_writer.write(pre_roll_data)

            self.is_recording = True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.current_file = None
            self.current_writer = None
            self.current_metadata = {}

    def stop_recording(self) -> Optional[Path]:
        """Stop recording and return the path to the recorded file."""
        if not self.is_recording or not self.current_writer:
            return None

        try:
            # Close the writer
            self.current_writer.close()
            self.current_writer = None

            # Update metadata
            self.current_metadata["ended_at"] = time.time()
            self.current_metadata["duration_sec"] = self.current_metadata["ended_at"] - self.current_metadata["started_at"]

            # Trim silence from the recording
            self._trim_silence()

            # Calculate audio stats
            self._calculate_audio_stats()

            # Write metadata file
            if self.current_file is not None:
                metadata_file = self.current_file.with_suffix(".meta.json")
                with open(metadata_file, 'w') as f:
                    json.dump(self.current_metadata, f, indent=2)

            # Append to index
            self._append_to_index()

            logger.info(f"Finished recording to {self.current_file} (duration: {self.current_metadata['duration_sec']:.2f}s)")

            result_file = self.current_file
            self.current_file = None
            self.is_recording = False

            return result_file
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            self.is_recording = False
            self.current_file = None
            return None

    def _calculate_audio_stats(self) -> None:
        """Calculate audio statistics for the recorded file."""
        if not self.current_file or not self.current_file.exists():
            return

        try:
            # Open the file for reading
            with sf.SoundFile(self.current_file, mode='r') as f:
                # Read in chunks to avoid loading large files entirely into memory
                block_size = 8192
                peak = 0.0
                rms_sum = 0.0
                total_samples = 0

                while True:
                    data = f.read(block_size)
                    if len(data) == 0:
                        break

                    # Convert to float for calculations
                    if data.dtype != np.float32:
                        if data.dtype == np.int16:
                            data = data.astype(np.float32) / 32768.0
                        else:
                            data = data.astype(np.float32)

                    # Calculate peak
                    block_peak = np.max(np.abs(data))
                    peak = max(peak, block_peak)

                    # Calculate RMS
                    rms_sum += np.sum(np.square(data))
                    total_samples += data.size

            # Calculate overall RMS
            if total_samples > 0:
                rms = np.sqrt(rms_sum / total_samples)

                # Convert to dBFS
                peak_dbfs = 20 * np.log10(peak) if peak > 0 else -120.0
                rms_dbfs = 20 * np.log10(rms) if rms > 0 else -120.0

                self.current_metadata["peak_dbfs"] = peak_dbfs
                self.current_metadata["rms_dbfs"] = rms_dbfs
        except Exception as e:
            logger.error(f"Error calculating audio stats: {e}")

    def _append_to_index(self) -> None:
        """Append recording metadata to the daily index file."""
        if not self.current_metadata:
            return

        date = self.current_metadata.get("date")
        if not date:
            return

        index_file = self.data_root / "indices" / f"transmissions.{date}.jsonl"

        # Create a simplified index entry
        index_entry = {
            "id": self.current_metadata["id"],
            "date": self.current_metadata["date"],
            "timestamp_utc": self.current_metadata["timestamp_utc"],
            "session_id": self.current_metadata["session_id"],
            "audio_path": str(self.current_file.relative_to(self.data_root)) if self.current_file is not None else "",
            "metadata_path": str(self.current_file.with_suffix(".meta.json").relative_to(self.data_root)) if self.current_file is not None else "",
            "duration_sec": self.current_metadata.get("duration_sec", 0),
            "rms_dbfs": self.current_metadata.get("rms_dbfs", 0),
            "peak_dbfs": self.current_metadata.get("peak_dbfs", 0),
            "device": self.current_metadata.get("device", "unknown")
        }

        # Append to index file
        with open(index_file, 'a') as f:
            f.write(json.dumps(index_entry) + "\n")

    def _calculate_energy(self, data: np.ndarray) -> np.ndarray:
        """Calculate energy (absolute values) from audio data."""
        if len(data.shape) > 1:  # Multi-channel
            # Use max of all channels
            return np.max(np.abs(data), axis=1)
        else:  # Mono
            return np.abs(data)

    def _detect_ptt_noise(self, energy: np.ndarray, sample_rate: int, window_size: int) -> tuple[bool, int]:
        """Detect PTT noise pattern in the audio.

        Args:
            energy: Energy values of the audio
            sample_rate: Sample rate in Hz
            window_size: Window size in samples

        Returns:
            Tuple of (ptt_detected, ptt_end_idx)
        """
        # Only check the first second for PTT noise
        max_ptt_check_samples = min(int(sample_rate), len(energy) - window_size)

        # Look for a characteristic energy pattern: high energy followed by lower energy
        if len(energy) > window_size * 4:  # Need enough samples to detect pattern
            # Calculate short-term energy in the first second
            short_term_energy = []
            for i in range(0, max_ptt_check_samples, window_size // 2):
                window_energy = np.mean(energy[i:i+window_size])
                short_term_energy.append(window_energy)

            # Look for a peak followed by a dip - characteristic of PTT noise
            if len(short_term_energy) > 4:  # Need enough windows
                for i in range(1, len(short_term_energy) - 3):
                    # Check if this window is a peak (higher than neighbors)
                    if (short_term_energy[i] > short_term_energy[i-1] and
                            short_term_energy[i] > short_term_energy[i+1]):
                        # Check if followed by a significant dip
                        peak_to_dip_ratio = short_term_energy[i] / (short_term_energy[i+2] + 1e-10)

                        # Also check absolute energy level - PTT clicks are usually very loud
                        is_loud_enough = short_term_energy[i] > 10 ** (-25 / 20)  # -25 dB threshold

                        # More sensitive detection: either a strong peak-to-dip ratio OR a very loud peak at the beginning
                        if (peak_to_dip_ratio > 1.5 or  # Reduced from 2.0 for more sensitivity
                            (i < 3 and is_loud_enough)):  # Special case for very beginning
                            ptt_end_idx = (i + 3) * (window_size // 2)  # Skip past the dip
                            logger.info(f"PTT noise pattern detected, skipping first {ptt_end_idx/sample_rate:.2f}s")
                            return True, ptt_end_idx

        return False, 0

    def _find_quiet_section_after_ptt(self, energy: np.ndarray, sample_rate: int, ptt_end_idx: int) -> int:
        """Find a quiet section after PTT noise.

        Args:
            energy: Energy values of the audio
            sample_rate: Sample rate in Hz
            ptt_end_idx: Index where PTT noise ends

        Returns:
            Index where to start the trimmed audio
        """
        # Calculate a dynamic threshold based on the energy levels
        # Get the energy level of the PTT noise (first 50ms)
        ptt_samples = min(int(0.05 * sample_rate), ptt_end_idx)
        if ptt_samples > 0:
            ptt_energy = np.mean(energy[:ptt_samples])
            # Set quiet threshold to be 20dB below the PTT noise
            quiet_threshold = ptt_energy * 0.1  # -20dB relative to PTT noise
        else:
            quiet_threshold = 10 ** (-45 / 20)  # -45 dB absolute threshold as fallback

        # Use a longer window to ensure we find a stable quiet section
        quiet_window_size = int(0.02 * sample_rate)  # 20ms window
        min_quiet_duration = int(0.03 * sample_rate)  # Require at least 30ms of quiet

        # Variables for quiet section detection
        quiet_section_found = False
        quiet_start_idx = ptt_end_idx
        quiet_end_idx = ptt_end_idx
        quiet_duration = 0

        # Look for a quiet section after the PTT noise
        # We'll search up to 500ms after the PTT noise
        search_end = min(ptt_end_idx + int(0.5 * sample_rate), len(energy) - quiet_window_size)

        logger.debug(f"Looking for quiet section from {ptt_end_idx/sample_rate:.3f}s to {search_end/sample_rate:.3f}s")
        logger.debug(f"Quiet threshold: {20*np.log10(quiet_threshold):.1f} dB")

        for i in range(ptt_end_idx, search_end, quiet_window_size // 2):
            window = energy[i:i+quiet_window_size]
            window_energy = np.mean(window)

            # Log every few windows for debugging
            if i % (quiet_window_size * 2) == 0:
                window_db = 20 * np.log10(window_energy) if window_energy > 0 else -120
                logger.debug(f"Window at {i/sample_rate:.3f}s: {window_db:.1f} dB")

            if window_energy < quiet_threshold:
                # Found a quiet window
                if quiet_duration == 0:
                    # Start of quiet section
                    quiet_start_idx = i

                quiet_duration += quiet_window_size // 2

                if quiet_duration >= min_quiet_duration:
                    # Found a sufficiently long quiet section
                    quiet_section_found = True
                    quiet_end_idx = i + quiet_window_size
                    logger.debug(f"Found quiet section from {quiet_start_idx/sample_rate:.3f}s to {quiet_end_idx/sample_rate:.3f}s")
                    break
            else:
                # Reset quiet duration counter if we encounter a loud section
                quiet_duration = 0

        # If we found a quiet section, start from after it
        # Otherwise, use the original PTT end index plus a small buffer
        if quiet_section_found:
            return quiet_end_idx
        else:
            # Fall back to a fixed buffer if no quiet section found
            extra_buffer_ms = 100  # Increased from 50ms
            extra_buffer_samples = int(extra_buffer_ms * sample_rate / 1000)
            start_idx = ptt_end_idx + extra_buffer_samples
            logger.debug(f"No quiet section found, using fixed buffer: {extra_buffer_ms}ms")
            return start_idx

    def _find_speech_after_quiet(self, energy: np.ndarray, sample_rate: int, window_size: int,
                                 start_idx: int, threshold: float, min_active_ratio: float) -> tuple[int, bool]:
        """Find speech after a quiet section.

        Args:
            energy: Energy values of the audio
            sample_rate: Sample rate in Hz
            window_size: Window size in samples
            start_idx: Starting index to search from
            threshold: Energy threshold
            min_active_ratio: Minimum ratio of samples above threshold to consider active

        Returns:
            Tuple of (start_idx, found_signal)
        """
        found_signal = False

        # Look for the actual speech after the quiet section
        for i in range(start_idx, len(energy) - window_size, window_size // 2):
            window = energy[i:i+window_size]
            active_ratio = np.sum(window > threshold) / len(window)
            if active_ratio > min_active_ratio:
                # Found first active window after quiet section
                # Don't look back - we want to start right at the speech
                start_idx = i
                found_signal = True
                logger.debug(f"Found speech signal after quiet section at {i/sample_rate:.2f}s")
                break

        return start_idx, found_signal

    def _find_start_without_ptt(self, energy: np.ndarray, sample_rate: int, window_size: int,
                                threshold: float, start_threshold: float, min_active_ratio: float) -> int:
        """Find start index when no PTT noise is detected.

        Args:
            energy: Energy values of the audio
            sample_rate: Sample rate in Hz
            window_size: Window size in samples
            threshold: Energy threshold
            start_threshold: Higher threshold for beginning detection
            min_active_ratio: Minimum ratio of samples above threshold to consider active

        Returns:
            Start index for trimming
        """
        # Always skip the first 100ms which almost always contains noise
        min_skip_ms = 100
        min_skip_samples = int(min_skip_ms * sample_rate / 1000)

        # Then look for the first real signal after the initial noise period
        skip_initial_ms = 600  # Increased from 500ms
        skip_samples = min(int(skip_initial_ms * sample_rate / 1000), len(energy) // 2)

        # Find start index (first active window after initial noise)
        start_idx = min_skip_samples  # Start at least after the minimum skip
        found_signal = False

        # First check if there's a strong signal after the initial period
        for i in range(skip_samples, len(energy) - window_size, window_size // 2):  # 50% overlap
            window = energy[i:i+window_size]
            active_ratio = np.sum(window > start_threshold) / len(window)
            if active_ratio > min_active_ratio:
                # Found first strong active window after initial noise
                # Use a smaller lookback (10ms) to avoid including any residual noise
                start_idx = max(min_skip_samples, i - int(0.01 * sample_rate))
                found_signal = True
                logger.debug(f"Found strong signal at {i/sample_rate:.2f}s")
                break

        # If no strong signal found, fall back to normal threshold but still enforce minimum skip
        if not found_signal:
            for i in range(min_skip_samples, len(energy) - window_size, window_size // 2):  # 50% overlap
                window = energy[i:i+window_size]
                active_ratio = np.sum(window > threshold) / len(window)
                if active_ratio > min_active_ratio:
                    # Found first active window
                    start_idx = max(min_skip_samples, i - int(0.01 * sample_rate))
                    break

        return start_idx

    def _find_end_index(self, energy: np.ndarray, sample_rate: int, window_size: int,
                        threshold: float, min_active_ratio: float) -> int:
        """Find the end index for trimming.

        Args:
            energy: Energy values of the audio
            sample_rate: Sample rate in Hz
            window_size: Window size in samples
            threshold: Energy threshold
            min_active_ratio: Minimum ratio of samples above threshold to consider active

        Returns:
            End index for trimming
        """
        end_idx = len(energy) - 1

        # Search backwards from the end
        for i in range(len(energy) - window_size, 0, -(window_size // 2)):  # 50% overlap
            window = energy[i:i+window_size]
            active_ratio = np.sum(window > threshold) / len(window)
            if active_ratio > min_active_ratio:
                # Found last active window
                # Go forward a bit to include decay
                end_idx = min(len(energy) - 1, i + window_size + int(0.1 * sample_rate))  # window + 100ms
                break

        return end_idx

    def _apply_fades(self, data: np.ndarray, sample_rate: int, start_idx: int, end_idx: int) -> None:
        """Apply fade in/out to the audio data.

        Args:
            data: Audio data to modify in-place
            sample_rate: Sample rate in Hz
            start_idx: Start index for fade in
            end_idx: End index for fade out
        """
        fade_samples = int(0.01 * sample_rate)  # 10ms fade
        if start_idx + fade_samples < end_idx:
            # Apply fade in
            fade_in = np.linspace(0, 1, fade_samples)
            if len(data.shape) > 1:  # Multi-channel
                for ch in range(data.shape[1]):
                    data[start_idx:start_idx+fade_samples, ch] *= fade_in
            else:  # Mono
                data[start_idx:start_idx+fade_samples] *= fade_in

            # Apply fade out
            fade_out = np.linspace(1, 0, fade_samples)
            if len(data.shape) > 1:  # Multi-channel
                for ch in range(data.shape[1]):
                    data[end_idx-fade_samples:end_idx, ch] *= fade_out
            else:  # Mono
                data[end_idx-fade_samples:end_idx] *= fade_out

    def _log_trimming_info(self, data: np.ndarray, energy: np.ndarray, sample_rate: int,
                           start_idx: int, end_idx: int, threshold_db: float, start_threshold_db: float) -> None:
        """Log detailed information about the trimming process.

        Args:
            data: Audio data
            energy: Energy values of the audio
            sample_rate: Sample rate in Hz
            start_idx: Start index for trimming
            end_idx: End index for trimming
            threshold_db: Main threshold in dB
            start_threshold_db: Start threshold in dB
        """
        logger.info(f"Original recording length: {len(data)/sample_rate:.2f}s ({len(data)} samples)")
        logger.info(f"Trimming from sample {start_idx} to {end_idx} (keeping {end_idx-start_idx+1} samples)")
        logger.info(f"Start threshold: {start_threshold_db} dB, Main threshold: {threshold_db} dB")

        # Calculate energy levels for logging
        if len(data) > 0:
            # Calculate energy levels for different parts of the recording
            start_samples = min(500, len(energy))
            start_energy_db = 20 * np.log10(np.mean(energy[:start_samples])) if np.mean(energy[:start_samples]) > 0 else -120

            # Calculate energy for the first 100ms (likely PTT noise)
            ptt_samples = min(int(0.1 * sample_rate), len(energy))
            ptt_energy_db = 20 * np.log10(np.mean(energy[:ptt_samples])) if np.mean(energy[:ptt_samples]) > 0 else -120

            # Calculate energy for the middle section (likely speech)
            middle_energy_db = 20 * np.log10(np.mean(energy[len(energy)//4:3*len(energy)//4])) if np.mean(energy[len(energy)//4:3*len(energy)//4]) > 0 else -120

            # Calculate energy for the trimmed section
            if start_idx > 0:
                trimmed_start_energy_db = 20 * np.log10(np.mean(energy[:start_idx])) if np.mean(energy[:start_idx]) > 0 else -120
                logger.info(f"Energy levels - First 100ms: {ptt_energy_db:.1f} dB, Trimmed start: {trimmed_start_energy_db:.1f} dB, Middle: {middle_energy_db:.1f} dB")
            else:
                logger.info(f"Energy levels - First 100ms: {ptt_energy_db:.1f} dB, Start: {start_energy_db:.1f} dB, Middle: {middle_energy_db:.1f} dB")

    def _trim_silence(self) -> None:
        """Trim silence from the beginning and end of the recording."""
        if not self.current_file or not self.current_file.exists():
            return

        try:
            # Read the audio file
            data, sample_rate = sf.read(self.current_file, dtype='float32')

            # Parameters for silence detection - much more aggressive trimming
            threshold_db = -35  # Even higher threshold to detect more noise (was -40)
            start_threshold_db = -30  # Special higher threshold just for the beginning
            min_silence_duration_ms = 30  # Even shorter minimum silence to keep (was 50)
            min_silence_samples = int(min_silence_duration_ms * sample_rate / 1000)

            # Convert thresholds to linear scale
            threshold = 10 ** (threshold_db / 20)
            start_threshold = 10 ** (start_threshold_db / 20)

            # Calculate energy (absolute values)
            energy = self._calculate_energy(data)

            # Use sliding window approach for more robust detection
            window_size = int(0.02 * sample_rate)  # 20ms window
            min_active_ratio = 0.2  # Lower ratio for more sensitivity

            # Detect PTT noise pattern
            ptt_detected, ptt_end_idx = self._detect_ptt_noise(energy, sample_rate, window_size)

            # If PTT detected, start from after it
            found_signal = False  # Initialize found_signal variable
            if ptt_detected:
                # Instead of a fixed buffer, look for the transition between PTT noise and speech
                # PTT noise is typically characterized by:
                # 1. A loud initial click/pop
                # 2. Followed by a brief quiet period
                # 3. Then the actual speech

                # Find quiet section after PTT noise
                start_idx = self._find_quiet_section_after_ptt(energy, sample_rate, ptt_end_idx)

                # Find speech after quiet section
                start_idx, found_signal = self._find_speech_after_quiet(energy, sample_rate, window_size, start_idx, threshold, min_active_ratio)
            else:
                # No PTT detected, use standard approach
                start_idx = self._find_start_without_ptt(energy, sample_rate, window_size, threshold, start_threshold, min_active_ratio)

            # Find end index (last active window)
            end_idx = self._find_end_index(energy, sample_rate, window_size, threshold, min_active_ratio)

            # Ensure we have enough audio left
            if end_idx - start_idx < min_silence_samples:
                logger.warning("Recording too short after trimming, keeping original")
                return

            # Apply fades to the audio
            self._apply_fades(data, sample_rate, start_idx, end_idx)

            # Extract the trimmed audio
            trimmed_data = data[start_idx:end_idx+1]

            # Calculate how much was trimmed
            trimmed_start_ms = start_idx * 1000 / sample_rate
            trimmed_end_ms = (len(data) - end_idx - 1) * 1000 / sample_rate

            # Log detailed information about the trimming
            self._log_trimming_info(data, energy, sample_rate, start_idx, end_idx, threshold_db, start_threshold_db)

            # Update metadata with trimming info
            self.current_metadata["trimmed_start_ms"] = trimmed_start_ms
            self.current_metadata["trimmed_end_ms"] = trimmed_end_ms
            self.current_metadata["original_duration_sec"] = self.current_metadata["duration_sec"]
            self.current_metadata["duration_sec"] = len(trimmed_data) / sample_rate

            # Write the trimmed audio back to file
            if self.current_file is not None and self.recording_config is not None:
                sf.write(self.current_file, trimmed_data, sample_rate, format=self.recording_config.codec)

            logger.info(f"Trimmed {trimmed_start_ms:.0f}ms from start and {trimmed_end_ms:.0f}ms from end of recording")
            logger.info(f"New recording length: {len(trimmed_data)/sample_rate:.2f}s")
        except Exception as e:
            logger.error(f"Error trimming silence: {e}")

    def reset(self) -> None:
        """Reset the recorder state."""
        if self.is_recording:
            self.stop_recording()

        self.pre_roll.clear()
        self.current_file = None
        self.current_writer = None
        self.current_metadata = {}
