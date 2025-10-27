#!/usr/bin/env python3
"""
Generate test audio files for Lauschomat development and testing.

This script creates WAV files with simulated radio transmissions
including noise, speech, and silence periods.
"""
import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


def generate_noise(duration, sample_rate=16000, amplitude=0.1):
    """Generate white noise."""
    return np.random.normal(0, amplitude, int(duration * sample_rate))


def generate_tone(duration, freq=1000, sample_rate=16000, amplitude=0.5):
    """Generate a sine tone."""
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    tone = amplitude * np.sin(2 * np.pi * freq * t)
    return tone


def generate_silence(duration, sample_rate=16000):
    """Generate silence."""
    return np.zeros(int(duration * sample_rate))


def apply_envelope(signal, attack_time=0.01, release_time=0.01, sample_rate=16000):
    """Apply an attack/release envelope to the signal."""
    attack_samples = int(attack_time * sample_rate)
    release_samples = int(release_time * sample_rate)

    if len(signal) <= attack_samples + release_samples:
        # Signal is too short, use linear envelope
        envelope = np.concatenate([
            np.linspace(0, 1, min(attack_samples, len(signal) // 2)),
            np.linspace(1, 0, len(signal) - min(attack_samples, len(signal) // 2))
        ])
        return signal * envelope

    # Create envelope
    attack = np.linspace(0, 1, attack_samples)
    sustain = np.ones(len(signal) - attack_samples - release_samples)
    release = np.linspace(1, 0, release_samples)

    envelope = np.concatenate([attack, sustain, release])
    return signal * envelope


def generate_transmission(duration, sample_rate=16000):
    """Generate a simulated radio transmission."""
    # Start with background noise
    background = generate_noise(duration, sample_rate, amplitude=0.02)

    # Add a tone at the beginning (PTT sound)
    ptt_duration = random.uniform(0.1, 0.2)
    ptt_tone = generate_tone(ptt_duration, freq=800, sample_rate=sample_rate, amplitude=0.4)
    ptt_tone = apply_envelope(ptt_tone, attack_time=0.01, release_time=0.05, sample_rate=sample_rate)

    # Add some silence after PTT
    silence_duration = random.uniform(0.1, 0.3)
    silence = generate_silence(silence_duration)

    # Add some "speech" (just tones of different frequencies for now)
    speech_duration = duration - ptt_duration - silence_duration - 0.2  # Leave some room at the end
    if speech_duration > 0:
        speech = np.zeros(int(speech_duration * sample_rate))

        # Add several tone bursts to simulate speech
        position = 0
        while position < len(speech):
            tone_duration = random.uniform(0.1, 0.5)
            tone_samples = int(tone_duration * sample_rate)
            if position + tone_samples > len(speech):
                tone_samples = len(speech) - position

            freq = random.uniform(300, 3000)
            amplitude = random.uniform(0.2, 0.5)

            tone = generate_tone(tone_duration, freq=freq, sample_rate=sample_rate, amplitude=amplitude)
            tone = tone[:tone_samples]  # Ensure we don't exceed the allocated space
            tone = apply_envelope(tone, attack_time=0.01, release_time=0.01, sample_rate=sample_rate)

            speech[position:position+len(tone)] += tone
            position += tone_samples + int(random.uniform(0.05, 0.2) * sample_rate)  # Add some silence between tones
    else:
        speech = np.array([])

    # Combine all parts
    transmission = np.zeros(int(duration * sample_rate))

    # Add PTT tone at the beginning
    transmission[:len(ptt_tone)] += ptt_tone

    # Add speech after PTT and silence
    start_idx = len(ptt_tone) + len(silence)
    if len(speech) > 0:
        end_idx = min(start_idx + len(speech), len(transmission))
        transmission[start_idx:end_idx] += speech[:end_idx-start_idx]

    # Add background noise
    transmission += background

    # Normalize to avoid clipping
    max_amplitude = np.max(np.abs(transmission))
    if max_amplitude > 0.95:
        transmission = transmission * (0.95 / max_amplitude)

    return transmission


def generate_test_dataset(output_dir, num_files=10, min_duration=2.0, max_duration=10.0, sample_rate=16000):
    """Generate a test dataset of simulated radio transmissions."""
    os.makedirs(output_dir, exist_ok=True)
    indices_dir = os.path.join(output_dir, "indices")
    recordings_dir = os.path.join(output_dir, "recordings")

    today = datetime.now().strftime("%Y-%m-%d")
    recordings_date_dir = os.path.join(recordings_dir, today)

    os.makedirs(indices_dir, exist_ok=True)
    os.makedirs(recordings_date_dir, exist_ok=True)

    index_file = os.path.join(indices_dir, f"transmissions.{today}.jsonl")
    session_id = f"test-session-{int(time.time())}"

    for i in range(num_files):
        # Generate random duration
        duration = random.uniform(min_duration, max_duration)

        # Generate transmission
        audio = generate_transmission(duration, sample_rate)

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        seq = f"{i+1:04d}"
        filename = f"{timestamp}_{seq}"

        # Create paths
        audio_path = os.path.join(recordings_date_dir, f"{filename}.wav")
        meta_path = os.path.join(recordings_date_dir, f"{filename}.meta.json")
        transcript_path = os.path.join(recordings_date_dir, f"{filename}.transcript.json")

        # Save audio file
        sf.write(audio_path, audio, sample_rate)

        # Calculate audio stats
        rms = np.sqrt(np.mean(np.square(audio)))
        peak = np.max(np.abs(audio))

        # Convert to dBFS
        rms_dbfs = 20 * np.log10(rms) if rms > 0 else -120.0
        peak_dbfs = 20 * np.log10(peak) if peak > 0 else -120.0

        # Create metadata
        metadata = {
            "id": filename,
            "date": today,
            "timestamp_utc": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "session_id": session_id,
            "sample_rate": sample_rate,
            "channels": 1,
            "started_at": time.time(),
            "ended_at": time.time() + duration,
            "duration_sec": duration,
            "rms_dbfs": rms_dbfs,
            "peak_dbfs": peak_dbfs,
            "device": "test-device"
        }

        # Create transcription
        transcription = {
            "model": "test_model",
            "version": "0.1.0",
            "text": f"This is test transmission {i+1} with duration {duration:.1f} seconds.",
            "confidence": random.uniform(0.7, 0.98),
            "words": [
                {"w": "This", "start": 0.3, "end": 0.5, "conf": 0.95},
                {"w": "is", "start": 0.6, "end": 0.7, "conf": 0.97},
                {"w": "test", "start": 0.8, "end": 1.0, "conf": 0.96},
                {"w": "transmission", "start": 1.1, "end": 1.8, "conf": 0.94},
                {"w": str(i+1), "start": 1.9, "end": 2.1, "conf": 0.99},
            ],
            "latency_ms": int(duration * 100),
            "runtime_device": "cpu"
        }

        # Save metadata and transcription
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        with open(transcript_path, 'w') as f:
            json.dump(transcription, f, indent=2)

        # Create index entry
        index_entry = {
            "id": filename,
            "date": today,
            "timestamp_utc": metadata["timestamp_utc"],
            "session_id": session_id,
            "audio_path": os.path.relpath(audio_path, output_dir),
            "metadata_path": os.path.relpath(meta_path, output_dir),
            "transcription_path": os.path.relpath(transcript_path, output_dir),
            "duration_sec": duration,
            "rms_dbfs": rms_dbfs,
            "peak_dbfs": peak_dbfs,
            "device": "test-device",
            "text": transcription["text"],
            "confidence": transcription["confidence"],
            "model": transcription["model"]
        }

        # Append to index file
        with open(index_file, 'a') as f:
            f.write(json.dumps(index_entry) + "\n")

        print(f"Generated transmission {i+1}/{num_files}: {audio_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate test audio for Lauschomat")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--num-files", type=int, default=10, help="Number of files to generate")
    parser.add_argument("--min-duration", type=float, default=2.0, help="Minimum transmission duration in seconds")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Maximum transmission duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz")

    args = parser.parse_args()

    print(f"Generating {args.num_files} test transmissions in {args.output_dir}")
    generate_test_dataset(
        args.output_dir,
        num_files=args.num_files,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        sample_rate=args.sample_rate
    )
    print("Done!")


if __name__ == "__main__":
    main()
