#!/usr/bin/env python3
"""
Generate a test tone to trigger the squelch detector.
This script plays a simple sine wave tone through the default audio output.
"""
import argparse
import sys
import time

try:
    import numpy as np
    import sounddevice as sd
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install them with: pip install numpy sounddevice")
    sys.exit(1)

def generate_tone(frequency=1000, duration=0.5, amplitude=0.5, sample_rate=16000):
    """Generate a sine wave tone."""
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Apply a simple envelope to avoid clicks
    envelope = np.ones_like(tone)
    attack_samples = int(0.01 * sample_rate)
    release_samples = int(0.01 * sample_rate)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    
    return (tone * envelope).astype(np.float32)

def play_tone_sequence(frequency=1000, duration=0.5, gap=0.5, count=3, amplitude=0.5, sample_rate=16000):
    """Play a sequence of tones with gaps."""
    print(f"Playing {count} tones at {frequency} Hz with {gap}s gaps...")
    
    # Create the tone
    tone = generate_tone(frequency, duration, amplitude, sample_rate)
    
    # Play the sequence
    for i in range(count):
        print(f"Playing tone {i+1}/{count}...")
        sd.play(tone, sample_rate)
        sd.wait()
        if i < count - 1:
            print(f"Waiting {gap}s...")
            time.sleep(gap)
    
    print("Done!")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate test tones to trigger squelch")
    parser.add_argument("--frequency", type=float, default=1000, help="Tone frequency in Hz")
    parser.add_argument("--duration", type=float, default=1.0, help="Tone duration in seconds")
    parser.add_argument("--gap", type=float, default=2.0, help="Gap between tones in seconds")
    parser.add_argument("--count", type=int, default=3, help="Number of tones to play")
    parser.add_argument("--amplitude", type=float, default=0.5, help="Tone amplitude (0-1)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz")
    
    args = parser.parse_args()
    
    try:
        play_tone_sequence(
            frequency=args.frequency,
            duration=args.duration,
            gap=args.gap,
            count=args.count,
            amplitude=args.amplitude,
            sample_rate=args.sample_rate
        )
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
