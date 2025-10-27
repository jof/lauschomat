#!/usr/bin/env python3
"""
Simulate a radio transmission by playing audio through the default output device.
This script plays a simulated radio transmission with PTT sounds, speech, and noise.
"""
import argparse
import sys
import time
import random
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice package not installed.")
    print("Please install it with: pip install sounddevice")
    sys.exit(1)

def generate_noise(duration, sample_rate=16000, amplitude=0.05):
    """Generate white noise."""
    return np.random.normal(0, amplitude, int(duration * sample_rate))

def generate_tone(duration, freq=1000, sample_rate=16000, amplitude=0.5):
    """Generate a sine tone."""
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    tone = amplitude * np.sin(2 * np.pi * freq * t)
    return tone

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

def generate_ptt_sound(sample_rate=16000):
    """Generate a PTT (push-to-talk) sound."""
    duration = random.uniform(0.1, 0.2)
    tone = generate_tone(duration, freq=800, sample_rate=sample_rate, amplitude=0.4)
    return apply_envelope(tone, attack_time=0.01, release_time=0.05, sample_rate=sample_rate)

def generate_speech_simulation(duration, sample_rate=16000):
    """Generate a simulated speech pattern using tones."""
    speech = np.zeros(int(duration * sample_rate))
    
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
    
    return speech

def generate_transmission(duration, sample_rate=16000, with_ptt=True):
    """Generate a simulated radio transmission."""
    # Start with background noise
    background = generate_noise(duration, sample_rate, amplitude=0.02)
    transmission = np.zeros(int(duration * sample_rate))
    
    # Add PTT sound at the beginning
    current_pos = 0
    if with_ptt:
        ptt_sound = generate_ptt_sound(sample_rate)
        transmission[:len(ptt_sound)] += ptt_sound
        current_pos = len(ptt_sound)
        
        # Add a short silence after PTT
        silence_duration = random.uniform(0.1, 0.3)
        current_pos += int(silence_duration * sample_rate)
    
    # Add speech simulation
    speech_duration = duration - (current_pos / sample_rate) - 0.2  # Leave some room at the end
    if speech_duration > 0:
        speech = generate_speech_simulation(speech_duration, sample_rate)
        end_pos = min(current_pos + len(speech), len(transmission))
        transmission[current_pos:end_pos] += speech[:end_pos-current_pos]
    
    # Add background noise
    transmission += background
    
    # Normalize to avoid clipping
    max_amplitude = np.max(np.abs(transmission))
    if max_amplitude > 0.95:
        transmission = transmission * (0.95 / max_amplitude)
    
    return transmission

def play_transmission(duration, sample_rate=16000, with_ptt=True):
    """Generate and play a simulated radio transmission."""
    print(f"Generating {duration}s transmission...")
    transmission = generate_transmission(duration, sample_rate, with_ptt)
    
    print("Playing transmission...")
    sd.play(transmission, sample_rate)
    sd.wait()
    print("Transmission complete")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simulate a radio transmission")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of transmission in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--no-ptt", action="store_true", help="Don't include PTT sound")
    parser.add_argument("--count", type=int, default=1, help="Number of transmissions to generate")
    parser.add_argument("--gap", type=float, default=2.0, help="Gap between transmissions in seconds")
    
    args = parser.parse_args()
    
    try:
        for i in range(args.count):
            if i > 0:
                print(f"Waiting {args.gap}s before next transmission...")
                time.sleep(args.gap)
            
            print(f"Transmission {i+1}/{args.count}")
            play_transmission(
                duration=args.duration,
                sample_rate=args.sample_rate,
                with_ptt=not args.no_ptt
            )
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
