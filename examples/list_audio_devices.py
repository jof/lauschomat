#!/usr/bin/env python3
"""
List available audio devices for Lauschomat configuration.
"""
import sys

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice package not installed.")
    print("Please install it with: pip install sounddevice")
    sys.exit(1)

def main():
    """List all available audio devices with details."""
    print("Available audio devices:")
    print("-" * 80)
    print(f"{'ID':<4} {'Name':<40} {'Channels':<10} {'Default':<8} {'API':<15}")
    print("-" * 80)
    
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    default_output = sd.default.device[1]
    
    for i, device in enumerate(devices):
        # Determine direction
        direction = []
        if device.get('max_input_channels', 0) > 0:
            direction.append("IN")
        if device.get('max_output_channels', 0) > 0:
            direction.append("OUT")
        direction_str = "+".join(direction)
        
        # Determine if default
        is_default = []
        if i == default_input:
            is_default.append("IN")
        if i == default_output:
            is_default.append("OUT")
        default_str = "+".join(is_default) if is_default else ""
        
        # Format channels
        if "IN" in direction_str:
            channels = f"in:{device.get('max_input_channels', 0)}"
        else:
            channels = ""
        
        if "OUT" in direction_str:
            if channels:
                channels += ", "
            channels += f"out:{device.get('max_output_channels', 0)}"
        
        # Print device info
        print(f"{i:<4} {device['name'][:39]:<40} {channels:<10} {default_str:<8} {device.get('hostapi', 'unknown')}")
    
    print("-" * 80)
    print("\nFor PulseAudio devices, use the device name in your configuration.")
    print("Example configuration entry:")
    print("""
audio:
  backend: pulse
  device_name: alsa_input.pci-0000_00_1f.3.analog-stereo  # Replace with your device name
  channels: 1
  sample_rate: 16000
  format: S16_LE
  gain_db: 0
""")

if __name__ == "__main__":
    main()
