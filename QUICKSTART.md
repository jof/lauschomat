# Lauschomat Quick Start Guide

This guide will help you quickly get started with Lauschomat for development and testing.

## Development Mode

Development mode runs all components on a single machine, making it easy to test and debug the system.

### Prerequisites

- Python 3.10+
- PulseAudio
- (Optional) NVIDIA GPU with CUDA for transcription

### Step 1: Install Dependencies

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/yourusername/lauschomat.git
cd lauschomat

# Install the package in development mode
pip install -e .

# If you have a GPU and want to use the transcription features
pip install -e ".[transcribe]"

# Install development tools (optional)
pip install -e ".[dev]"
```

### Step 2: List Available Audio Devices

```bash
# List available audio devices
./examples/list_audio_devices.py
```

Take note of the device name you want to use.

### Step 3: Configure the System

```bash
# Create data directories
mkdir -p ./data/recordings
mkdir -p ./data/indices
mkdir -p ./data/logs
mkdir -p ./tmp

# Edit the development configuration
nano config/dev_config.yaml
```

Update the `audio.device_name` setting to match your audio device.

### Step 4: Generate Test Data (Optional)

If you don't have a radio receiver connected, you can generate test data:

```bash
# Generate 10 test transmissions
./examples/generate_test_audio.py --num-files 10
```

### Step 5: Run the Development Mode

```bash
# Run the development mode
./examples/run_dev_mode.sh
```

### Step 6: Access the Web Interface

Open your browser and navigate to:

```
http://localhost:8080
```

## Production Mode

For production deployment, you'll need two systems:

1. **Capture System** (e.g., Raspberry Pi): Connects to the radio receiver and captures audio
2. **Processing System** (e.g., GPU Server): Runs the transcription and web interface

### Capture System Setup

```bash
# On the Raspberry Pi
sudo ./examples/install_production.sh --type capture

# Edit the configuration
sudo nano /etc/lauschomat/capture_config.yaml

# Start the service
sudo systemctl enable --now lauschomat-capture.service
```

### Processing System Setup

```bash
# On the GPU Server
sudo ./examples/install_production.sh --type transcribe
sudo ./examples/install_production.sh --type web

# Edit the configurations
sudo nano /etc/lauschomat/transcribe_config.yaml
sudo nano /etc/lauschomat/web_config.yaml

# Start the services
sudo systemctl enable --now lauschomat-transcribe.service
sudo systemctl enable --now lauschomat-web.service
```

## Troubleshooting

### Audio Device Issues

If you're having trouble with the audio device:

1. Check available devices with `./examples/list_audio_devices.py`
2. Make sure the user has permission to access audio devices
3. For PulseAudio, try using `pactl list sources` to verify the device name

### Transcription Issues

If transcription isn't working:

1. Check if the NeMo toolkit is installed correctly
2. Verify CUDA is available with `python -c "import torch; print(torch.cuda.is_available())"`
3. Check the logs in `./data/logs/transcribe.log`

### Web Interface Issues

If the web interface isn't working:

1. Make sure the web service is running
2. Check if the port is open (default: 8080)
3. Check the logs in `./data/logs/web.log`

## Next Steps

- Read the full [README.md](README.md) for more details
- Check [PLAN.md](PLAN.md) for the system architecture and design
- Explore the code in the `lauschomat` directory
