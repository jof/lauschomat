# Lauschomat: Realtime Radio Transcription System

Lauschomat is a Linux-based application that listens to a radio receiver, detects signal activity (squelch), records transmissions, and transcribes them using NVIDIA Parakeet TDT. It includes a web interface for visualizing the recordings and transcriptions.

## Features

- **Audio Capture**: Listens on a configurable audio device connected to a radio receiver
- **Squelch Detection**: Performs signal edge detection to identify transmissions
- **Recording**: Captures audio recordings of each transmission with configurable pre/post-roll
- **Transcription**: Processes audio through NVIDIA Parakeet TDT for speech-to-text
- **Storage**: Logs audio and transcribed text to configurable filesystem locations
- **Web Interface**: Visualizes recordings with audio playback and transcription display
- **Development Mode**: Combined capture and processing on a single machine for easier development
- **Production Mode**: Distributed architecture with separate capture (Raspberry Pi) and processing (GPU server) components

## Architecture

### Development Mode

In development mode, all components run on a single machine:

```
Radio Receiver -> Audio Device -> Ingest & Squelch -> Recorder -> Transcription -> Storage -> Web App
```

### Production Mode

In production, components are distributed:

- **Raspberry Pi**: Audio capture, squelch detection, and recording
- **GPU Server**: Transcription processing and web interface

## Requirements

- Python 3.10+
- PulseAudio
- NVIDIA GPU with CUDA (for transcription)
- NVIDIA NeMo toolkit with Parakeet TDT model

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lauschomat.git
   cd lauschomat
   ```

2. Install dependencies:
   ```bash
   # Install from requirements.txt
   pip install -r requirements.txt
   
   # Or install package
   pip install -e .
   
   # With transcription support
   pip install -e ".[transcribe]"
   
   # With development tools
   pip install -e ".[dev]"
   ```
   
   Note: For Parakeet TDT support, you need to install NeMo with ASR support:
   ```bash
   pip install nemo_toolkit[asr]
   ```

3. Create configuration files:
   ```bash
   # Copy example configs
   mkdir -p ~/.config/lauschomat
   cp config/*.yaml ~/.config/lauschomat/
   ```

4. Edit configuration files to match your environment:
   ```bash
   # Edit development config
   nano ~/.config/lauschomat/dev_config.yaml
   ```

## Usage

### Development Mode

Run the combined capture and transcription service:

```bash
# List available audio devices
lauschomat-dev --list-devices

# Run with default config
lauschomat-dev

# Run with specific config
lauschomat-dev --config /path/to/dev_config.yaml
```

### Production Mode

On the Raspberry Pi (capture device):

```bash
# List available audio devices
lauschomat-capture --list-devices

# Run capture service
lauschomat-capture --config /path/to/capture_config.yaml
```

On the GPU server:

```bash
# Run transcription service
lauschomat-transcribe --config /path/to/transcribe_config.yaml

# Run web visualization service
lauschomat-web --config /path/to/web_config.yaml
```

## Configuration

### Parakeet TDT Model

The system uses NVIDIA's Parakeet TDT (Text-Dependent Transcription) model via the NeMo toolkit. Configure it in your config file:

```yaml
transcription:
  enabled: true
  engine: nemo_parakeet_tdt
  model_name: nvidia/parakeet-ctc-1.1b  # Or path to a .nemo file
  device: cuda:0                      # Use 'cpu' if no GPU available
  batch_size: 1
  language: en-US
  diarization: false                  # Usually single-speaker for radio
```

Supported model options:
- Use a pretrained model name from NeMo's model catalog
- Specify a path to a downloaded .nemo file

The implementation automatically handles:
- Audio preprocessing (resampling, normalization)
- Word-level timestamp extraction
- Graceful fallback to CPU if GPU is unavailable

### Audio Capture

- **Backend**: PulseAudio (default)
- **Device**: Configurable audio input device
- **Format**: 16-bit PCM (default)
- **Sample Rate**: 16kHz (default, configurable)

### Squelch Detection

- **Method**: Energy-based hysteresis (default)
- **Thresholds**: Configurable open/close thresholds in dBFS
- **Timing**: Configurable hang time and minimum open time

### Recording

- **Format**: WAV (default)
- **Pre/Post-roll**: Configurable buffer before/after transmission
- **Naming**: Configurable filename templates

### Transcription

- **Model**: NVIDIA Parakeet TDT via NeMo
- **Device**: CUDA GPU (configurable, falls back to CPU if GPU unavailable)
- **Output**: JSON with text, confidence, and word timings
- **Word Timestamps**: Automatic extraction of word-level timing information
- **Performance**: Sub-second latency for short transmissions on modern GPUs

### Web Interface

- **Port**: 8080 (default, configurable)
- **Features**: Audio playback, waveform visualization, transcription display

## Directory Structure

```
data_root/
  recordings/
    2025-10-05/
      20251005T205900Z_0001.wav
      20251005T205900Z_0001.meta.json
      20251005T205900Z_0001.transcript.json
  indices/
    transmissions.2025-10-05.jsonl
  logs/
    ingest.log
    transcribe.log
    web.log
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black lauschomat
isort lauschomat
```

## License

[MIT License](LICENSE)
