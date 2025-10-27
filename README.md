# Lauschomat: Realtime Radio Transcription System

Lauschomat is a Linux-based application that listens to a radio receiver, detects signal activity (squelch), records transmissions, and transcribes them using multiple speech-to-text engines (Whisper, Granite Speech, or Parakeet TDT). It includes a web interface for visualizing the recordings and transcriptions.

## Features

- **Audio Capture**: Listens on a configurable audio device connected to a radio receiver
- **Squelch Detection**: Performs signal edge detection to identify transmissions
- **Recording**: Captures audio recordings of each transmission with configurable pre/post-roll
- **Transcription**: Processes audio through multiple supported speech-to-text engines (Whisper, Granite Speech, Parakeet TDT)
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
- Speech recognition models (one of the following):
  - OpenAI Whisper (recommended for noisy environments)
  - IBM Granite Speech
  - NVIDIA NeMo toolkit with Parakeet TDT model

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jof/lauschomat.git
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

   Note: For transcription support, install the appropriate dependencies based on your chosen engine:
   ```bash
   # For Whisper (recommended for noisy environments)
   pip install transformers accelerate safetensors
   
   # For Granite Speech
   pip install transformers accelerate safetensors sentencepiece
   
   # For Parakeet TDT
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

### Transcription Engines

The system supports multiple speech-to-text engines that can be configured in your config file:

```yaml
transcription:
  enabled: true
  # Available engines: whisper, granite_speech, nemo_parakeet_tdt, dummy
  engine: whisper
  # For whisper, use: openai/whisper-large-v3
  # For granite_speech, use: ibm-granite/granite-speech-3.3-2b (lower memory) or ibm-granite/granite-speech-3.3-8b (higher accuracy)
  # For nemo_parakeet_tdt, use: nvidia/parakeet-ctc-1.1b
  model_name: openai/whisper-large-v3
  device: cuda:0            # Falls back to CPU if GPU unavailable
  batch_size: 1
  language: en-US
  diarization: false        # Usually single-speaker for radio
```

#### Supported Engines

1. **Whisper Large V3** (Recommended for noisy environments)
   - High accuracy in noisy conditions
   - Excellent multilingual support
   - Good noise resilience
   - Moderate model size (1.5B parameters)

2. **Granite Speech 3.3**
   - High accuracy in noisy conditions
   - Excellent noise resilience
   - Available in 2B (memory-efficient) and 8B (higher accuracy) versions
   - Requires Hugging Face authentication

3. **Parakeet TDT**
   - Fast processing
   - Moderate accuracy
   - Smaller model size (0.6B parameters)

All implementations automatically handle:
- Audio preprocessing (resampling, normalization)
- Word-level timestamp extraction (native or estimated)
- Graceful fallback to CPU if GPU is unavailable

#### Hardware Requirements

- **Whisper Large V3**: Requires at least 8GB VRAM for GPU acceleration, or 16GB RAM for CPU-only
- **Granite Speech 3.3**: 
  - 2B version: Requires at least 8GB VRAM for GPU acceleration, or 16GB RAM for CPU-only
  - 8B version: Requires at least 16GB VRAM for GPU acceleration, or 32GB RAM for CPU-only
- **Parakeet TDT**: Requires at least 4GB VRAM for GPU acceleration, or 8GB RAM for CPU-only

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

- **Models**: Multiple supported engines:
  - OpenAI Whisper Large V3 (recommended for noisy environments)
  - IBM Granite Speech 3.3
  - NVIDIA Parakeet TDT via NeMo
- **Device**: CUDA GPU (configurable, falls back to CPU if GPU unavailable)
- **Output**: JSON with text, confidence, and word timings
- **Word Timestamps**: Automatic extraction of word-level timing information
- **Performance**: Processing time varies by model:
  - Whisper: ~5 seconds for short transmissions (CPU), faster on GPU
  - Granite Speech: ~7 seconds for short transmissions (CPU), faster on GPU
  - Parakeet TDT: Sub-second latency for short transmissions on modern GPUs

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

### Removing Trailing Whitespace

To remove trailing whitespace from all files tracked by git in the repository:

```bash
# Run the script from the repository root
python examples/remove_trailing_whitespace.py
```

This script will scan all git-tracked files and remove any trailing whitespace, reporting which files were modified.

## License

[MIT License](LICENSE)
