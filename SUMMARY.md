# Lauschomat: Project Summary

## Overview

Lauschomat is a realtime radio transcription system that:

1. Listens to a radio receiver via a configurable audio device
2. Detects signal activity using squelch detection
3. Records audio transmissions as WAV files
4. Transcribes the audio using NVIDIA Parakeet TDT
5. Stores recordings and transcriptions in a filesystem
6. Provides a web interface to browse and play recordings

## Architecture

The system is designed with a flexible architecture that supports both:

- **Development Mode**: All components run on a single machine
- **Production Mode**: Distributed architecture with separate capture and processing systems

### Components

1. **Audio Capture**: Interfaces with PulseAudio to record from radio receivers
2. **Squelch Detection**: Uses energy-based hysteresis to detect transmissions
3. **Recording**: Saves transmissions as WAV files with metadata
4. **Transcription**: Processes audio with NVIDIA Parakeet TDT model
5. **Storage**: Organizes files in a structured filesystem
6. **Web Interface**: Visualizes recordings and transcriptions

## Implementation Details

### Audio Capture

- Uses `sounddevice` for audio capture
- Supports configurable audio devices and parameters
- Implements pre-roll buffer to capture the beginning of transmissions

### Squelch Detection

- Energy-based hysteresis with configurable thresholds
- Debouncing to prevent false triggers
- Configurable hang time and minimum open time

### Recording

- WAV file format for simplicity and compatibility
- Separate metadata files in JSON format
- Append-only index files for efficient lookup

### Transcription

- Integration with NVIDIA Parakeet TDT via NeMo
- Placeholder implementation for development without GPU
- Separate files for transcription results

### Web Interface

- FastAPI backend with RESTful API
- Modern frontend with waveform visualization
- Responsive design for desktop and mobile

## Project Structure

```
lauschomat/
├── capture/        # Audio capture and squelch detection
├── common/         # Shared utilities and configuration
├── dev/            # Development mode implementation
├── transcribe/     # Transcription service
└── web/            # Web visualization interface

config/             # Configuration templates
examples/           # Example scripts and utilities
systemd/            # Systemd service files for production
```

## Future Enhancements

1. **Audio Processing**:
   - Additional squelch methods (VAD, ML-based)
   - Audio filtering and noise reduction

2. **Transcription**:
   - Support for multiple languages
   - Custom vocabulary and language models
   - Speaker diarization for multi-speaker transmissions

3. **Web Interface**:
   - Advanced search and filtering
   - Real-time updates
   - User authentication and access control

4. **Monitoring**:
   - Prometheus metrics integration
   - Grafana dashboards
   - Health checks and alerts

5. **Deployment**:
   - Docker containers
   - Kubernetes orchestration
   - Cloud deployment options
