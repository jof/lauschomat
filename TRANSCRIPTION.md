# Lauschomat Transcription

This document provides information about the transcription capabilities in Lauschomat.

## Supported Models

Lauschomat supports multiple speech-to-text models:

1. **Whisper Large V3** (Recommended for noisy environments)
   - High accuracy in noisy conditions
   - Excellent multilingual support
   - Good noise resilience
   - Moderate model size (1.5B parameters)

2. **Granite Speech 3.3** (Alternative for noisy environments)
   - High accuracy in noisy conditions
   - Excellent noise resilience
   - Larger model size (8B parameters)
   - Requires Hugging Face authentication

3. **Parakeet TDT**
   - Fast processing
   - Moderate accuracy
   - Smaller model size (0.6B parameters)

4. **Dummy Model**
   - For testing purposes only
   - No actual transcription

## Installation

To install the required dependencies for transcription:

```bash
# Install base requirements
pip install -r requirements-transcription.txt

# For Granite Speech (recommended for noisy environments)
pip install transformers accelerate safetensors sentencepiece

# For Parakeet TDT
pip install nemo_toolkit[asr]
```

## Configuration

Configure the transcription model in your config file (e.g., `config/dev_config.yaml`):

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
  diarization: false
  # Whisper Large V3 has good performance in noisy environments with proper capitalization and punctuation
  # Granite Speech has good performance in noisy environments with a different style (lowercase, minimal punctuation)
```

## Hardware Requirements

- **Whisper Large V3**: Requires at least 8GB VRAM for GPU acceleration, or 16GB RAM for CPU-only
- **Granite Speech 3.3**: Requires at least 16GB VRAM for GPU acceleration, or 32GB RAM for CPU-only
- **Parakeet TDT**: Requires at least 4GB VRAM for GPU acceleration, or 8GB RAM for CPU-only

## Performance Considerations

- **Whisper Large V3** provides excellent accuracy in noisy environments with good multilingual support and reasonable resource requirements. It properly capitalizes words and adds punctuation.
- **Granite Speech 3.3** provides good accuracy in noisy environments with slightly different transcription style (lowercase, fewer punctuation marks). The 2B parameter version is more memory-efficient than the 8B version.
- **Parakeet TDT** is faster but may have lower accuracy in challenging audio conditions.
- If running on CPU, processing will be significantly slower than on GPU.

## Model Comparison Results

Here's a comparison of Whisper and Granite Speech on the same audio file:

**Whisper Large V3**:
- Text: "Testing audio. Testing silence. Kilo 6 Bravo Golf Papa."
- Processing time: ~5 seconds (on CPU)
- Features: Proper capitalization, punctuation

**Granite Speech 3.3 (2B)**:
- Text: "testing audio testing silence kilo six bravo golf papa"
- Processing time: ~7 seconds (on CPU)
- Features: Lowercase text, minimal punctuation

## Troubleshooting

If you encounter issues with model initialization:

1. Check that you have installed all required dependencies
2. Ensure you have sufficient GPU memory (or switch to CPU by setting `device: cpu`)
3. Check the logs for specific error messages
4. Try the dummy model (`engine: dummy`) to verify the rest of the pipeline is working correctly
