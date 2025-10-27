#!/bin/bash
# Run Lauschomat transcription service (for GPU server)

# Create data directories if they don't exist
mkdir -p /var/lib/lauschomat/incoming
mkdir -p /var/lib/lauschomat/processed
mkdir -p /var/lib/lauschomat/indices
mkdir -p /var/lib/lauschomat/logs

# Check if config file exists, otherwise use default
CONFIG_FILE="/etc/lauschomat/transcribe_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="./config/transcribe_config.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Config file not found"
        exit 1
    fi
fi

# Run the transcription service
python -m lauschomat.transcribe.main --config "$CONFIG_FILE"
