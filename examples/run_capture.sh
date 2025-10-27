#!/bin/bash
# Run Lauschomat capture service (for Raspberry Pi)

# Create data directories if they don't exist
mkdir -p /var/lib/lauschomat/recordings
mkdir -p /var/lib/lauschomat/indices
mkdir -p /var/lib/lauschomat/logs
mkdir -p /var/lib/lauschomat/transfer/queue
mkdir -p /var/lib/lauschomat/transfer/sent
mkdir -p /var/tmp/lauschomat

# Check if config file exists, otherwise use default
CONFIG_FILE="/etc/lauschomat/capture_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="./config/capture_config.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Config file not found"
        exit 1
    fi
fi

# Run the capture service
python -m lauschomat.capture.main --config "$CONFIG_FILE"
