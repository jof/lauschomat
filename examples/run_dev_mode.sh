#!/bin/bash
# Run Lauschomat in development mode

# Create data directories if they don't exist
mkdir -p ./data/recordings
mkdir -p ./data/indices
mkdir -p ./data/logs
mkdir -p ./tmp

# Check if config file exists, otherwise use default
CONFIG_FILE="./config/dev_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file $CONFIG_FILE not found, using default config"
    CONFIG_FILE=""
fi

# Run the development mode
python -m lauschomat.dev.main --config "$CONFIG_FILE"
