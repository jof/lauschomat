#!/bin/bash
# Run Lauschomat in development mode with verbose logging

# Set up logging level
export LOGLEVEL=DEBUG

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

# Clear previous log file
LOG_FILE="./data/logs/dev.log"
> "$LOG_FILE"

echo "Starting Lauschomat in verbose mode..."
echo "Log file: $LOG_FILE"
echo "Audio levels will be displayed every 2 seconds"
echo "Press Ctrl+C to stop"
echo ""
echo "To generate test tones in another terminal, run:"
echo "  ./examples/generate_test_tone.py"
echo ""

# Run the development mode with verbose logging
python -m lauschomat.dev.main --config "$CONFIG_FILE" --log-level DEBUG 2>&1 | tee -a "$LOG_FILE"
