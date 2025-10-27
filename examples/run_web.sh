#!/bin/bash
# Run Lauschomat web visualization service

# Create data directories if they don't exist
mkdir -p /var/lib/lauschomat/logs
mkdir -p /var/lib/lauschomat/web/static
mkdir -p /var/lib/lauschomat/web/templates

# Check if config file exists, otherwise use default
CONFIG_FILE="/etc/lauschomat/web_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="./config/web_config.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Config file not found"
        exit 1
    fi
fi

# Run the web service
python -m lauschomat.web.main --config "$CONFIG_FILE"
