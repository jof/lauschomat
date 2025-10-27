#!/bin/bash
# Clear all recordings and indices

echo "This script will delete all recordings and indices."
echo "Press Ctrl+C to cancel or Enter to continue..."
read

# Define data directories
RECORDINGS_DIR="./data/recordings"
INDICES_DIR="./data/indices"

# Check if directories exist
if [ ! -d "$RECORDINGS_DIR" ] && [ ! -d "$INDICES_DIR" ]; then
    echo "No recordings or indices directories found."
    exit 0
fi

# Clear recordings
if [ -d "$RECORDINGS_DIR" ]; then
    echo "Clearing recordings directory..."
    rm -rf "$RECORDINGS_DIR"/*
    echo "Recordings cleared."
fi

# Clear indices
if [ -d "$INDICES_DIR" ]; then
    echo "Clearing indices directory..."
    rm -f "$INDICES_DIR"/*.jsonl
    echo "Indices cleared."
fi

echo "All recordings and indices have been cleared."
echo "Please restart the application for changes to take effect."
