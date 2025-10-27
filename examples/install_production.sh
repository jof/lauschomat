#!/bin/bash
# Installation script for Lauschomat in production mode
# This script should be run as root or with sudo

set -e  # Exit on error

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo"
  exit 1
fi

# Parse arguments
INSTALL_TYPE="all"
CONFIG_DIR="/etc/lauschomat"
DATA_DIR="/var/lib/lauschomat"
USER="lauschomat"
GROUP="lauschomat"

print_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --type TYPE     Installation type: capture, transcribe, web, or all (default: all)"
  echo "  --config-dir DIR Configuration directory (default: /etc/lauschomat)"
  echo "  --data-dir DIR   Data directory (default: /var/lib/lauschomat)"
  echo "  --user USER      User to run services as (default: lauschomat)"
  echo "  --group GROUP    Group to run services as (default: lauschomat)"
  echo "  --help           Show this help message"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --type)
      INSTALL_TYPE="$2"
      shift 2
      ;;
    --config-dir)
      CONFIG_DIR="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --user)
      USER="$2"
      shift 2
      ;;
    --group)
      GROUP="$2"
      shift 2
      ;;
    --help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

# Validate installation type
if [[ ! "$INSTALL_TYPE" =~ ^(capture|transcribe|web|all)$ ]]; then
  echo "Invalid installation type: $INSTALL_TYPE"
  print_usage
  exit 1
fi

echo "Installing Lauschomat ($INSTALL_TYPE) with the following settings:"
echo "  Config directory: $CONFIG_DIR"
echo "  Data directory: $DATA_DIR"
echo "  User: $USER"
echo "  Group: $GROUP"
echo ""

# Create user and group if they don't exist
if ! getent group "$GROUP" > /dev/null; then
  echo "Creating group $GROUP"
  groupadd "$GROUP"
fi

if ! id -u "$USER" > /dev/null 2>&1; then
  echo "Creating user $USER"
  useradd -m -g "$GROUP" -s /bin/bash "$USER"
fi

# Create directories
echo "Creating directories"
mkdir -p "$CONFIG_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/recordings"
mkdir -p "$DATA_DIR/indices"
mkdir -p "$DATA_DIR/logs"

if [[ "$INSTALL_TYPE" == "capture" || "$INSTALL_TYPE" == "all" ]]; then
  mkdir -p "$DATA_DIR/transfer/queue"
  mkdir -p "$DATA_DIR/transfer/sent"
  mkdir -p "/var/tmp/lauschomat"
fi

if [[ "$INSTALL_TYPE" == "transcribe" || "$INSTALL_TYPE" == "all" ]]; then
  mkdir -p "$DATA_DIR/incoming"
  mkdir -p "$DATA_DIR/processed"
fi

if [[ "$INSTALL_TYPE" == "web" || "$INSTALL_TYPE" == "all" ]]; then
  mkdir -p "$DATA_DIR/web/static"
  mkdir -p "$DATA_DIR/web/templates"
fi

# Set permissions
echo "Setting permissions"
chown -R "$USER:$GROUP" "$DATA_DIR"
chmod -R 750 "$DATA_DIR"

if [[ "$INSTALL_TYPE" == "capture" || "$INSTALL_TYPE" == "all" ]]; then
  chown -R "$USER:$GROUP" "/var/tmp/lauschomat"
  chmod -R 750 "/var/tmp/lauschomat"
fi

# Copy configuration files
echo "Copying configuration files"
if [[ "$INSTALL_TYPE" == "capture" || "$INSTALL_TYPE" == "all" ]]; then
  cp -n config/capture_config.yaml "$CONFIG_DIR/" || true
fi

if [[ "$INSTALL_TYPE" == "transcribe" || "$INSTALL_TYPE" == "all" ]]; then
  cp -n config/transcribe_config.yaml "$CONFIG_DIR/" || true
fi

if [[ "$INSTALL_TYPE" == "web" || "$INSTALL_TYPE" == "all" ]]; then
  cp -n config/web_config.yaml "$CONFIG_DIR/" || true
fi

chown -R "$USER:$GROUP" "$CONFIG_DIR"
chmod -R 640 "$CONFIG_DIR"/*.yaml

# Install systemd service files
echo "Installing systemd service files"
if [[ "$INSTALL_TYPE" == "capture" || "$INSTALL_TYPE" == "all" ]]; then
  cp systemd/lauschomat-capture.service /etc/systemd/system/
fi

if [[ "$INSTALL_TYPE" == "transcribe" || "$INSTALL_TYPE" == "all" ]]; then
  cp systemd/lauschomat-transcribe.service /etc/systemd/system/
fi

if [[ "$INSTALL_TYPE" == "web" || "$INSTALL_TYPE" == "all" ]]; then
  cp systemd/lauschomat-web.service /etc/systemd/system/
fi

# Reload systemd
systemctl daemon-reload

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit the configuration files in $CONFIG_DIR to match your environment"
echo "2. Install the Python package: pip install -e ."
echo "3. Start the services:"

if [[ "$INSTALL_TYPE" == "capture" || "$INSTALL_TYPE" == "all" ]]; then
  echo "   systemctl enable --now lauschomat-capture.service"
fi

if [[ "$INSTALL_TYPE" == "transcribe" || "$INSTALL_TYPE" == "all" ]]; then
  echo "   systemctl enable --now lauschomat-transcribe.service"
fi

if [[ "$INSTALL_TYPE" == "web" || "$INSTALL_TYPE" == "all" ]]; then
  echo "   systemctl enable --now lauschomat-web.service"
fi
