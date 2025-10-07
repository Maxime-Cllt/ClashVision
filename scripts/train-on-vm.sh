#!/bin/bash

# This script run the model training process using the specified Python script.

# Exit immediately if a command exits with a non-zero status
set -e

# Update package lists and install necessary dependencies
echo "Updating package lists..."
sudo apt-get update

echo "Installing necessary dependencies..."
sudo apt-get install -y libgl1-mesa-dev -y
sudo apt-get install -y libglib2.0-0 -y

echo "Starting model training..."
uv run python src/clashvision/train/train.py