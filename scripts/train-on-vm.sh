#!/bin/bash

# This script run the model training process using the specified Python script.

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting model training..."
uv run python src/clashvision/train/train.py