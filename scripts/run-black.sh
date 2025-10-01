#!/bin/bash

# This script run the Black code formatter using uv.

# Exit immediately if a command exits with a non-zero status
set -e

# Run Black code formatter
echo "Running Black code formatter"
uv run black .