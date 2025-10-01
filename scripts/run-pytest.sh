#!/bin/bash

# This script runs pytest using uv.

# Exit immediately if a command exits with a non-zero status
set -e

# Run pytest
echo "Running pytest"
uv run pytest