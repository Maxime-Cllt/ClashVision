#!/bin/bash

# This script run the Ruff linter using uv.

# Exit immediately if a command exits with a non-zero status
set -e

# Run Ruff linter
echo "Running Ruff linter"
uv run ruff check . --fix