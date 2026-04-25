#!/usr/bin/env bash
set -e

# Automatically run our python script and handle sys path variables natively via uv if needed.
# But we can just use normal python assuming the env is active.
cd "$(dirname "$0")/.."
echo "Starting multi-modal visualizations..."

PYTHONPATH="$(pwd)/1_training" uv run python 4_Visulizations/visualize_src/run_visualizations.py
