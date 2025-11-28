#!/bin/bash
set -e

echo "=== Installing PyTorch CPU-only (smaller size, ~200MB vs ~1.5GB) ==="
pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cpu

echo "=== Installing other dependencies ==="
pip install -r requirements.txt

echo "=== Setup complete ==="
