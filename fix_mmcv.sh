#!/bin/bash
# Fix mmcv installation for Python 3.12 + CUDA 12.1
# Run this if mmcv fails to install via pip

set -e

echo "===== Fixing mmcv installation ====="

# Step 1: Ensure setuptools has pkg_resources
pip install 'setuptools<71' -q

# Step 2: Download precompiled wheel for Python 3.12, CUDA 12.1, torch 2.3
WHEEL_URL="https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.0/mmcv-2.2.0-cp312-cp312-manylinux1_x86_64.whl"
WHEEL_FILE="/tmp/mmcv-2.2.0-cp312.whl"

if [ ! -f "$WHEEL_FILE" ]; then
    echo "Downloading mmcv 2.2.0 wheel..."
    wget -q "$WHEEL_URL" -O "$WHEEL_FILE" || curl -sL "$WHEEL_URL" -o "$WHEEL_FILE"
fi

if [ -f "$WHEEL_FILE" ]; then
    echo "Installing from wheel..."
    pip install --no-deps "$WHEEL_FILE"
    echo "mmcv 2.2.0 installed successfully (2.2.0 is API-compatible with 2.1.0)"
else
    echo "Wheel download failed. Trying pip install with fallback..."
    pip install -q mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html || \
    pip install -q mmcv-lite
fi

echo ""
echo "Verifying installation..."
python -c "import mmcv; print(f'mmcv version: {mmcv.__version__}')" || echo "mmcv import failed"
