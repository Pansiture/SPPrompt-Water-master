#!/bin/bash
# Install dependencies for SPPrompt-Water (excluding torch which is pre-installed)
# Run: bash install_deps.sh

set -e

echo "===== Installing SPPrompt-Water Dependencies ====="
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(f'  {torch.__version__}, CUDA={torch.cuda.is_available()}')"

echo ""
echo "[1/5] Installing scipy, pandas, scikit-learn, scikit-image..."
pip install -q scipy pandas scikit-learn scikit-image

echo "[2/5] Installing opencv-python, addict, ftfy, tqdm..."
pip install -q opencv-python addict ftfy tqdm

echo "[3/5] Installing transformers..."
pip install -q transformers==4.37.2

echo "[4/5] Installing mmengine..."
pip install -q mmengine==0.10.3

echo "[5/5] Installing monai and mmcv (this may take a while)..."
pip install -q monai==1.3.2

# mmcv often requires special handling
echo "  Installing mmcv (using openmim if available, otherwise pip)..."
pip install -q openmim 2>/dev/null || true
mim install mmcv==2.1.0 2>/dev/null || pip install -q mmcv==2.1.0 || echo "WARNING: mmcv installation failed. You may need to install it manually."

echo ""
echo "===== Installation Complete ====="
echo "Installed packages:"
python -c "
import pkg_resources
packages = ['transformers', 'monai', 'mmcv', 'mmengine', 'scikit-image', 'matplotlib', 'pandas', 'scikit-learn', 'opencv-python', 'addict', 'ftfy', 'tqdm', 'PyYAML', 'pillow', 'scipy', 'numpy']
for p in packages:
    try:
        dist = pkg_resources.get_distribution(p)
        print(f'  {dist.key}=={dist.version}')
    except:
        print(f'  {p}: NOT INSTALLED')
"
