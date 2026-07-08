#!/usr/bin/env python
"""
Check SPPrompt-Water dependency installation status.
Run: python check_deps.py
"""

import importlib
import sys

def check_pkg(module_name, pkg_name=None):
    pkg_name = pkg_name or module_name
    try:
        mod = importlib.import_module(module_name)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"  [OK] {pkg_name:20s}  {ver}")
        return True
    except ImportError:
        print(f"  [MISSING] {pkg_name:20s}  NOT INSTALLED")
        return False

print("=" * 50)
print("SPPrompt-Water Dependency Check")
print("=" * 50)

print("\nCore Deep Learning:")
check_pkg('torch')
check_pkg('torchvision')
check_pkg('transformers')

print("\nSegmentation & Vision:")
check_pkg('monai')
check_pkg('mmcv')
check_pkg('mmengine')
check_pkg('cv2', 'opencv-python')

print("\nImage & Data Processing:")
check_pkg('numpy')
check_pkg('scipy')
check_pkg('skimage', 'scikit-image')
check_pkg('pandas')
check_pkg('sklearn', 'scikit-learn')
check_pkg('PIL', 'pillow')

print("\nUtilities:")
check_pkg('matplotlib')
check_pkg('tqdm')
check_pkg('yaml', 'PyYAML')
check_pkg('addict')
check_pkg('ftfy')

print("\n" + "=" * 50)
print("If mmcv is missing, run: bash fix_mmcv.sh")
print("=" * 50)
