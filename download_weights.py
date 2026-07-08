#!/usr/bin/env python
"""
Download pretrain weights for SPPrompt-Water.

Requires:
- Swin Transformer Tiny pretrain weights
- SAM ViT-B checkpoint (optional, for prompt module initialization)

Usage:
    python download_weights.py --output_dir ./pretrain

Swin weights download URL:
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth

SAM weights download URL (optional):
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
"""

import os
import argparse
import urllib.request
from pathlib import Path


def download_file(url, output_path, desc=""):
    """Download a file with progress."""
    print(f"Downloading {desc}...")
    print(f"  URL: {url}")
    print(f"  Save to: {output_path}")
    
    def report_hook(count, block_size, total_size):
        percent = min(int(count * block_size * 100 / total_size), 100)
        print(f"\r  Progress: {percent}%", end="")
    
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=report_hook)
        print("\n  Done.")
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download pretrain weights")
    parser.add_argument("--output_dir", type=str, default="./pretrain",
                        help="Directory to save weights")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Swin Transformer Tiny weights
    swin_url = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
    swin_path = os.path.join(args.output_dir, "swin_tiny_patch4_window7_224.pth")
    
    if not os.path.exists(swin_path):
        download_file(swin_url, swin_path, "Swin Transformer Tiny")
    else:
        print(f"Swin Transformer weights already exist: {swin_path}")
    
    # SAM ViT-B weights (optional, for prompt module)
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    sam_path = os.path.join(args.output_dir, "sam_vit_b_01ec64.pth")
    
    print("\nNOTE: SAM weights are optional if you do not use the prompt module with pretrain init.")
    print("      The model can still initialize without SAM weights, but prompt module will start from scratch.")
    
    if not os.path.exists(sam_path):
        ans = input("\nDownload SAM ViT-B weights? (y/n): ")
        if ans.lower() == 'y':
            download_file(sam_url, sam_path, "SAM ViT-B")
    else:
        print(f"SAM weights already exist: {sam_path}")
    
    print("\n===== Download Summary =====")
    print(f"Output directory: {args.output_dir}")
    if os.path.exists(swin_path):
        print(f"  [OK] Swin Tiny: {swin_path}")
    if os.path.exists(sam_path):
        print(f"  [OK] SAM ViT-B: {sam_path}")
    
    print("\nNext steps:")
    print("  1. Rename Swin weight if needed: swin_tiny_patch4_window7_224.pth -> swin_tiny_patch4_window7_224_20220317-1cdeb081.pth")
    print("  2. Or update train_promptwaternet.py --SwintransformerPretrain to point to the actual filename")


if __name__ == "__main__":
    main()
