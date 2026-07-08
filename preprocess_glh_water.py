#!/usr/bin/env python
"""
GLH-Water Dataset Preprocessing Script for SPPrompt-Water

This script prepares the GLH-Water dataset according to the paper:
"Breaking the Scale Barrier: A Scale-Progressive Water Body Extraction
Framework Inspired by Human Cognitive Mechanisms"

Steps:
1. Unzip dataset (if needed)
2. Split into train/val/test (60/20/20)
3. Generate multi-scale image pyramids (level0, level1, level2)
4. Crop into 1024x1024 patches
5. Filter patches without water bodies
6. Generate prompt masks via:
   - Connected-component decomposition (TopologicalDecomposition)
   - RandomMerge of subregions
   - Morphological perturbation (Erosion/Dilation, kernel size 3-10)
   - 4x downsampling to 256x256

Usage:
    python preprocess_glh_water.py --input_dir /path/to/GLH-water --output_dir /path/to/data

Output structure:
    data/
    ├── level0/
    │   ├── train/imgs, gts, prompt_mask_256
    │   ├── val/imgs, gts, prompt_mask_256
    │   └── test/imgs, gts, prompt_mask_256
    ├── level1/
    └── level2/
"""

import os
import argparse
import shutil
import random
import zipfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
from skimage import io, measure, morphology
from scipy import ndimage
from PIL import Image

# 解除PIL大图像安全限制，GLH-Water是超高分辨率影像
Image.MAX_IMAGE_PIXELS = None

join = os.path.join


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess GLH-Water dataset for SPPrompt-Water")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to raw GLH-Water dataset (contains images/ and labels/ or .zip)")
    parser.add_argument("--output_dir", type=str, default="/autodl-pub/glh_data",
                        help="Output directory for processed data (use autodl-pub for enough disk space)")
    parser.add_argument("--patch_size", type=int, default=1024,
                        help="Size of cropped patches (default: 1024)")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4],
                        help="Downsampling factors for each level (default: 1 2 4 means level0=1x, level1=2x, level2=4x)")
    parser.add_argument("--overlap", type=int, default=0,
                        help="Overlap between patches in pixels (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_prompts", type=int, default=3,
                        help="Number of prompt variants to generate per patch (default: 3)")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers for preprocessing (default: 16 for 25 vCPU)")
    return parser.parse_args()


def find_files(directory, extensions=(".tif", ".tiff", ".png", ".jpg", ".jpeg")):
    """Recursively find files with given extensions."""
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*{ext}"))
    # Also check uppercase extensions
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*{ext.upper()}"))
    return sorted(list(set(files)))


def unzip_dataset(zip_path, output_dir):
    """Unzip dataset if input is a zip file."""
    extract_dir = join(output_dir, "raw_glh_water")
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Unzipping {zip_path} to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Unzip complete.")
    return extract_dir


def discover_dataset_structure(root_dir):
    """
    Auto-discover images and labels directories.
    Common patterns: images/labels, img/gt, image/label, etc.
    """
    root = Path(root_dir)
    
    # Possible directory names
    img_candidates = ["images", "image", "imgs", "img", "IMG", "Images", "IMAGES"]
    lbl_candidates = ["labels", "label", "gts", "gt", "GT", "Labels", "LABELS", "masks", "mask"]
    
    img_dir = None
    lbl_dir = None
    
    for cand in img_candidates:
        if (root / cand).is_dir():
            img_dir = str(root / cand)
            break
    
    for cand in lbl_candidates:
        if (root / cand).is_dir():
            lbl_dir = str(root / cand)
            break
    
    # If not found at top level, search one level deeper
    if img_dir is None or lbl_dir is None:
        for subdir in root.iterdir():
            if subdir.is_dir():
                for cand in img_candidates:
                    if (subdir / cand).is_dir():
                        img_dir = str(subdir / cand)
                        break
                for cand in lbl_candidates:
                    if (subdir / cand).is_dir():
                        lbl_dir = str(subdir / cand)
                        break
    
    if img_dir is None:
        # Fallback: just find all images and assume labels have same basename
        all_files = find_files(root_dir)
        # Heuristic: if there are files with _label or _mask in name
        raise ValueError(f"Could not auto-discover image/label directories in {root_dir}. "
                         "Please organize as: input_dir/images/ and input_dir/labels/")
    
    print(f"Discovered image dir: {img_dir}")
    print(f"Discovered label dir: {lbl_dir}")
    return img_dir, lbl_dir


def discover_splits(root_dir, img_dir, lbl_dir):
    """
    Check if root_dir already has train/val/test splits.
    Each split should contain img/ and label/ (or similar) subdirectories.
    Returns {split_name: [(img_path, lbl_path), ...]} or None.
    """
    root = Path(root_dir)
    splits = {}
    split_names = ["train", "val", "test", "validation"]
    
    for split_name in split_names:
        split_dir = root / split_name
        if not split_dir.is_dir():
            continue
        
        # Find img and label subdirs within this split
        img_candidates = ["img", "imgs", "image", "images", "IMG", "Images"]
        lbl_candidates = ["label", "labels", "gt", "gts", "GT", "mask", "masks", "Mask"]
        
        split_img_dir = None
        split_lbl_dir = None
        
        for cand in img_candidates:
            if (split_dir / cand).is_dir():
                split_img_dir = str(split_dir / cand)
                break
        
        for cand in lbl_candidates:
            if (split_dir / cand).is_dir():
                split_lbl_dir = str(split_dir / cand)
                break
        
        if split_img_dir and split_lbl_dir:
            pairs = match_image_label_pairs(split_img_dir, split_lbl_dir)
            if len(pairs) > 0:
                splits[split_name] = pairs
                print(f"  {split_name}: found {len(pairs)} pairs")
    
    if len(splits) >= 2:
        return splits
    return None


def match_image_label_pairs(img_dir, lbl_dir):
    """Match image files with label files by basename."""
    img_files = find_files(img_dir)
    lbl_files = find_files(lbl_dir)
    
    # Create basename map for labels
    lbl_map = {}
    for lf in lbl_files:
        stem = lf.stem
        # Remove common suffixes like _label, _mask, _gt
        for suffix in ["_label", "_mask", "_gt", "_L", "_M"]:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        lbl_map[stem] = str(lf)
    
    pairs = []
    for img in img_files:
        stem = img.stem
        # Try exact match first
        if stem in lbl_map:
            pairs.append((str(img), lbl_map[stem]))
        else:
            # Try without extension-based suffixes
            found = False
            for key in lbl_map:
                if stem == key or key == stem:
                    pairs.append((str(img), lbl_map[key]))
                    found = True
                    break
            if not found:
                print(f"Warning: no label found for image {img}")
    
    print(f"Matched {len(pairs)} image-label pairs.")
    return pairs


def split_dataset(pairs, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Split pairs into train/val/test."""
    random.seed(seed)
    np.random.seed(seed)
    
    indices = np.random.permutation(len(pairs))
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    test_pairs = [pairs[i] for i in test_idx]
    
    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
    return {"train": train_pairs, "val": val_pairs, "test": test_pairs}


def read_image(path):
    """Read image as numpy array (H, W, C) or (H, W)."""
    img = io.imread(path)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    return img


def resize_image(img, scale_factor, order=1):
    """Resize image by scale_factor using scipy.ndimage.zoom."""
    if scale_factor == 1:
        return img
    if img.ndim == 3:
        zoom = (scale_factor, scale_factor, 1)
    else:
        zoom = (scale_factor, scale_factor)
    return ndimage.zoom(img, zoom, order=order)


def crop_patches(img, lbl, patch_size=1024, overlap=0):
    """
    Crop image and label into patches.
    Returns list of (img_patch, lbl_patch, row, col).
    """
    h, w = img.shape[:2]
    stride = patch_size - overlap
    patches = []
    
    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            img_patch = img[r:r + patch_size, c:c + patch_size]
            lbl_patch = lbl[r:r + patch_size, c:c + patch_size]
            patches.append((img_patch, lbl_patch, r, c))
    
    # Handle edge cases: if image is smaller than patch_size, pad
    if h < patch_size or w < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        lbl_pad = np.pad(lbl, ((0, pad_h), (0, pad_w)), mode='constant')
        patches.append((img_pad[:patch_size, :patch_size], lbl_pad[:patch_size, :patch_size], 0, 0))
    
    return patches


def topological_decomposition(lbl_patch):
    """
    Perform connected-component decomposition on binary label.
    Returns list of binary masks, one per connected component.
    """
    lbl_patch = (lbl_patch > 0).astype(np.uint8)
    labeled, num_features = ndimage.label(lbl_patch)
    
    subregions = []
    for i in range(1, num_features + 1):
        component = (labeled == i).astype(np.uint8)
        subregions.append(component)
    
    return subregions


def random_merge(subregions, min_keep=1):
    """
    Randomly sample a subset of subregions and merge them via pixelwise union.
    Returns a binary mask.
    """
    if len(subregions) == 0:
        return np.zeros_like(subregions[0]) if subregions else None
    
    if len(subregions) == 1:
        return subregions[0]
    
    # Randomly decide how many to keep (at least min_keep, at most all)
    n_keep = random.randint(min_keep, len(subregions))
    selected = random.sample(subregions, n_keep)
    
    merged = np.zeros_like(selected[0])
    for s in selected:
        merged = np.logical_or(merged, s).astype(np.uint8)
    
    return merged


def morphological_perturbation(mask, kernel_range=(3, 10)):
    """
    Apply random erosion or dilation.
    Kernel size randomly chosen from kernel_range, iterations=1.
    """
    kernel_size = random.randint(kernel_range[0], kernel_range[1])
    
    # Create disk-shaped structuring element (approximated by square for simplicity)
    # For better results, use skimage.morphology.disk
    try:
        selem = morphology.disk(kernel_size // 2)
    except Exception:
        selem = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    op = random.choice(["erosion", "dilation"])
    
    if op == "erosion":
        result = morphology.erosion(mask, selem)
    else:
        result = morphology.dilation(mask, selem)
    
    return result


def generate_prompt_mask(label_patch, num_prompts=3):
    """
    Generate prompt masks according to the paper's Algorithm 1.
    
    Steps:
    1. Topological decomposition (connected components)
    2. RandomMerge of subregions
    3. Morphological perturbation (erosion/dilation, kernel 3-10)
    4. Downsample 4x to 256x256
    
    Returns list of (prompt_256, guide_label_1024) tuples.
    """
    subregions = topological_decomposition(label_patch)
    
    if len(subregions) == 0:
        # No water in this patch
        return []
    
    prompts = []
    for _ in range(num_prompts):
        # Step 1: Random merge
        merged = random_merge(subregions)
        
        # Step 2: Morphological perturbation
        perturbed = morphological_perturbation(merged)
        
        # Step 3: 4x downsampling to 256x256
        prompt_256 = ndimage.zoom(perturbed, 0.25, order=0)
        
        # Ensure binary and uint8
        prompt_256 = (prompt_256 > 0).astype(np.uint8) * 255
        
        # Guide label is the original merged region (before perturbation), full resolution
        guide_label = (merged > 0).astype(np.uint8) * 255
        
        prompts.append((prompt_256, guide_label))
    
    return prompts


def save_patch(img_patch, lbl_patch, prompt_256, guide_label, 
               out_img_dir, out_gt_dir, out_prompt_dir, out_guide_dir,
               basename, idx, prompt_idx):
    """Save a single patch and its prompt."""
    # Image: ensure 3 channels, uint8
    if img_patch.ndim == 2:
        img_patch = np.stack([img_patch] * 3, axis=-1)
    elif img_patch.shape[2] == 1:
        img_patch = np.repeat(img_patch, 3, axis=-1)
    elif img_patch.shape[2] > 3:
        img_patch = img_patch[:, :, :3]
    
    # Convert to uint8 if needed
    if img_patch.dtype != np.uint8:
        if img_patch.max() <= 1.0:
            img_patch = (img_patch * 255).astype(np.uint8)
        else:
            img_patch = img_patch.astype(np.uint8)
    
    # Label: uint8, 0 or 255
    lbl_patch = (lbl_patch > 0).astype(np.uint8) * 255
    
    filename = f"{basename}_p{idx}_v{prompt_idx}.png"
    
    io.imsave(join(out_img_dir, filename), img_patch, check_contrast=False)
    io.imsave(join(out_gt_dir, filename), lbl_patch.astype(np.uint8), check_contrast=False)
    io.imsave(join(out_prompt_dir, filename), prompt_256.astype(np.uint8), check_contrast=False)
    
    # Optional: save guide label (full-res prompt mask for reference)
    if out_guide_dir is not None:
        io.imsave(join(out_guide_dir, filename), guide_label.astype(np.uint8), check_contrast=False)


import multiprocessing as mp
from functools import partial

def process_single_image(args_tuple):
    """Process a single image-label pair for one scale level."""
    img_path, lbl_path, scale_factor, patch_size, overlap, num_prompts, out_dirs = args_tuple
    out_img_dir, out_gt_dir, out_prompt_dir, out_guide_dir = out_dirs
    
    total_patches = 0
    kept_patches = 0
    
    try:
        img = read_image(img_path)
        lbl = read_image(lbl_path)
        
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        
        img_scaled = resize_image(img, 1.0 / scale_factor, order=1)
        lbl_scaled = resize_image(lbl, 1.0 / scale_factor, order=0)
        
        patches = crop_patches(img_scaled, lbl_scaled, patch_size, overlap)
        basename = Path(img_path).stem
        
        for idx, (img_patch, lbl_patch, r, c) in enumerate(patches):
            total_patches += 1
            
            if np.sum(lbl_patch > 0) == 0:
                continue
            
            prompt_list = generate_prompt_mask(lbl_patch, num_prompts=num_prompts)
            
            if len(prompt_list) == 0:
                continue
            
            for prompt_idx, (prompt_256, guide_label) in enumerate(prompt_list):
                save_patch(img_patch, lbl_patch, prompt_256, guide_label,
                           out_img_dir, out_gt_dir, out_prompt_dir, out_guide_dir,
                           basename, idx, prompt_idx)
            
            kept_patches += 1
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    
    return total_patches, kept_patches

def process_scale_level_parallel(pairs_dict, level_idx, scale_factor, patch_size, overlap,
                                  output_dir, num_prompts, num_workers=None):
    """Process one scale level for all splits using parallel processing."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 4)
    
    level_name = f"level{level_idx}"
    print(f"\n===== Processing {level_name} (scale={scale_factor}) with {num_workers} workers =====")
    
    for split_name, pairs in pairs_dict.items():
        print(f"  Processing {split_name} split ({len(pairs)} images)...")
        
        out_img_dir = join(output_dir, level_name, split_name, "imgs")
        out_gt_dir = join(output_dir, level_name, split_name, "gts")
        out_prompt_dir = join(output_dir, level_name, split_name, "prompt_mask_256")
        out_guide_dir = join(output_dir, level_name, split_name, "prompt_gts")
        
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_gt_dir, exist_ok=True)
        os.makedirs(out_prompt_dir, exist_ok=True)
        os.makedirs(out_guide_dir, exist_ok=True)
        
        out_dirs = (out_img_dir, out_gt_dir, out_prompt_dir, out_guide_dir)
        tasks = [(img, lbl, scale_factor, patch_size, overlap, num_prompts, out_dirs) for img, lbl in pairs]
        
        total_patches = 0
        kept_patches = 0
        
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_single_image, tasks), total=len(pairs),
                                desc=f"  {level_name}/{split_name}"))
        
        for tp, kp in results:
            total_patches += tp
            kept_patches += kp
        
        print(f"  {split_name}: {total_patches} total patches, {kept_patches} kept (with water)")


def main():
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Handle zip file
    if input_dir.endswith(".zip"):
        if not os.path.exists(input_dir):
            print(f"Error: zip file not found: {input_dir}")
            return
        input_dir = unzip_dataset(input_dir, output_dir)
    
    # Auto-discover structure
    img_dir, lbl_dir = discover_dataset_structure(input_dir)
    
    # Check if already split into train/val/test
    splits = discover_splits(input_dir, img_dir, lbl_dir)
    
    if splits is not None:
        print(f"Detected pre-split dataset structure: {list(splits.keys())}")
        pairs_dict = splits
    else:
        # Match pairs
        pairs = match_image_label_pairs(img_dir, lbl_dir)
        if len(pairs) == 0:
            print("Error: No image-label pairs found. Please check dataset structure.")
            return
        
        # Split
        pairs_dict = split_dataset(pairs, train_ratio=0.6, val_ratio=0.2, seed=args.seed)
    
    # Process each scale level
    for level_idx, scale_factor in enumerate(args.scales):
        process_scale_level_parallel(pairs_dict, level_idx, scale_factor, 
                                      args.patch_size, args.overlap, 
                                      output_dir, args.num_prompts, args.num_workers)
    
    print(f"\n===== Preprocessing Complete =====")
    print(f"Output directory: {output_dir}")
    print(f"Scale levels: {args.scales}")
    print(f"Patch size: {args.patch_size}")
    print(f"Prompts per patch: {args.num_prompts}")
    print("\nDirectory structure:")
    for level_idx in range(len(args.scales)):
        for split in ["train", "val", "test"]:
            d = join(output_dir, f"level{level_idx}", split)
            if os.path.exists(d):
                n = len(os.listdir(join(d, "imgs"))) if os.path.exists(join(d, "imgs")) else 0
                print(f"  {d}/imgs: {n} files")


if __name__ == "__main__":
    main()
