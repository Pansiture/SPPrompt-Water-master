#!/usr/bin/env python
"""
GID-15 Water Body Preprocessing Script for SPPrompt-Water

This script prepares the GID-15 dataset for the paper:
"Breaking the Scale Barrier: A Scale-Progressive Water Body Extraction
Framework Inspired by Human Cognitive Mechanisms"

Steps:
1. Read GID-15 images (4-channel NIR+RGB .tif) and 15-class labels (.tif)
2. Convert 15-class labels to binary water masks (class 13,14,15 -> 255)
3. Remove NIR channel, keep only RGB
4. Split into train/val/test (60/20/20)
5. Generate multi-scale image pyramids (level0, level1, level2)
6. Crop into 1024x1024 patches
7. Filter patches without water bodies
8. Generate prompt masks via:
   - Connected-component decomposition (TopologicalDecomposition)
   - RandomMerge of subregions
   - Morphological perturbation (Erosion/Dilation, kernel size 3-10)
   - 4x downsampling to 256x256 prompt mask

Usage:
    python preprocess_gid_water.py --input_dir /path/to/data/GID/01 --output_dir /path/to/data/GID_processed

Output structure:
    data/GID_processed/
    ├── level0/
    │   ├── train/imgs, gts, prompt_mask_256, prompt_gts
    │   ├── val/imgs, gts, prompt_mask_256, prompt_gts
    │   └── test/imgs, gts, prompt_mask_256, prompt_gts
    ├── level1/
    └── level2/
"""

import os
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage import io, morphology
from scipy import ndimage
import multiprocessing as mp

join = os.path.join


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess GID-15 for SPPrompt-Water")
    parser.add_argument("--input_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GID/01",
                        help="Path to raw GID-15 data, containing Image__8bit_NirRGB/ and 150-15classes/")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed",
                        help="Output directory for processed data")
    parser.add_argument("--patch_size", type=int, default=1024,
                        help="Size of cropped patches (default: 1024)")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4],
                        help="Downsampling factors for each level (default: 1 2 4)")
    parser.add_argument("--overlap", type=int, default=0,
                        help="Overlap between patches in pixels (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_prompts", type=int, default=3,
                        help="Number of prompt variants to generate per patch (default: 3)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--water_classes", type=int, nargs="+", default=[13, 14, 15],
                        help="Pixel values representing water in GID-15 labels (default: 13 14 15)")
    return parser.parse_args()


def discover_gid_structure(root_dir):
    """
    Discover GID image and label directories.
    Expected structure:
        root_dir/
        ├── Image__8bit_NirRGB/
        └── 150-15classes/
    """
    root = Path(root_dir)

    img_dir = str(root / "Image__8bit_NirRGB") if (root / "Image__8bit_NirRGB").is_dir() else None
    lbl_dir = str(root / "150-15classes") if (root / "150-15classes").is_dir() else None

    if img_dir is None or lbl_dir is None:
        raise ValueError(
            f"Could not find 'Image__8bit_NirRGB/' or '150-15classes/' under {root_dir}. "
            f"Please ensure the GID-15 raw data is placed correctly."
        )

    print(f"Discovered image dir: {img_dir}")
    print(f"Discovered label dir: {lbl_dir}")
    return img_dir, lbl_dir


def match_pairs(img_dir, lbl_dir):
    """Match image and label files by exact filename (both are .tif)."""
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".tif")])
    lbl_files = set(f for f in os.listdir(lbl_dir) if f.endswith(".tif"))

    pairs = []
    for img in img_files:
        if img in lbl_files:
            pairs.append((join(img_dir, img), join(lbl_dir, img)))
        else:
            print(f"Warning: no label found for {img}")

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

    return {
        "train": [pairs[i] for i in train_idx],
        "val": [pairs[i] for i in val_idx],
        "test": [pairs[i] for i in test_idx]
    }


def read_image(path):
    """Read image as numpy array (H, W, C) or (H, W)."""
    img = io.imread(path)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    return img


def convert_label_to_water(label, water_classes):
    """Convert 15-class GID label to binary water mask (0 or 255)."""
    water = np.isin(label, water_classes).astype(np.uint8) * 255
    return water


def preprocess_image(img):
    """
    Remove NIR channel if present, keep only RGB.
    GID images are stored as NIR-R-G-B (4 channels).
    The first channel (NIR) is removed to match the model's 3-channel input.
    """
    if img.ndim == 3 and img.shape[2] >= 4:
        # NIR is the first channel, keep RGB (channels 1,2,3)
        img = img[:, :, 1:4]
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
        return None
    if len(subregions) == 1:
        return subregions[0]

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
        return []

    prompts = []
    for _ in range(num_prompts):
        merged = random_merge(subregions)
        perturbed = morphological_perturbation(merged)
        prompt_256 = ndimage.zoom(perturbed, 0.25, order=0)
        prompt_256 = (prompt_256 > 0).astype(np.uint8) * 255
        guide_label = (merged > 0).astype(np.uint8) * 255
        prompts.append((prompt_256, guide_label))
    return prompts


def save_patch(img_patch, lbl_patch, prompt_256, guide_label,
               out_img_dir, out_gt_dir, out_prompt_dir, out_guide_dir,
               basename, idx, prompt_idx):
    """Save a single patch and its prompt."""
    # Ensure 3-channel RGB
    if img_patch.ndim == 2:
        img_patch = np.stack([img_patch] * 3, axis=-1)
    elif img_patch.shape[2] == 1:
        img_patch = np.repeat(img_patch, 3, axis=-1)
    elif img_patch.shape[2] > 3:
        img_patch = img_patch[:, :, :3]

    # Convert to uint8
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
    if out_guide_dir is not None:
        io.imsave(join(out_guide_dir, filename), guide_label.astype(np.uint8), check_contrast=False)


def process_single_image(args_tuple):
    """Process a single image-label pair for one scale level."""
    img_path, lbl_path, scale_factor, patch_size, overlap, num_prompts, out_dirs, water_classes = args_tuple
    out_img_dir, out_gt_dir, out_prompt_dir, out_guide_dir = out_dirs

    total_patches = 0
    kept_patches = 0

    try:
        img = read_image(img_path)
        lbl = read_image(lbl_path)

        # Convert label to binary water mask
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        lbl = convert_label_to_water(lbl, water_classes)

        # Preprocess image: remove NIR, keep RGB
        img = preprocess_image(img)

        # Resize for multi-scale
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
        import traceback
        traceback.print_exc()

    return total_patches, kept_patches


def process_scale_level_parallel(pairs_dict, level_idx, scale_factor, patch_size, overlap,
                                  output_dir, num_prompts, num_workers, water_classes):
    """Process one scale level for all splits using parallel processing."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)

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
        tasks = [(img, lbl, scale_factor, patch_size, overlap, num_prompts, out_dirs, water_classes)
                 for img, lbl in pairs]

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

    # Auto-discover structure
    img_dir, lbl_dir = discover_gid_structure(input_dir)

    # Match pairs
    pairs = match_pairs(img_dir, lbl_dir)
    if len(pairs) == 0:
        print("Error: No image-label pairs found. Please check dataset structure.")
        return

    # Split dataset
    pairs_dict = split_dataset(pairs, train_ratio=0.6, val_ratio=0.2, seed=args.seed)
    print(f"Dataset split: {len(pairs_dict['train'])} train, {len(pairs_dict['val'])} val, {len(pairs_dict['test'])} test")

    # Process each scale level
    for level_idx, scale_factor in enumerate(args.scales):
        process_scale_level_parallel(pairs_dict, level_idx, scale_factor,
                                      args.patch_size, args.overlap,
                                      output_dir, args.num_prompts, args.num_workers,
                                      args.water_classes)

    print(f"\n===== Preprocessing Complete =====")
    print(f"Output directory: {output_dir}")
    print(f"Scale levels: {args.scales}")
    print(f"Patch size: {args.patch_size}")
    print(f"Prompts per patch: {args.num_prompts}")
    print(f"Water classes (GID-15): {args.water_classes}")
    print("\nDirectory structure:")
    for level_idx in range(len(args.scales)):
        for split in ["train", "val", "test"]:
            d = join(output_dir, f"level{level_idx}", split)
            if os.path.exists(d) and os.path.exists(join(d, "imgs")):
                n = len(os.listdir(join(d, "imgs")))
                print(f"  {d}/imgs: {n} files")

    print("\nNext step: modify train_promptwaternet.py --data_train to point to:")
    print(f"  {join(output_dir, 'level0', 'train')}")


if __name__ == "__main__":
    main()
