import os
import glob
import numpy as np
from skimage import io

def check_data(data_dir, num_samples=5):
    gt_dir = os.path.join(data_dir, "gts")
    prompt_dir = os.path.join(data_dir, "prompt_mask_256")
    img_dir = os.path.join(data_dir, "imgs")
    
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "**/*.png"), recursive=True))[:num_samples]
    
    print(f"Checking {num_samples} samples from {data_dir}")
    print("=" * 60)
    
    for f in gt_files:
        base = os.path.splitext(os.path.basename(f))[0]
        gt = io.imread(f)
        prompt = io.imread(os.path.join(prompt_dir, base + ".png"))
        
        # Try to find image
        img = None
        for ext in [".tif", ".png"]:
            img_path = os.path.join(img_dir, base + ext)
            if os.path.exists(img_path):
                img = io.imread(img_path)
                break
        
        print(f"\nSample: {base}")
        print(f"  GT:       shape={gt.shape}, dtype={gt.dtype}, min={gt.min()}, max={gt.max()}, unique={np.unique(gt)[:5]}")
        print(f"  Prompt:   shape={prompt.shape}, dtype={prompt.dtype}, min={prompt.min()}, max={prompt.max()}")
        if img is not None:
            print(f"  Image:    shape={img.shape}, dtype={img.dtype}, min={img.min():.2f}, max={img.max():.2f}")
        
        # After division by 255
        gt_norm = gt / 255.0
        prompt_norm = prompt / 255.0
        print(f"  GT/255:   min={gt_norm.min():.4f}, max={gt_norm.max():.4f}, mean={gt_norm.mean():.4f}")
        print(f"  Prompt/255: min={prompt_norm.min():.4f}, max={prompt_norm.max():.4f}, mean={prompt_norm.mean():.4f}")

check_data("./data/level0/train")
