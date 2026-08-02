#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference on 7 specific images and save ONLY the prediction masks.
"""
import os
import sys
import numpy as np
from PIL import Image
import torch
from skimage import io

sys.path.insert(0, os.path.dirname(__file__))
from SPP_model.modeling.swin_net import SwinTransformerNet

join = os.path.join

# ---- config ----
CHECKPOINT = "/root/autodl-tmp/SPPrompt-Water-master/work_dir/GLH_swin_7epoch_e6_0.7324_2.4671/GLH_swin_7epoch_e6_0.7324_2.4671.pth"
DATA_VAL = "/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/val"
OUTPUT_DIR = "/root/autodl-tmp/SPPrompt-Water-master/output/swin_"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_NAMES = [
    "04_p134_v0.png",
    "106_p59_v1.png",
    "106_p93_v2.png",
    "116_p125_v2.png",
    "118_p58_v0.png",
    "137_p55_v0.png",
    "143_p33_v2.png",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading SwinTransformerNet...")
    model = SwinTransformerNet(pretrained=None).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    img_dir = join(DATA_VAL, "imgs")

    with torch.no_grad():
        for fname in IMAGE_NAMES:
            base_name = os.path.splitext(fname)[0]
            print(f"Processing {fname} ...")

            img_path = join(img_dir, fname)
            img = io.imread(img_path)  # (1024, 1024, 3), uint8
            img = np.transpose(img, (2, 0, 1))  # (3, 1024, 1024)
            img_tensor = torch.tensor(img).float().unsqueeze(0).to(DEVICE)

            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
            pred_np = pred[0, 0].cpu().numpy()  # (256, 256)

            # Upsample to 1024x1024 and binarize
            pred_1024 = np.array(
                Image.fromarray((pred_np * 255).astype(np.uint8)).resize((1024, 1024))
            )
            pred_binary = (pred_1024 > 127).astype(np.uint8) * 255

            out_path = join(OUTPUT_DIR, f"{base_name}_pred.png")
            Image.fromarray(pred_binary, mode='L').save(out_path)
            print(f"Saved -> {out_path}")

    print("All done.")


if __name__ == "__main__":
    main()
