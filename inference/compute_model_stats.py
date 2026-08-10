#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute Params, FLOPs, and Inference Time for all models.
Results for TABLE VII: Model Size and Inference Efficiency Comparison.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SPP_model.modeling.other_model.MECNet import MECNet
from SPP_model.modeling.other_model.MSResNet import MSResNet
from SPP_model.modeling.other_model.QTNUnet import qtnUnet
from SPP_model.build_sam import sam_model_registry
from SPP_model.modeling.prompt_water_net import SPPromptWaterNet
from train.train_deeplabv3plus import build_deeplabv3plus_res101
from train.train_baseline import UNet


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_time(model, device, input_size=(512, 512), num_runs=100, warmup=10):
    """Measure average inference time in ms for a single image."""
    model.eval()
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Synchronize and measure
    torch.cuda.synchronize() if device.type == 'cuda' else None
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def try_get_flops(model, input_size=(512, 512)):
    """Try to compute FLOPs using thop."""
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9, params / 1e6  # GFLOPs, MParams
    except Exception:
        return None, None


def build_sppromptwaternet(device):
    """Build SPPromptWaterNet with frozen image encoder."""
    sam_checkpoint = "/root/autodl-tmp/SPPrompt-Water-master/pretrained/sam_vit_b_01ec64.pth"
    model = SPPromptWaterNet(promptcheckpoint=sam_checkpoint, freeze_prompt=True)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--num_runs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_size = tuple(args.input_size)

    print("=" * 80)
    print(f"Computing Model Stats on {device}, input size: {input_size}")
    print("=" * 80)

    # Define all models
    models_info = []

    # 1. UNet
    models_info.append(("UNet", lambda: UNet(in_channels=3, out_channels=1)))

    # 2. MSResNet
    models_info.append(("MSResNet", lambda: MSResNet(in_channels=3, num_classes=1, pretrained_path=None)))

    # 3. QTNUnet
    models_info.append(("QTNUnet", lambda: qtnUnet(
        img_size=1024, patch_size=4, in_chans=3, num_classes=1,
        embed_dims=[16, 32, 64, 128, 256], mlp_ratios=[4, 4, 4, 4, 4],
        drop_rate=0.0, drop_path_rate=0.1, depths=[3, 3, 3, 3, 2],
        num_stages=5, linear=False, pretrained=None
    )))

    # 4. MECNet
    models_info.append(("MECNet", lambda: MECNet()))

    # 5. DeepLabV3+
    models_info.append(("DeepLabV3+", lambda: build_deeplabv3plus_res101(num_classes=1, pretrained=False)))

    # 6. SPPromptWaterNet (EDL) - need dummy prompt_mask
    def build_spprompt():
        model = build_sppromptwaternet(device)
        # Wrap to accept single input
        class Wrapper(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, x):
                B = x.shape[0]
                prompt = torch.zeros(B, 1, 256, 256, device=x.device)
                return self.base(x, prompt)
        return Wrapper(model)
    models_info.append(("SPPromptWaterNet", build_spprompt))

    results = []
    for name, build_fn in models_info:
        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print(f"{'='*40}")

        try:
            model = build_fn()
            if not next(model.parameters()).is_cuda:
                model = model.to(device)

            total_p, trainable_p = count_parameters(model)
            print(f"  Total params:      {total_p:,} ({total_p/1e6:.2f} M)")
            print(f"  Trainable params:  {trainable_p:,} ({trainable_p/1e6:.2f} M)")

            # FLOPs - move to CPU for thop
            model_cpu = model.cpu()
            flops_g, params_m = try_get_flops(model_cpu, input_size)
            if flops_g is not None:
                print(f"  FLOPs:             {flops_g:.2f} G")
            else:
                print(f"  FLOPs:             N/A")

            # Move back to device for timing
            model = model.to(device)
            avg_time, std_time = measure_inference_time(model, device, input_size, args.num_runs)
            print(f"  Inference time:    {avg_time:.2f} ± {std_time:.2f} ms")

            results.append({
                "Method": name,
                "Total (M)": total_p / 1e6,
                "Trainable (M)": trainable_p / 1e6,
                "FLOPs (G)": flops_g if flops_g is not None else "N/A",
                "Time (ms)": f"{avg_time:.2f}±{std_time:.2f}"
            })

            # Clean up
            del model, model_cpu
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "Method": name,
                "Total (M)": "ERROR",
                "Trainable (M)": "ERROR",
                "FLOPs (G)": "ERROR",
                "Time (ms)": "ERROR"
            })

    # Print TABLE VII
    print("\n" + "=" * 80)
    print("TABLE VII: Model Size and Inference Efficiency Comparison")
    print("=" * 80)
    print(f"{'Method':<20} {'Params (M)':<15} {'Trainable (M)':<17} {'FLOPs (G)':<12} {'Time/ms (512x512)':<18}")
    print("-" * 80)
    for r in results:
        total_str = f"{r['Total (M)']:.2f}" if isinstance(r['Total (M)'], float) else str(r['Total (M)'])
        train_str = f"{r['Trainable (M)']:.2f}" if isinstance(r['Trainable (M)'], float) else str(r['Trainable (M)'])
        flops_str = f"{r['FLOPs (G)']:.2f}" if isinstance(r['FLOPs (G)'], float) else str(r['FLOPs (G)'])
        print(f"{r['Method']:<20} {total_str:<15} {train_str:<17} {flops_str:<12} {r['Time (ms)']:<18}")
    print("=" * 80)


if __name__ == "__main__":
    main()
