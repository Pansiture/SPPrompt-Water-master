# author MengFanlin + EDL modification
# time 2024/12/6
# filename train_edl
# description: Training script for EDL-enhanced SPPrompt-Water
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import sys
import argparse
import torch
import torch.nn.functional as F
import monai
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from skimage import io
from tqdm import tqdm
from datetime import datetime
import shutil
import os
from SPP_model.modeling.prompt_water_net_edl import SPPromptWaterNetEDL
from SPP_model.modeling.edl_utils import edl_loss, evidence_to_prob_uncertainty, sam_referee_loss
from RSDataloader import PromptDataset_GID5
from sklearn.metrics import confusion_matrix


join = os.path.join

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument("--data_train", type=str, default=r"/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/train",
                    help="path to training data; 3 subfolders: gts , imgs and prompt_mask_256")
parser.add_argument("--data_val", type=str, default=None,
                    help="Optional: explicit val data path. If None, will replace 'train' with 'val' in data_train")
parser.add_argument("--promptcp", type=str, default=r"/root/autodl-tmp/SPPrompt-Water-master/pretrain/sam_vit_b_01ec64.pth",
                    help="The checkpoint of Prompt model (SAM vit-b pretrain weights, optional)")
parser.add_argument(
    "--freeze_prompt", type=str2bool, default=False, help="Freeze the prompt module (default False to allow fine-tuning SAM)"
)
parser.add_argument("--SwintransformerPretrain", type=str, default="/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth",
                    help="Path to Swin Transformer pretrain weights")
parser.add_argument("--work_dir", type=str, default=r"/root/autodl-tmp/SPPrompt-Water-master/work_dir")

parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--task_name", type=str, default="SPP_GID_EDL")

parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--val_batch_size", type=int, default=4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--use_wandb", type=str2bool, default=False)
parser.add_argument("--use_amp", action="store_true", default=True)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--device", type=str, default="cuda:0")

# EDL-specific hyperparameters
parser.add_argument("--referee_weight", type=float, default=0.1, help="Weight for SAM referee consistency loss")
parser.add_argument("--kl_anneal_ratio", type=float, default=0.5, help="Ratio of epochs for KL annealing")
parser.add_argument("--referee_interval", type=int, default=1, help="Compute referee loss every N batches (1=every batch)")
parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs for linear lr warmup (0 to disable)")
parser.add_argument("--pretrain_ckpt", type=str, default="", help="Path to original SPPromptWaterNet best checkpoint for compatible weight initialization (EDL head will be randomly initialized)")
parser.add_argument("--kl_scale", type=float, default=0.5, help="Global scaling factor for KL term in EDL loss")
parser.add_argument("--pos_weight", type=float, default=10.0, help="Foreground weight for EDL MSE loss to combat class imbalance")
parser.add_argument("--seg_anchor_weight", type=float, default=0.5, help="BCE segmentation anchor loss weight to prevent EDL from collapsing")
parser.add_argument("--use_referee", type=str2bool, default=False, help="Use frozen SAM semantic referee branch (needs extra VRAM)")
parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")

args = parser.parse_args()

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)

# Disable cuDNN to avoid CUDNN_STATUS_INTERNAL_ERROR with certain PyTorch versions
# Trade: slower training, but guaranteed stability
# torch.backends.cudnn.enabled = False  # Uncomment if error persists after reducing batch_size
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

os.makedirs(model_save_path, exist_ok=True)

log_file = join(model_save_path, f"{run_id}_training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

logger.info("=" * 60)
logger.info("Training Configuration (EDL Version):")
logger.info("  data_train:          %s", args.data_train)
logger.info("  data_val:            %s", args.data_val if args.data_val else args.data_train.replace("train", "val"))
logger.info("  promptcp (SAM ckpt): %s", args.promptcp)
logger.info("  Swin Pretrain:       %s", args.SwintransformerPretrain)
logger.info("  freeze_prompt:       %s", args.freeze_prompt)
logger.info("  work_dir:            %s", args.work_dir)
logger.info("  task_name:           %s", args.task_name)
logger.info("  num_epochs:          %d", args.num_epochs)
logger.info("  batch_size:          %d", args.batch_size)
logger.info("  val_batch_size:      %d", args.val_batch_size)
logger.info("  lr:                  %.6f", args.lr)
logger.info("  weight_decay:        %.6f", args.weight_decay)
logger.info("  device:              %s", args.device)
logger.info("  use_amp:             %s", args.use_amp)
logger.info("  referee_weight:      %.3f", args.referee_weight)
logger.info("  kl_anneal_ratio:     %.2f", args.kl_anneal_ratio)
logger.info("  warmup_epochs:       %d", args.warmup_epochs)
logger.info("=" * 60)


def calculate_metrics(preds, labels, num_classes=2):
    """Calculate mIoU, accuracy, and F1-score."""
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    miou_list = []
    f1_list = []
    acc_list = []

    for pred, label in zip(preds, labels):
        pred = pred.squeeze()
        pred = (pred > 0.5).astype(int)
        label = label.squeeze()
        flat_pred = pred.flatten()
        flat_label = label.flatten()

        cm = confusion_matrix(flat_label, flat_pred, labels=list(range(num_classes)))

        intersection = np.diag(cm)
        union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
        iou = intersection / (union + 1e-6)
        miou = np.nanmean(iou)

        precision = intersection / (cm.sum(axis=0) + 1e-6)
        recall = intersection / (cm.sum(axis=1) + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        accuracy = intersection.sum() / (cm.sum() + 1e-6)

        miou_list.append(miou)
        f1_list.append(np.mean(f1))
        acc_list.append(accuracy)

    return np.mean(miou_list), np.mean(acc_list), np.mean(f1_list)


def validate(prompt_water_net, val_dataloader, device, epoch, num_epochs, use_referee=False, kl_scale=0.5, kl_anneal_ratio=0.5, referee_weight=0.1, pos_weight=10.0):
    """Validation loop with EDL losses."""
    prompt_water_net.eval()
    val_loss = 0
    val_edl_loss = 0
    val_ref_loss = 0
    miou_sum, acc_sum, f1_sum = 0, 0, 0
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for image, labels, prompts, _ in tqdm(val_dataloader, desc="Validation"):
            image, prompts, labels = image.to(device), prompts.to(device), labels.to(device).float()
            # Normalize labels to 0/1 for EDL (masks may be 0/255)
            if labels.max() > 1.0:
                labels = labels / 255.0

            # Ensure label size matches evidence size (1024x1024)
            evidence = prompt_water_net(image, prompts, return_evidence=True)
            if labels.shape[-2:] != evidence.shape[-2:]:
                labels = F.interpolate(labels, size=evidence.shape[-2:], mode='nearest')

            # EDL loss
            loss, mse, kl, annealing = edl_loss(
                evidence, labels, epoch, num_epochs,
                kl_anneal_epochs_ratio=kl_anneal_ratio,
                kl_scale=kl_scale,
                pos_weight=pos_weight
            )

            # Convert evidence to probability for referee and metrics
            prob, _ = evidence_to_prob_uncertainty(evidence)
            prob_fg = prob[:, 1:2, :, :]

            # SAM referee loss
            if use_referee:
                sam_prob = prompt_water_net.forward_referee(image)
                ref_loss = F.mse_loss(prob_fg, sam_prob)
            else:
                ref_loss = torch.tensor(0.0, device=device)

            total_loss = loss + referee_weight * ref_loss
            val_loss += total_loss.item()
            val_edl_loss += loss.item()
            val_ref_loss += ref_loss.item()

            # Metrics from foreground probability
            miou, acc, f1 = calculate_metrics(prob_fg.detach(), labels)
            miou_sum += miou
            acc_sum += acc
            f1_sum += f1

    return (val_loss / num_batches, val_edl_loss / num_batches, val_ref_loss / num_batches,
            miou_sum / num_batches, acc_sum / num_batches, f1_sum / num_batches)


def main():
    os.makedirs(model_save_path, exist_ok=True)

    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    promptcheckpoint = args.promptcp if args.promptcp else None
    prompt_water_net = SPPromptWaterNetEDL(
        promptcheckpoint,
        args.SwintransformerPretrain,
        freeze_prompt=args.freeze_prompt,
        use_referee=args.use_referee
    ).to(device)
    prompt_water_net.train()

    logger.info("Number of total parameters: %s", sum(p.numel() for p in prompt_water_net.parameters()))
    logger.info("Number of trainable parameters: %s", sum(p.numel() for p in prompt_water_net.parameters() if p.requires_grad))

    # Freeze backbone: keep SAM prompt frozen, unfreeze last 2 Swin stages for EDL adaptation
    frozen_backbone = 0
    for name, p in prompt_water_net.named_parameters():
        if "prompt_module" in name:
            p.requires_grad = False
            frozen_backbone += 1
        elif "swin" in name:
            # Unfreeze last two stages (layers.2 and layers.3) to adapt features for new EDL head
            if any(f"layers.{i}" in name for i in [2, 3]):
                p.requires_grad = True
            else:
                p.requires_grad = False
                frozen_backbone += 1
    logger.info("Frozen backbone params: %d (Swin early stages + SAM prompt_module)", frozen_backbone)
    logger.info("Trainable after freeze: %d", sum(p.numel() for p in prompt_water_net.parameters() if p.requires_grad))

    # Parameter groups: tune (uperhead internals) vs new (conv_seg, conv_fusion, evidence_act)
    tune_params = []
    new_params = []
    for name, p in prompt_water_net.named_parameters():
        if not p.requires_grad:
            continue
        if "uperhead.conv_seg" in name or "conv_fusion" in name or "evidence_act" in name or "sam_referee" in name:
            new_params.append(p)
        else:
            tune_params.append(p)

    logger.info("Trainable tune params (uperhead internals): %d", len(tune_params))
    logger.info("Trainable new params (heads): %d", len(new_params))

    optimizer = torch.optim.AdamW(
        [
            {"params": tune_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": new_params, "lr": args.lr * 10, "weight_decay": args.weight_decay},
        ]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-5
    )

    # Warmup scheduler (if warmup_epochs > 0)
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        def warmup_fn(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            return 1.0
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)

    logger.info("=" * 60)
    logger.info("Weight Loading Paths:")
    logger.info("  SAM prompt checkpoint:  %s", args.promptcp if args.promptcp else "None")
    logger.info("  Swin Transformer:       %s", args.SwintransformerPretrain)
    logger.info("  Resume checkpoint:      %s", args.resume if args.resume else "None")
    logger.info("=" * 60)
    logger.info("Model Architecture Details:")
    logger.info("  Total parameters:     %s", sum(p.numel() for p in prompt_water_net.parameters()))
    logger.info("  Trainable parameters: %s", sum(p.numel() for p in prompt_water_net.parameters() if p.requires_grad))
    logger.info("  freeze_prompt:        %s", args.freeze_prompt)
    logger.info("  Optimizer:            AdamW")
    logger.info("  LR Scheduler:         Warmup(%d) + CosineAnnealingLR" % args.warmup_epochs)
    logger.info("  Loss:                 EDL (MSE + annealed KL)%s", " + SAM Referee" if args.use_referee else "")
    logger.info("  AMP:                  %s", args.use_amp)
    logger.info("=" * 60)

    logger.info("Pretrained Weights Status:")
    for name, path in [("SAM prompt", args.promptcp), ("Swin pretrain", args.SwintransformerPretrain)]:
        exists = os.path.isfile(path) if path else False
        logger.info("  %-30s %s  (%s)", name, "EXISTS" if exists else "MISSING", path if path else "None")
    logger.info("=" * 60)

    logger.info("Dataset Path Check:")
    levels = ["level0", "level1", "level2"]
    all_ok = True
    for lv in levels:
        train_path = args.data_train.replace("level0", lv)
        val_path = args.data_train.replace("train", "val").replace("level0", lv)
        for name, path in [(f"{lv}/train", train_path), (f"{lv}/val", val_path)]:
            exists = os.path.isdir(path)
            if not exists:
                all_ok = False
            logger.info("  %-15s %s  (%s)", name, "EXISTS" if exists else "MISSING", path)
    if not all_ok:
        logger.error("Some dataset paths are missing!")
        raise FileNotFoundError("Dataset paths missing.")
    logger.info("=" * 60)

    num_epochs = args.num_epochs
    train_loss = []
    val_loss_list = []
    val_edl_loss_list = []
    val_ref_loss_list = []
    best_Valscore = 0.001
    previous_best_model = None

    # Multi-level unified dataset
    train_datasets = []
    for lv in levels:
        ds = PromptDataset_GID5(args.data_train.replace("level0", lv))
        train_datasets.append(ds)
    train_dataset = ConcatDataset(train_datasets)
    logger.info("Number of training samples: %s", len(train_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_datasets = []
    for lv in levels:
        ds = PromptDataset_GID5(args.data_train.replace("train", "val").replace("level0", lv))
        val_datasets.append(ds)
    val_dataset = ConcatDataset(val_datasets)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    start_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        prompt_water_net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Resumed from epoch %d", start_epoch)

    # Load original best checkpoint for compatible weight initialization
    if args.pretrain_ckpt and os.path.isfile(args.pretrain_ckpt):
        logger.info("Loading compatible weights from pretrain checkpoint: %s", args.pretrain_ckpt)
        pretrain_ckpt = torch.load(args.pretrain_ckpt, map_location=device)
        pretrain_state = pretrain_ckpt.get("model", pretrain_ckpt)
        model_state = prompt_water_net.state_dict()

        compatible_state = {}
        skipped_keys = []
        for k, v in model_state.items():
            if k in pretrain_state and pretrain_state[k].shape == v.shape:
                compatible_state[k] = pretrain_state[k]
            else:
                skipped_keys.append(k)

        # === Smart weight mapping: old SPP (1ch) -> EDL (2ch) ===
        # 1. Map uperhead.conv_seg: [1,512,1,1] -> [2,512,1,1]
        if "uperhead.conv_seg.weight" in pretrain_state and "uperhead.conv_seg.weight" in model_state:
            old_w = pretrain_state["uperhead.conv_seg.weight"]  # [1, 512, 1, 1]
            new_w = model_state["uperhead.conv_seg.weight"].clone()  # [2, 512, 1, 1]
            new_w[0] = -old_w[0]       # bg evidence raw: mirror of fg (so bg = -fg after fusion)
            new_w[1] = old_w[0]       # fg evidence: inherit old seg head
            compatible_state["uperhead.conv_seg.weight"] = new_w
            if "uperhead.conv_seg.bias" in pretrain_state and "uperhead.conv_seg.bias" in model_state:
                old_b = pretrain_state["uperhead.conv_seg.bias"]
                new_b = model_state["uperhead.conv_seg.bias"].clone()
                new_b[0] = -old_b[0]    # bg bias: mirror of fg bias
                new_b[1] = old_b[0]
                compatible_state["uperhead.conv_seg.bias"] = new_b
            logger.info("[Weight Map] uperhead.conv_seg: old 1ch -> new 2ch (fg inherits, bg = -fg)")
            if "uperhead.conv_seg.weight" in skipped_keys:
                skipped_keys.remove("uperhead.conv_seg.weight")
            if "uperhead.conv_seg.bias" in skipped_keys:
                skipped_keys.remove("uperhead.conv_seg.bias")

        # 2. Map conv_fusion: old FeatureFusion.Conv2d(2,1) -> new Conv2d(3,2)
        if "conv_fusion.conv_fusion.weight" in pretrain_state and "conv_fusion.weight" in model_state:
            old_w = pretrain_state["conv_fusion.conv_fusion.weight"]  # [1, 2, 1, 1]
            new_w = model_state["conv_fusion.weight"].clone()  # [2, 3, 1, 1]
            # Correct channel mapping:
            #   new input ch0 = swin_ch0 (bg evidence raw, NEW)
            #   new input ch1 = swin_ch1 (fg evidence raw, inherited from old swin)
            #   new input ch2 = prompt (inherited from old prompt)
            # out_ch1 (fg evidence):
            new_w[1, 0, :, :] = 0.0                    # ignore swin_ch0
            new_w[1, 1, :, :] = old_w[0, 0, :, :]      # old swin weight -> new swin_ch1
            new_w[1, 2, :, :] = old_w[0, 1, :, :]      # old prompt weight -> new prompt
            # out_ch0 (bg evidence): mirror of fg weights so bg_raw = -fg_raw
            new_w[0, 0, :, :] = 0.0
            new_w[0, 1, :, :] = -old_w[0, 0, :, :]    # negative of old swin weight
            new_w[0, 2, :, :] = -old_w[0, 1, :, :]    # negative of old prompt weight
            compatible_state["conv_fusion.weight"] = new_w
            if "conv_fusion.conv_fusion.bias" in pretrain_state and "conv_fusion.bias" in model_state:
                old_b = pretrain_state["conv_fusion.conv_fusion.bias"]
                new_b = model_state["conv_fusion.bias"].clone()
                new_b[1] = old_b[0]
                new_b[0] = -old_b[0]     # bg bias: mirror of fg bias
                compatible_state["conv_fusion.bias"] = new_b
            logger.info("[Weight Map] conv_fusion: old FeatureFusion 2->1 -> new 3->2 (corrected channel alignment, bg = -fg)")
            if "conv_fusion.weight" in skipped_keys:
                skipped_keys.remove("conv_fusion.weight")
            if "conv_fusion.bias" in skipped_keys:
                skipped_keys.remove("conv_fusion.bias")

        model_state.update(compatible_state)
        prompt_water_net.load_state_dict(model_state, strict=False)
        logger.info("Loaded %d / %d parameters from pretrain_ckpt (including mapped)", len(compatible_state), len(model_state))
        if skipped_keys:
            logger.info("Skipped %d keys (shape mismatch or new layers): %s", len(skipped_keys), ", ".join(skipped_keys[:10]) + ("..." if len(skipped_keys) > 10 else ""))

    if args.use_amp:
        scaler = torch.amp.GradScaler('cuda')

    # ---- Smoke test: verify forward + backward + validation before first epoch ----
    logger.info("=" * 60)
    logger.info("Running smoke test on one batch...")
    try:
        prompt_water_net.train()
        batch = next(iter(train_dataloader))
        img_smoke, lbl_smoke, prm_smoke, _ = batch
        img_smoke = img_smoke[:1].to(device)
        lbl_smoke = lbl_smoke[:1].to(device).float()
        prm_smoke = prm_smoke[:1].to(device)

        optimizer.zero_grad()
        ev = prompt_water_net(img_smoke, prm_smoke, return_evidence=True)
        if lbl_smoke.shape[-2:] != ev.shape[-2:]:
            lbl_smoke = F.interpolate(lbl_smoke, size=ev.shape[-2:], mode='nearest')
        loss_smoke, mse_smoke, kl_smoke, anneal_smoke = edl_loss(
            ev, lbl_smoke, 0, 1, kl_anneal_epochs_ratio=args.kl_anneal_ratio, kl_scale=args.kl_scale
        )
        if args.use_referee:
            sp = prompt_water_net.forward_referee(img_smoke)
            prob_smoke, _ = evidence_to_prob_uncertainty(ev)
            ref_smoke = F.mse_loss(prob_smoke[:, 1:2], sp)
            loss_smoke = loss_smoke + args.referee_weight * ref_smoke
        loss_smoke.backward()
        optimizer.step()

        # Quick validation check (forward only)
        prompt_water_net.eval()
        with torch.no_grad():
            ev_val = prompt_water_net(img_smoke, prm_smoke, return_evidence=True)
            prob_val, _ = evidence_to_prob_uncertainty(ev_val)
            prob_fg_val = prob_val[:, 1:2, :, :]
            miou_smoke, acc_smoke, f1_smoke = calculate_metrics(prob_fg_val.detach(), lbl_smoke)
        logger.info("Smoke test passed: forward/backward/val OK. mIoU=%.4f, Acc=%.4f, F1=%.4f",
                    miou_smoke, acc_smoke, f1_smoke)
        logger.info("=" * 60)
    except Exception as e:
        logger.error("Smoke test FAILED: %s", str(e))
        raise

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_edl_loss = 0
        epoch_ref_loss = 0
        prompt_water_net.train()
        accum_counter = 0

        for step, (image, labels, prompts, _) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
            if accum_counter == 0:
                optimizer.zero_grad()
            image, prompts, labels = image.to(device), prompts.to(device), labels.to(device).float()
            # Normalize labels to 0/1 for EDL (masks may be 0/255)
            if labels.max() > 1.0:
                labels = labels / 255.0

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    evidence = prompt_water_net(image, prompts, return_evidence=True)
                    if labels.shape[-2:] != evidence.shape[-2:]:
                        labels = F.interpolate(labels, size=evidence.shape[-2:], mode='nearest')

                    # EDL loss
                    loss, mse, kl, annealing = edl_loss(
                        evidence, labels, epoch, num_epochs,
                        kl_anneal_epochs_ratio=args.kl_anneal_ratio,
                        kl_scale=args.kl_scale,
                        pos_weight=args.pos_weight
                    )

                    # Segmentation anchor (BCE with logits) to prevent collapse
                    # Anchor both fg and bg logits to old model behavior
                    logit_fg = evidence[:, 1:2, :, :]
                    logit_bg = evidence[:, 0:1, :, :]
                    seg_anchor_fg = F.binary_cross_entropy_with_logits(logit_fg, labels)
                    seg_anchor_bg = F.binary_cross_entropy_with_logits(logit_bg, 1.0 - labels)
                    seg_anchor = seg_anchor_fg + seg_anchor_bg

                    # SAM referee consistency (only when use_referee=True)
                    if args.use_referee and step % args.referee_interval == 0:
                        prob, _ = evidence_to_prob_uncertainty(evidence)
                        prob_fg = prob[:, 1:2, :, :]
                        sam_prob = prompt_water_net.forward_referee(image)
                        ref_loss = F.mse_loss(prob_fg, sam_prob)
                    else:
                        ref_loss = torch.tensor(0.0, device=device)

                    total_loss = (loss + args.seg_anchor_weight * seg_anchor + args.referee_weight * ref_loss) / args.grad_accum

                scaler.scale(total_loss).backward()
            else:
                evidence = prompt_water_net(image, prompts, return_evidence=True)
                if labels.shape[-2:] != evidence.shape[-2:]:
                    labels = F.interpolate(labels, size=evidence.shape[-2:], mode='nearest')

                loss, mse, kl, annealing = edl_loss(
                    evidence, labels, epoch, num_epochs,
                    kl_anneal_epochs_ratio=args.kl_anneal_ratio,
                    kl_scale=args.kl_scale,
                    pos_weight=args.pos_weight
                )

                # Segmentation anchor (BCE with logits) to prevent collapse
                logit_fg = evidence[:, 1:2, :, :]
                logit_bg = evidence[:, 0:1, :, :]
                seg_anchor_fg = F.binary_cross_entropy_with_logits(logit_fg, labels)
                seg_anchor_bg = F.binary_cross_entropy_with_logits(logit_bg, 1.0 - labels)
                seg_anchor = seg_anchor_fg + seg_anchor_bg

                if args.use_referee and step % args.referee_interval == 0:
                    prob, _ = evidence_to_prob_uncertainty(evidence)
                    prob_fg = prob[:, 1:2, :, :]
                    sam_prob = prompt_water_net.forward_referee(image)
                    ref_loss = F.mse_loss(prob_fg, sam_prob)
                else:
                    ref_loss = torch.tensor(0.0, device=device)

                total_loss = (loss + args.seg_anchor_weight * seg_anchor + args.referee_weight * ref_loss) / args.grad_accum
                total_loss.backward()

            epoch_loss += total_loss.item() * args.grad_accum
            epoch_edl_loss += loss.item()
            epoch_ref_loss += ref_loss.item()
            accum_counter += 1

            if accum_counter == args.grad_accum:
                if args.use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                accum_counter = 0

        # Handle any remaining accumulated gradients at end of epoch
        if accum_counter > 0:
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        num_steps = step + 1
        epoch_loss /= num_steps
        epoch_edl_loss /= num_steps
        epoch_ref_loss /= num_steps
        train_loss.append(epoch_loss)

        logger.info(
            'Time: %s, Epoch: %d, Loss: %.4f (EDL: %.4f, Ref: %.4f), LR: base=%.6f head=%.6f, KL_coef: %.4f',
            datetime.now().strftime("%Y%m%d-%H%M"), epoch, epoch_loss, epoch_edl_loss, epoch_ref_loss,
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], annealing
        )

        # Validation
        val_total, val_edl, val_ref, val_miou, val_acc, val_f1 = validate(
            prompt_water_net, val_dataloader, device, epoch, num_epochs,
            use_referee=args.use_referee,
            kl_scale=args.kl_scale,
            kl_anneal_ratio=args.kl_anneal_ratio,
            referee_weight=args.referee_weight,
            pos_weight=args.pos_weight
        )
        val_loss_list.append(val_total)
        val_edl_loss_list.append(val_edl)
        val_ref_loss_list.append(val_ref)

        Valscore = val_miou + val_acc + val_f1
        logger.info(
            'Validation - Loss: %.4f (EDL: %.4f, Ref: %.4f), mIoU: %.4f, Acc: %.4f, F1: %.4f, Valscore: %.4f',
            val_total, val_edl, val_ref, val_miou, val_acc, val_f1, Valscore
        )

        # Save best model
        checkpoint = {
            "model": prompt_water_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }

        if Valscore > best_Valscore:
            best_Valscore = Valscore
            new_best_model_filename = f"best_model_e{epoch}_Valscore{Valscore:.4f}.pth"
            new_best_model_path = join(model_save_path, new_best_model_filename)
            if previous_best_model is not None and os.path.exists(previous_best_model):
                os.remove(previous_best_model)
            torch.save(checkpoint, new_best_model_path)
            previous_best_model = new_best_model_path
            logger.info("New best model saved: %s", new_best_model_filename)

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, join(model_save_path, f'checkpoint_e{epoch}.pth'))

        # Step schedulers: warmup first, then cosine
        if warmup_scheduler is not None and epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Plot curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss_list, label="Val Total Loss")
        plt.plot(val_edl_loss_list, label="Val EDL Loss")
        plt.plot(val_ref_loss_list, label="Val Ref Loss")
        plt.title("Train and Validation Loss (EDL)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(join(model_save_path, args.task_name + "_loss_curve.png"))
        plt.close()


if __name__ == "__main__":
    main()
