import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SPP_model.modeling.prompt_water_net import SPPromptWaterNet
from RSDataloader import PromptDataset_GID5
import monai

device = torch.device("cuda:0")

# 初始化模型（与训练时相同）
model = SPPromptWaterNet(
    promptcheckpoint="./pretrain/sam_vit_b_01ec64.pth",
    swin_pretrained="./pretrain/swin_tiny_patch4_window7_224.pth",
    freeze_prompt=False
).to(device)
model.train()

# 检查 Swin 权重是否加载
def check_swin_weights(m):
    w = m.swin.patch_embed.projection.weight
    print(f"Swin patch_embed weight mean: {w.mean().item():.4f}, std: {w.std().item():.4f}")
    # 随机初始化的均值通常接近0，std很小；加载预训练后应该有明显不同

check_swin_weights(model)

# 检查 SAM decoder 是否可训练
sam_dec_params = sum(p.numel() for p in model.prompt_module.mask_decoder.parameters() if p.requires_grad)
print(f"SAM decoder trainable params: {sam_dec_params}")

# 加载一个 batch
dataset = PromptDataset_GID5("./data/level0/train")
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
image, labels, prompts, _ = next(iter(loader))
image, labels, prompts = image.to(device), labels.to(device).float(), prompts.to(device)

print(f"\nInput shapes: image={image.shape}, labels={labels.shape}, prompts={prompts.shape}")
print(f"Label range: [{labels.min().item():.4f}, {labels.max().item():.4f}], mean={labels.mean().item():.4f}")

# 前向传播
pred = model(image, prompts)
print(f"\nModel output shape: {pred.shape}")
print(f"Output range: [{pred.min().item():.4f}, {pred.max().item():.4f}], mean={pred.mean().item():.4f}")

# 计算损失
dice_focal = monai.losses.DiceFocalLoss(sigmoid=True, reduction="mean", squared_pred=True, alpha=0.75)
tversky = monai.losses.TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
def combined_loss(pred, target):
    return 0.6 * dice_focal(pred, target) + 0.4 * tversky(pred, target)

loss = combined_loss(pred, labels)
print(f"\nLoss: {loss.item():.4f}")

# 反向传播，检查梯度
loss.backward()

print("\n--- Gradient norms ---")
total_norm = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        norm = param.grad.data.norm(2).item()
        if norm > 0:
            total_norm += norm ** 2
            if 'swin' in name or 'prompt_module' in name or 'conv_fusion' in name:
                print(f"  {name}: {norm:.6f}")
    else:
        if param.requires_grad:
            print(f"  WARNING: {name} has NO gradient!")

total_norm = total_norm ** 0.5
print(f"\nTotal gradient norm: {total_norm:.6f}")

# 统计各组件梯度
swin_grad = sum(p.grad.data.norm(2).item() ** 2 for p in model.swin.parameters() if p.grad is not None) ** 0.5
sam_grad = sum(p.grad.data.norm(2).item() ** 2 for p in model.prompt_module.parameters() if p.grad is not None) ** 0.5
fusion_grad = sum(p.grad.data.norm(2).item() ** 2 for p in model.conv_fusion.parameters() if p.grad is not None) ** 0.5

print(f"\nSwin grad norm: {swin_grad:.6f}")
print(f"SAM grad norm: {sam_grad:.6f}")
print(f"Fusion grad norm: {fusion_grad:.6f}")
