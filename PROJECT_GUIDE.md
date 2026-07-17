# SPPrompt-Water 项目文件整理说明

## 一、项目目录结构

```
SPPrompt-Water-master/
├── data/                          # 数据集
│   └── GLH-water/               # GLH-Water 数据集
│       ├── test/                  # 原始测试图像 (25张高分辨率大图)
│       │   ├── img/               # 测试图像 (.jpg)
│       │   └── label/             # 测试标签 (.png)
│       ├── train/                 # 原始训练图像 (200对)
│       ├── val/                   # 原始验证图像 (25对)
│       ├── level0/                # 预处理后 1x 分辨率 (1024x1024 patches)
│       │   ├── train/imgs, gts, prompt_mask_256
│       │   ├── val/imgs, gts, prompt_mask_256
│       │   └── test/imgs, gts, prompt_mask_256
│       ├── level1/                # 预处理后 2x 下采样
│       └── level2/                # 预处理后 4x 下采样
│
├── SPP_model/                     # 模型定义
│   ├── modeling/                  # 核心网络架构
│   │   ├── prompt_water_net.py   # 【SPPrompt-Water 主模型】
│   │   ├── swin_net.py           # 【SwinTransformer + UPerHead】
│   │   ├── swin_mmcv.py          # Swin Transformer backbone (mmcv)
│   │   ├── uper_head.py          # UPerHead 分割头
│   │   ├── Unet.py               # 【U-Net 模型】
│   │   └── other_model/          # 其他对比模型 (MECNet, MSResNet, QTNUnet等)
│   ├── utils/                     # 工具函数
│   └── ...
│
├── pretrain/                      # 预训练权重
│   ├── sam_vit_b_01ec64.pth      # SAM ViT-B 预训练权重
│   ├── swin_tiny_patch4_window7_224.pth  # Swin Transformer ImageNet 预训练权重
│   └── preprocess_glh_water.py   # 数据预处理脚本
│
├── work_dir/                      # 训练输出目录
│   ├── UCFSP_GLH-*/              # SPPrompt-Water 训练结果
│   ├── UNET_GLH-*/               # U-Net 训练结果 (新生成)
│   └── SWIN_GLH-*/               # SwinTransformer 训练结果 (新生成)
│
├── output/                        # 测试结果输出
│   ├── test_patches/              # Patch 级别分割结果
│   │   ├── UNET/                 # U-Net 分割图
│   │   └── SWIN/                 # SwinTransformer 分割图
│   └── test_full_images/          # 全图分割结果 (论文对比图)
│
├── RSDataloader.py                # 数据加载器 (PromptDataset_GID5)
│
├── train_promptwaternet.py       # 【原论文】SPPrompt-Water 训练脚本
│
├── train_baseline.py             # 【新增】U-Net / SwinTransformer 训练脚本
├── test_baseline_patch.py        # 【新增】Patch 级别测试脚本
├── test_baseline_full_image.py   # 【新增】全图滑动窗口测试脚本 (生成论文对比图)
├── quick_eval.py                 # 【新增】快速评估脚本 (小样本验证)
│
├── PROJECT_STRUCTURE.md          # 项目结构说明 (Python注释版)
└── README.md                     # 原项目 README
```

---

## 二、各模型对应文件

| 模型 | 模型定义文件 | 训练脚本 | 测试脚本 |
|------|------------|---------|---------|
| **SPPrompt-Water** (论文主模型) | `SPP_model/modeling/prompt_water_net.py` | `train_promptwaternet.py` | `inference/inference_waterprompt.py` |
| **U-Net** (Baseline) | `SPP_model/modeling/other_model/Unet.py` | `train_baseline.py --model_type unet` | `test_baseline_patch.py` / `test_baseline_full_image.py` |
| **SwinTransformer** (Baseline) | `SPP_model/modeling/swin_net.py` | `train_baseline.py --model_type swin` | `test_baseline_patch.py` / `test_baseline_full_image.py` |
| **MECNet** | `SPP_model/modeling/other_model/MECNet.py` | - | - |
| **MSResNet** | `SPP_model/modeling/other_model/MSResNet.py` | - | - |
| **QTNUnet** | `SPP_model/modeling/other_model/QTNUnet/` | - | - |

---

## 三、使用流程

### 1. 训练 U-Net
```bash
cd /root/autodl-tmp/SPPrompt-Water-master
python train_baseline.py \
  --model_type unet \
  --num_epochs 50 \
  --batch_size 4 \
  --device cuda:0 \
  --work_dir ./work_dir \
  --num_workers 8
```

### 2. 训练 SwinTransformer
```bash
python train_baseline.py \
  --model_type swin \
  --num_epochs 50 \
  --batch_size 4 \
  --device cuda:0 \
  --work_dir ./work_dir \
  --num_workers 8
```

### 3. Patch 级别测试 (快速指标)
```bash
# U-Net
python test_baseline_patch.py \
  --model_type unet \
  --resume work_dir/UNET_GLH-XXXX/best_model_eXX_ValscoreX.XXXX.pth

# SwinTransformer
python test_baseline_patch.py \
  --model_type swin \
  --resume work_dir/SWIN_GLH-XXXX/best_model_eXX_ValscoreX.XXXX.pth
```

### 4. 全图测试 (生成论文对比图)
```bash
# U-Net
python test_baseline_full_image.py \
  --model_type unet \
  --resume work_dir/UNET_GLH-XXXX/best_model_eXX_ValscoreX.XXXX.pth \
  --output_dir ./output/test_full_images

# SwinTransformer
python test_baseline_full_image.py \
  --model_type swin \
  --resume work_dir/SWIN_GLH-XXXX/best_model_eXX_ValscoreX.XXXX.pth \
  --output_dir ./output/test_full_images
```

### 5. 快速评估 (训练过程中验证)
```bash
python quick_eval.py \
  --model_type unet \
  --resume work_dir/UNET_GLH-XXXX/checkpoint_eXX.pth \
  --num_samples 100
```

---

## 四、论文对比实验说明

根据论文 **Section IV.C Comparative Experiments**，对比实验的设置如下：

1. **训练方式**: 所有 baseline 方法 (U-Net, SwinTransformer 等) 采用**滑动窗口策略**进行训练和推理
2. **数据**: 使用相同的 1024×1024 patches 训练
3. **推理**: 全图测试时采用滑动窗口 (window=1024, stride=1024) 拼接得到完整分割图
4. **指标**: OA (Overall Accuracy), mIoU, F1-score
5. **可视化**: Fig. 8 (GID), Fig. 9 (CWBSD), Fig. 10/11 (GLH-Water) 展示了各方法的对比结果

论文中 GLH-Water 数据集上的对比结果 (Table III):
- U-Net: OA ~96.68%, mIoU ~71.29%, F1 ~75.81%
- SwinTransformer: OA ~97.83%, mIoU ~75.15%, F1 ~78.97%
- SPPrompt-Water (ours): OA ~98.52%, mIoU ~86.91%, F1 ~88.45%

---

## 五、预训练权重

| 权重文件 | 来源 | 用途 |
|---------|------|------|
| `pretrain/swin_tiny_patch4_window7_224.pth` | [SwinTransformer GitHub](https://github.com/microsoft/Swin-Transformer) | SwinTransformer backbone 初始化 |
| `pretrain/sam_vit_b_01ec64.pth` | SAM (Segment Anything) | SPPrompt-Water 的 prompt module |

> 注意: U-Net 是从头训练的，不需要预训练权重。

---

## 六、常见问题

1. **num_workers=0**: 如果多进程数据加载报错，设置 `--num_workers 0`
2. **OOM**: 如果显存不足，减小 `--batch_size` (如改为 2)
3. **CUBLAS error**: 通常是显存碎片问题，重启进程即可解决
4. **SwinTransformer 训练失败**: 确保 `pretrain/swin_tiny_patch4_window7_224.pth` 存在

---

## 七、输出文件说明

训练完成后，每个模型会在 `work_dir/` 下生成目录，包含：
- `best_model_eXX_ValscoreX.XXXX.pth` - 最佳模型权重
- `checkpoint_eXX.pth` - 每10 epoch 的检查点
- `*_loss_curve.png` - 训练/验证损失曲线
- `*_training.log` - 完整训练日志

测试完成后，会在 `output/` 下生成：
- `test_patches/UNET/` 或 `SWIN/` - patch 级别分割图
- `test_full_images/UNET_xxx_pred.png` - 全图分割结果 (论文用)
- `*_test_summary.txt` - 指标汇总文件
