# Vision: fine-tune a pretrained backbone

**The lesson (Deep Learning Era):** CNNs took vision — Diabetic Retinopathy 2015 (btgraham's preprocessing mattered more than architecture), Data Science Bowl 2017 (3D CNNs), SIIM-ISIC 2020 (diverse EfficientNet ensembles + metadata). The modern version: *never train from scratch.* Start from a pretrained foundation model (ConvNeXt, ViT, DINOv2 via `timm`), fine-tune, and squeeze the last points with test-time augmentation.

**When to reach for it:** any image competition — classification, or backbone for detection/segmentation.

**The pipeline:**
1. **Pretrained backbone** via `timm` (`TODO`: pick yours — `convnext_tiny`, `vit_base_patch16_224`, …).
2. **Augmentations that matter** — train/valid transforms, ImageNet normalization.
3. **Mixed precision** training (free ~2× speed on modern GPUs).
4. **Test-time augmentation (TTA)** — average predictions over flips/crops. The cheapest LB gain in vision.

**Files:**
- `dataset.py` — image dataset + train/valid transforms + TTA views.
- `train.py` — fine-tune loop with AMP, checkpointing, TTA inference.

```bash
# TODO: set DATA_DIR / CSV layout in dataset.py, MODEL_NAME in train.py
python templates/vision/train.py
```
