"""Fine-tune a pretrained timm backbone: AMP training + TTA inference.

Usage:
    python templates/vision/train.py
TODO: MODEL_NAME, NUM_CLASSES, DATA_DIR (in dataset.py), EPOCHS/LR for your task.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import ImageDataset, load_splits, train_transforms, tta_transforms, valid_transforms

# --- TODO: your knobs ---
MODEL_NAME = "convnext_tiny"   # try: vit_base_patch16_224, tf_efficientnet_b3, ...
NUM_CLASSES = 2                # TODO
EPOCHS = 10
LR = 3e-4
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model():
    # Pretrained backbone, new head. Never train vision from scratch on Kaggle.
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    return model.to(DEVICE)


def train_one_epoch(model, loader, opt, scaler, criterion):
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):  # mixed precision
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


@torch.no_grad()
def predict_tta(model, paths_df, batch_size=128):
    """Average predictions over TTA views. The cheapest reliable LB gain."""
    from dataset import DATA_DIR, IMAGE_COL
    import pandas as pd
    from PIL import Image
    model.eval()
    views = tta_transforms()
    all_probs = []
    imgs_all = [Image.open(DATA_DIR / p).convert("RGB") for p in paths_df[IMAGE_COL]]
    for tfm in views:
        ds = [(tfm(im), -1) for im in imgs_all]
        loader = DataLoader(ds, batch_size=batch_size)
        probs = []
        for batch, _ in loader:
            logits = model(batch.to(DEVICE))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs))
    return np.mean(all_probs, axis=0)  # average over views


def main():
    run_dir = Path("runs") / f"vision_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df_tr, df_va = load_splits()
    train_loader = DataLoader(ImageDataset(df_tr, train_transforms()),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(ImageDataset(df_va, valid_transforms()),
                              batch_size=BATCH_SIZE, num_workers=4)

    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    criterion = nn.CrossEntropyLoss()
    best_auc, best_state = 0.0, None

    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, opt, scaler, criterion)
        # quick valid check on center-crop view
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for imgs, lab in valid_loader:
                p = torch.softmax(model(imgs.to(DEVICE)), dim=1).cpu().numpy()
                probs.append(p)
                labels.append(lab.numpy())
        probs, labels = np.concatenate(probs), np.concatenate(labels)
        auc = roc_auc_score(labels, probs[:, 1]) if NUM_CLASSES == 2 else float("nan")
        print(f"[epoch {epoch}] valid_auc={auc:.4f}")
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save(best_state, run_dir / "best.pt")
    print(f"[done] best valid_auc={best_auc:.4f} -> {run_dir / 'best.pt'}")
    # TODO: run predict_tta() on test.csv for your submission.


if __name__ == "__main__":
    main()
