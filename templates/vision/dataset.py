"""Image dataset + transforms. TODO: adapt to your competition's layout."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# TODO: your data layout. Assumes train.csv with <IMAGE_COL>, <LABEL_COL>.
DATA_DIR = Path("data")
IMAGE_COL = "image_path"   # path relative to DATA_DIR
LABEL_COL = "label"
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transforms(img_size=IMG_SIZE):
    # TODO: tune these. btgraham's lesson: preprocessing > architecture.
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def valid_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def tta_transforms(img_size=IMG_SIZE):
    """The views averaged at inference. Cheap LB points — don't skip."""
    base = [transforms.Resize(int(img_size * 1.14)), transforms.CenterCrop(img_size)]
    return [
        transforms.Compose(base + [transforms.ToTensor(),
                                   transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0),
                                    transforms.ToTensor(),
                                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    ]


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(DATA_DIR / row[IMAGE_COL]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = row[LABEL_COL] if LABEL_COL in self.df.columns else -1
        return img, torch.tensor(label, dtype=torch.long)


def load_splits(csv_name="train.csv", valid_frac=0.2, seed=42):
    import numpy as np
    df = pd.read_csv(DATA_DIR / csv_name)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - valid_frac))
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]]
