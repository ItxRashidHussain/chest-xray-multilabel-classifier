"""STEP 2 - Run the frozen backbone over every image ONCE and cache the vectors.

Run:  python extract_features.py

This is the only slow step on CPU (a single forward pass per image), and you
only ever run it once. Afterwards, training is tiny and fast because it works
on the saved feature vectors instead of the raw images.

Note: the first run downloads the pretrained backbone weights (~14 MB for
mobilenet_v2), so you need an internet connection that one time.
"""
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import ChestXrayDataset
from model import build_backbone


def extract_split(backbone, df_split, device):
    ds = ChestXrayDataset(df_split["path"].tolist(),
                          df_split[config.DISEASE_LABELS].values)
    loader = DataLoader(ds, batch_size=config.EXTRACT_BATCH_SIZE,
                        shuffle=False, num_workers=0)  # num_workers=0 is safest on Windows
    feats, labels = [], []
    with torch.no_grad():
        for imgs, ys in tqdm(loader, desc="  extracting", leave=False):
            feats.append(backbone(imgs.to(device)).cpu().numpy())
            labels.append(ys.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def main():
    if not config.SPLITS_CSV.exists():
        sys.exit("Run prepare_data.py first.")
    df = pd.read_csv(config.SPLITS_CSV)
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    backbone, feat_dim = build_backbone(config.BACKBONE)
    backbone.to(device)
    print(f"Backbone: {config.BACKBONE}  (feature dim = {feat_dim})")

    for split in ["train", "val", "test"]:
        df_split = df[df["split"] == split].reset_index(drop=True)
        print(f"[{split}] {len(df_split)} images")
        X, y = extract_split(backbone, df_split, device)
        np.save(config.FEATURES_DIR / f"X_{split}.npy", X)
        np.save(config.FEATURES_DIR / f"y_{split}.npy", y)
        print(f"   -> X {X.shape}, y {y.shape}")

    (config.FEATURES_DIR / "backbone.txt").write_text(config.BACKBONE)
    print("\nDone. Next: python train.py")


if __name__ == "__main__":
    main()
