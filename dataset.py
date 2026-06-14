"""Image preprocessing, multi-label parsing, and the PyTorch Dataset.

These helpers are shared by extract_features.py (bulk) and app.py (one image).
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

import config


def build_transform():
    """Resize + normalize an image exactly the way the ImageNet backbone expects."""
    return T.Compose([
        T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def labels_to_vector(finding_labels: str) -> np.ndarray:
    """Convert 'Cardiomegaly|Effusion' -> a 14-dim multi-hot vector.

    'No Finding' (or anything unrecognised) -> all zeros.
    """
    vec = np.zeros(config.NUM_CLASSES, dtype=np.float32)
    if not isinstance(finding_labels, str):
        return vec
    for label in finding_labels.split("|"):
        label = label.strip()
        if label in config.DISEASE_LABELS:
            vec[config.DISEASE_LABELS.index(label)] = 1.0
    return vec


class ChestXrayDataset(Dataset):
    """Returns (image_tensor, label_vector) pairs.

    X-rays are grayscale; .convert("RGB") replicates the single channel to 3 so
    it matches the 3-channel input the pretrained backbone was trained on.
    """

    def __init__(self, paths, labels, transform=None):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.transform = transform or build_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, torch.from_numpy(self.labels[idx])
