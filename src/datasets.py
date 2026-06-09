import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset


class RawClassifierDataset(Dataset):
    def __init__(self, img_dir: str):
        self.img_dir = img_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith('_img.tif')]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        if img is None:
            raise FileNotFoundError(f"Nie można wczytać: {img_name}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        label = 1.0 if "miasto_" in img_name else 0.0
        return torch.tensor(img), torch.tensor([label], dtype=torch.float32)


class BuildingDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, augment: bool = True,
                 file_list: list | None = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.images = file_list if file_list is not None else [
            f for f in os.listdir(img_dir) if f.endswith('.png')
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        mask = cv2.imread(os.path.join(self.mask_dir, img_name), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"Nie można wczytać: {img_name}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment:
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            k = random.choice([0, 1, 2, 3])
            if k:
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
            if random.random() > 0.6:
                factor = 0.8 + random.random() * 0.4
                img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0).astype(np.float32) / 255.0
        return torch.tensor(img), torch.tensor(mask)
