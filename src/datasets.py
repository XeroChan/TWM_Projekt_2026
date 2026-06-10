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
        self.images = []
        for f in os.listdir(img_dir):
            if f.endswith('_img.tif'):
                self.images.append(f)

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
                 file_list: list | None = None, ext: str = '.png',
                 mask_name_fn=None, cache: bool = False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        # mapowanie nazwy obrazu -> nazwy maski (domyślnie ta sama nazwa)
        self.mask_name_fn = mask_name_fn or (lambda name: name)
        if file_list is not None:
            self.images = file_list
        else:
            self.images = []
            for f in os.listdir(img_dir):
                if f.endswith(ext):
                    self.images.append(f)
        self.cache = None
        if cache:
            print(f"Cache RAM: wczytuję {len(self.images)} kafli...")
            self.cache = []
            for i in range(len(self.images)):
                self.cache.append(self._read(i))

    def __len__(self) -> int:
        return len(self.images)

    def _read(self, idx: int):
        img_name = self.images[idx]
        mask_name = self.mask_name_fn(img_name)
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"Nie można wczytać: {img_name}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask

    def __getitem__(self, idx: int):
        if self.cache is not None:
            img, mask = self.cache[idx]
            img, mask = img.copy(), mask.copy()
        else:
            img, mask = self._read(idx)

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
