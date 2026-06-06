import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"  # Wyłączenie logów OpenCV
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.unet_model import UNet
from src.classifier_model import CityVillageClassifier

class RawClassifierDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith('_img.tif')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Klasyfikator potrzebuje mniejszych zdjęć do szybkiej nauki (np. 256x256)
        img = cv2.resize(img, (256, 256)) 
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0

        # Klasyfikator potrzebuje "prawdy" (Ground Truth) do nauki. 
        # Nazwy plików służą TU TYLKO do nauki klasyfikatora.
        label = 1.0 if "miasto_" in img_name else 0.0

        return torch.tensor(img), torch.tensor([label], dtype=torch.float32)

def train_classifier(raw_dir="data/raw", batch_size=8, epochs=5, save_path="models/classifier_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trening klasyfikatora na urządzeniu: {device}")
    
    img_dir = os.path.join(raw_dir, "images")
    dataset = RawClassifierDataset(img_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CityVillageClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        loop = tqdm(loader, leave=True)
        epoch_loss = 0

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_description(f"Klasyfikator Epoka [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), save_path)
    print("Zapisano wagi klasyfikatora!")

class BuildingDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0).astype(np.float32) / 255.0

        return torch.tensor(img), torch.tensor(mask)

def train_model(processed_dir="data/processed", batch_size=8, epochs=10, save_path="models/unet_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_dir = os.path.join(processed_dir, "images")
    mask_dir = os.path.join(processed_dir, "masks")
    
    dataset = BuildingDataset(img_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        loop = tqdm(loader, leave=True)
        epoch_loss = 0

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), save_path)
    print("Zapisano wagi modelu!")