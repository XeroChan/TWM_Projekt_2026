import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import RawClassifierDataset, BuildingDataset
from src.unet_model import UNet
from src.classifier_model import CityVillageClassifier
from src.utils import dice_coeff


def train_classifier(raw_dir="data/raw", batch_size=8, epochs=5,
                     save_path="models/classifier_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trening klasyfikatora na urządzeniu: {device}")

    dataset = RawClassifierDataset(os.path.join(raw_dir, "images"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CityVillageClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        loop = tqdm(loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Klasyfikator Epoka [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), save_path)
    print("Zapisano wagi klasyfikatora!")


def train_model(processed_dir="data/processed", batch_size=8, epochs=10,
                save_path="models/unet_weights.pth", val_split=0.15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = os.path.join(processed_dir, "images")
    mask_dir = os.path.join(processed_dir, "masks")

    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    random.seed(42)
    random.shuffle(all_files)
    val_size = max(1, int(len(all_files) * val_split))
    val_files = all_files[:val_size]
    train_files = all_files[val_size:]

    print(f"Podział danych: {len(train_files)} treningowych, {len(val_files)} walidacyjnych")

    train_loader = DataLoader(
        BuildingDataset(img_dir, mask_dir, augment=True, file_list=train_files),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        BuildingDataset(img_dir, mask_dir, augment=False, file_list=val_files),
        batch_size=batch_size, shuffle=False
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    pos_weight = torch.tensor([10.0]).to(device)  # budynki ~10% pikseli → wyrównaj klasy
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_dice = 0.0

    for epoch in range(epochs):
        # --- trening ---
        model.train()
        loop = tqdm(train_loader, leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            loss = criterion(outputs, masks) + (1.0 - dice_coeff(probs, masks))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item(), dice=dice_coeff(probs, masks).item())

        # --- walidacja ---
        model.eval()
        val_loss = val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                d = dice_coeff(probs, masks)
                val_loss += (criterion(outputs, masks) + (1.0 - d)).item()
                val_dice += d.item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        print(f"  Val loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  -> Najlepszy model zapisany (val dice: {best_val_dice:.4f})")

    print(f"Trening zakończony. Najlepszy val Dice: {best_val_dice:.4f}")
