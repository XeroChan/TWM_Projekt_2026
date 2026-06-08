import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.datasets import RawClassifierDataset, BuildingDataset
from src.unet_model import UNet
from src.classifier_model import CityVillageClassifier
from src.utils import dice_coeff, iou_coeff, calculate_accuracy


def plot_metrics(history, title, save_path):
    """Pomocnicza funkcja do rysowania wykresów strat i metryk."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Wykres funkcji straty
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title(f'{title} - Funkcja Straty (Loss)')
    plt.xlabel('Epoki')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Wykres wybranej metryki (np. Accuracy lub IoU)
    plt.subplot(1, 2, 2)
    if 'train_acc' in history:
        plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
        plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
        plt.title(f'{title} - Dokładność (Accuracy)')
        plt.ylabel('Accuracy')
    elif 'train_iou' in history:
        plt.plot(epochs, history['train_iou'], 'b-', label='Train IoU')
        plt.plot(epochs, history['val_iou'], 'r-', label='Val IoU')
        plt.title(f'{title} - Metryka (IoU)')
        plt.ylabel('IoU')
        
    plt.xlabel('Epoki')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Wykres zapisany w: {save_path}")


def train_classifier(raw_dir="data/raw", batch_size=8, epochs=15,
                     save_path="models/classifier_weights.pth", val_split=0.15, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trening klasyfikatora na urządzeniu: {device}")

    dataset = RawClassifierDataset(os.path.join(raw_dir, "images"))
    
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Podział danych: {train_size} treningowych, {val_size} walidacyjnych")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CityVillageClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # --- trening ---
        model.train()
        train_loss = train_acc = 0.0
        loop = tqdm(train_loader, leave=False, desc=f"Klasyfikator Epoka [{epoch+1}/{epochs}]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device).float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
                
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, labels)
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # --- walidacja ---
        model.eval()
        val_loss = val_acc = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                    
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, labels)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Early Stopping & Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Najlepszy model zapisany (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping po {patience} epokach braku poprawy.")
                break

    print("Trening klasyfikatora zakończony!")
    
    # Generowanie wykresów
    plot_path = save_path.replace(".pth", "_metrics.png")
    plot_metrics(history, title="Klasyfikator (Miasto vs Wieś)", save_path=plot_path)


def train_model(processed_dir="data/processed", batch_size=8, epochs=15,
                save_path="models/unet_weights.pth", val_split=0.15, patience=5):
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
    pos_weight = torch.tensor([10.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': [], 'train_iou': [], 'val_iou': []}

    for epoch in range(epochs):
        # --- trening ---
        model.train()
        train_loss = train_dice = train_iou = 0.0
        loop = tqdm(train_loader, leave=False, desc=f"U-Net Epoka [{epoch+1}/{epochs}]")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            d_coeff = dice_coeff(probs, masks)
            loss = criterion(outputs, masks) + (1.0 - d_coeff)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += d_coeff.item()
            train_iou += iou_coeff(probs, masks).item()
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)

        # --- walidacja ---
        model.eval()
        val_loss = val_dice = val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                
                d_coeff = dice_coeff(probs, masks)
                loss = criterion(outputs, masks) + (1.0 - d_coeff)
                
                val_loss += loss.item()
                val_dice += d_coeff.item()
                val_iou += iou_coeff(probs, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} IoU: {train_iou:.4f} | Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f}")

        # Early Stopping na podstawie najniższego Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Najlepszy model zapisany (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping po {patience} epokach braku poprawy.")
                break

    print(f"Trening U-Net zakończony. Najlepszy Val Loss: {best_val_loss:.4f}")
    
    # Generowanie wykresów
    plot_path = save_path.replace(".pth", "_metrics.png")
    plot_metrics(history, title="U-Net (Segmentacja)", save_path=plot_path)