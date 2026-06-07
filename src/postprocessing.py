import torch
import cv2
import numpy as np
import os
from src.unet_model import UNet
import rasterio
import plotly.express as px
import random

def interactive_instance_viewer(original_img_path, prediction_mask_path):
    img = cv2.imread(original_img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    mask = cv2.imread(prediction_mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print("Błąd: Nie można wczytać obrazu lub maski!")
        return

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 10:
            continue

        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        
        # --- Rysujemy TYLKO kontur (grubość 1, żeby był subtelniejszy) ---
        cv2.drawContours(img_rgb, [cnt], -1, color, thickness=1)

        # Pobieramy kordynaty, żeby wiedzieć, gdzie "wisi" najwyższy punkt dachu (y)
        x, y, w, h = cv2.boundingRect(cnt)

        label = f"Budynek {i + 1}"
        font_scale = 0.4
        font_thickness = 1
        
        # Pobieramy dokładne wymiary tekstu
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Margines, żeby odkleić napis od dachu (np. 5 pikseli)
        padding = 5

        # Obliczanie pozycji napisu
        if y > text_h + padding * 2:
            # Jest miejsce NAD budynkiem
            bg_y1 = y - text_h - padding * 2
            bg_y2 = y - padding
            text_y = y - padding - baseline
        else:
            # Budynek jest przyklejony do górnej krawędzi zdjęcia - dajemy napis POD
            bg_y1 = y + h + padding
            bg_y2 = y + h + text_h + padding * 2
            text_y = bg_y2 - padding - baseline

        # Rysowanie tła oraz tekstu
        cv2.rectangle(img_rgb, (x, bg_y1), (x + text_w, bg_y2), (0, 0, 0), thickness=cv2.FILLED)
        cv2.putText(img_rgb, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    # Wyświetlanie w Plotly
    fig = px.imshow(img_rgb, title="Detekcja Instancji (Tylko kontury)")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(dragmode="pan", margin=dict(l=0, r=0, b=0, t=30))
    fig.show()

def build_clean_mask(pred_mask, threshold=0.5, min_area=30, kernel_size=3):
    binary_mask = (pred_mask >= threshold).astype(np.uint8) * 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 255

    return cleaned_mask

def predict_and_count(image_path, model_path="models/unet_weights.pth", threshold=0.3, min_area=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print("Błąd: Nie znaleziono wytrenowanego modelu. Najpierw uruchom trening.")
        return

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_tensor = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_logits = model(img_tensor)
        pred_mask = torch.sigmoid(pred_logits).squeeze().cpu().numpy()

    binary_mask = build_clean_mask(pred_mask, threshold=threshold, min_area=min_area)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    building_count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            building_count += 1

    print(f"Znaleziono {building_count} budynków na zdjęciu.")
    
    out_path = f"prediction_{os.path.basename(image_path)}"
    
    # Zapisz maskę predykcyjną jako GeoTIFF, zachowując georeferencję, jeśli to możliwe
    try:
        with rasterio.open(image_path) as src:
            profile = src.profile
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw'
            )
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(binary_mask, 1)
        print(f"Zapisano maskę predykcyjną (GeoTIFF) jako {out_path}")
    except Exception as e:
        # Jeśli wystąpi błąd (np. brak georeferencji), zapisz maskę jako zwykły obraz PNG
        cv2.imwrite(out_path, binary_mask)
        print(f"Zapisano wizualizację maski predykcyjnej jako {out_path} (bez georeferencji)")