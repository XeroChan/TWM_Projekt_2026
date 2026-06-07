import torch
import cv2
import numpy as np
import os
from src.unet_model import UNet
import rasterio
import plotly.express as px
import random

def interactive_instance_viewer(original_img_path, prediction_mask_path):
    # 1. Wczytanie plików
    img = cv2.imread(original_img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    mask = cv2.imread(prediction_mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print("Błąd: Nie można wczytać obrazu lub maski!")
        return

    # 2. Szukanie wszystkich osobnych plam na masce
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Przetwarzanie każdego budynku osobno
    # Zmienna 'i' to numer budynku, 'cnt' to jego kształt
    for i, cnt in enumerate(contours):
        
        # Ignorujemy ewentualne mikroskopijne kropki (szum) o polu mniejszym niż 10 pikseli
        if cv2.contourArea(cnt) < 10:
            continue

        # Losowanie jaskrawego koloru RGB dla każdego budynku (wartości od 50 do 255)
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        
        # --- KROK A: Rysowanie dokładnego obrysu dachu ---
        cv2.drawContours(img_rgb, [cnt], -1, color, thickness=2)

        # --- KROK B: Tworzenie prostokąta (Bounding Box) wokół budynku ---
        # Funkcja boundingRect zwraca współrzędne lewego górnego rogu (x, y) oraz szerokość i wysokość (w, h)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, thickness=1)

        # --- KROK C: Dodawanie etykiety (np. "Budynek 1") ---
        label = f"Budynek {i + 1}"
        
        # Aby tekst był czytelny na każdym tle, rysujemy najpierw mały czarny prostokącik jako tło pod napis
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img_rgb, (x, y - text_h - 4), (x + text_w, y), (0, 0, 0), thickness=cv2.FILLED)
        
        # Nakładanie tekstu w kolorze przypisanym do budynku
        cv2.putText(img_rgb, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness=1)

    # 4. Wyświetlenie interaktywne w Plotly
    fig = px.imshow(img_rgb, title="Detekcja Instancji (Każdy budynek osobno)")
    
    # Kosmetyka okna
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