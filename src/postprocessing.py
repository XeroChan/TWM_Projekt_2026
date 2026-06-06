import torch
import cv2
import numpy as np
import os
from src.unet_model import UNet

def predict_and_count(image_path, model_path="models/unet_weights.pth", tile_size=256, stride=128, threshold=0.3, min_area=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print("Błąd: Nie znaleziono wytrenowanego modelu. Najpierw uruchom trening.")
        return

    # 1. Inicjalizacja modelu
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Wczytanie pełnego obrazu testowego
    img = cv2.imread(image_path)
    h, w, c = img.shape
    
    # Przygotowanie pustych macierzy do akumulacji wyników z nakładających się okien
    compiled_mask = np.zeros((h, w), dtype=np.float32)
    weight_mask = np.zeros((h, w), dtype=np.float32)

    print(f"Uruchamianie Sliding Window Inference dla obrazu {w}x{h}...")

    # 3. Pętla okna przesuwnego (Sliding Window)
    # Przesuwamy się w pionie i poziomie z określonym krokiem (stride)
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            
            # Wycięcie fragmentu (kafla) o rozmiarze treningowym 256x256
            tile = img[y:y+tile_size, x:x+tile_size]
            
            # Normalizacja i transformacja na tensor tak jak podczas treningu
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile_tensor = tile_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
            tile_tensor = torch.tensor(tile_tensor).unsqueeze(0).to(device)
            
            # Predykcja sieci dla pojedynczego okna
            with torch.no_grad():
                pred_logits = model(tile_tensor)
                pred_tile = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
            
            # Dodanie prawdopodobieństw z tego okna do maski zbiorczej
            compiled_mask[y:y+tile_size, x:x+tile_size] += pred_tile
            # Zwiększenie licznika odwiedzin tych pikseli (waga do uśrednienia)
            weight_mask[y:y+tile_size, x:x+tile_size] += 1.0

    # Obsługa potencjalnych pikseli brzegowych, do których okno mogło nie dotrzeć
    weight_mask[weight_mask == 0] = 1.0
    
    # 4. Uśrednienie wyników w miejscach, gdzie okna nakładały się na siebie
    final_prob_mask = compiled_mask / weight_mask

    # 5. Progowe zbinaryzowanie uśrednionej maski
    binary_mask = (final_prob_mask > threshold).astype(np.uint8) * 255
    
    # 6. Post-processing geometryczny (Wymuszenie ostrych prostokątów budynków)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sharp_mask = np.zeros_like(binary_mask)
    building_count = 0
    
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            # Tworzymy zorientowany prostokąt o najmniejszym polu wokół wykrytej plamy
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box) # Pobranie 4 wierzchołków prostokąta
            
            # Rysujemy na wynikowej masce idealny, wypełniony czworokąt zamiast obłej plamy
            cv2.drawContours(sharp_mask, [box], 0, 255, -1)
            building_count += 1

    print(f"Znaleziono {building_count} budynków na zdjęciu.")
    
    out_path = f"sliding_prediction_{os.path.basename(image_path)}"
    cv2.imwrite(out_path, sharp_mask)
    print(f"Zapisano geometryczną maskę predykcyjną jako: {out_path}")
    
    return building_count