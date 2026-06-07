import torch
import cv2
import numpy as np
import os
from src.unet_model import UNet

def build_clean_mask(pred_mask, threshold=0.6, min_area=30):
    # 1. Zwiększamy próg pewności (z 0.5 na 0.6) - bierzemy tylko te piksele,
    # których model jest bardziej pewien. To pomaga odciąć "szum" wokół domów.
    binary_mask = (pred_mask >= threshold).astype(np.uint8) * 255

    # 2. Agresywniejsze otwieranie (rozrywanie cienkich pikselowych mostków między domami)
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Szukanie konturów po wyczyszczeniu
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(binary_mask)

    # 4. Inteligentne wymuszanie prostokątów
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            # Tworzymy potencjalny idealny prostokąt
            rect = cv2.minAreaRect(cnt)
            
            # Pobieramy fizyczne wymiary prostokąta z obiektu rect
            # (rect[1][0] to szerokość, rect[1][1] to wysokość)
            width, height = rect[1]
            box_area = width * height
            
            if box_area == 0:
                continue
                
            # OBLICZENIE "EXTENT" - jaki % prostokąta stanowi prawdziwa plama?
            extent = area / box_area

            # Jeśli plama wypełnia ponad 55% prostokąta -> to stabilny blok/budynek
            if extent > 0.55:
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(cleaned_mask, [box], 0, 255, thickness=cv2.FILLED)
            
            # W przeciwnym razie (Extent <= 55%) -> to połączone budynki lub litera "L"
            # Odrzucamy gigantyczny prostokąt i używamy kanciastego wielokąta
            else:
                epsilon = 0.03 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.drawContours(cleaned_mask, [approx], 0, 255, thickness=cv2.FILLED)

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
    cv2.imwrite(out_path, binary_mask)
    print(f"Zapisano wizualizację maski predykcyjnej jako {out_path}")