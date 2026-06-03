import os
import cv2
import numpy as np
import torch
from glob import glob
from src.classifier_model import CityVillageClassifier

def prepare_all_masks(input_mask_dir="data/raw/masks", input_img_dir="data/raw/images", 
                      output_dir="data/interim/masks", classifier_path="models/classifier_weights.pth", 
                      min_area=20):
    
    os.makedirs(output_dir, exist_ok=True)
    mask_files = glob(os.path.join(input_mask_dir, "*_mask.tif"))
    
    if not mask_files:
        print("Brak surowych masek do przetworzenia.")
        return

    # 1. Wczytanie wytrenowanego klasyfikatora
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = CityVillageClassifier().to(device)
    
    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        print("Pomyślnie załadowano sztuczną inteligencję (Klasyfikator) do oceny zdjęć.")
    else:
        print("BŁĄD: Nie znaleziono klasyfikatora! Najpierw uruchom: python main.py --step train_classifier")
        return

    # 2. Przetwarzanie masek
    for mask_path in mask_files:
        filename = os.path.basename(mask_path)
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
            continue

        # Wczytujemy zdjęcie i maskę
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img_filename = filename.replace("_mask.tif", "_img.tif")
        img_path = os.path.join(input_img_dir, img_filename)
        img = cv2.imread(img_path)

        if mask_img is None or img is None:
            continue

        # --- PREDYKCJA KLASYFIKATOREM (Ocenianie zdjęcia) ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256)) 
        img_tensor = img_resized.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            class_logits = classifier(img_tensor)
            class_prob = torch.sigmoid(class_logits).item()

        # --- DECYZJA OPARTA NA WYNIKU MODELU ---
        if class_prob > 0.5:
            # Model uważa, że to MIASTO
            print(f"[{class_prob*100:.1f}% MIASTO] Klasyfikator decyduje: Zerowanie maski -> {filename}")
            black_mask = np.zeros_like(mask_img)
            cv2.imwrite(output_path, black_mask)
        else:
            # Model uważa, że to WIEŚ
            print(f"[{100 - class_prob*100:.1f}% WIEŚ] Klasyfikator decyduje: Czyszczenie maski -> {filename}")
            _, thresh = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros(mask_img.shape, dtype=np.uint8)
            for cnt in contours:
                if cv2.contourArea(cnt) >= min_area:
                    cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
            cv2.imwrite(output_path, filled)