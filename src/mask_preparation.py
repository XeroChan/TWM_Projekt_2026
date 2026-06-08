import os
import cv2
import numpy as np
import torch
from glob import glob

from src.classifier_model import CityVillageClassifier
from src.utils import image_to_tensor, load_model


def prepare_all_masks(input_mask_dir="data/raw/masks", input_img_dir="data/raw/images",
                      output_dir="data/interim/masks",
                      classifier_path="models/classifier_weights.pth",
                      min_area=20):

    os.makedirs(output_dir, exist_ok=True)
    mask_files = glob(os.path.join(input_mask_dir, "*_mask.tif"))

    if not mask_files:
        print("Brak surowych masek do przetworzenia.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(classifier_path):
        print("BŁĄD: Nie znaleziono klasyfikatora! Najpierw uruchom: python main.py --step train_classifier")
        return

    classifier = load_model(CityVillageClassifier(), classifier_path, device)
    print("Pomyślnie załadowano klasyfikator.")

    for mask_path in mask_files:
        filename = os.path.basename(mask_path)
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            continue

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(os.path.join(input_img_dir, filename.replace("_mask.tif", "_img.tif")))

        if mask_img is None or img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = image_to_tensor(img_rgb, size=256).to(device)

        with torch.no_grad():
            class_prob = torch.sigmoid(classifier(img_tensor)).item()

        if class_prob > 0.5:
            print(f"[{class_prob*100:.1f}% MIASTO] Zerowanie maski -> {filename}")
            cv2.imwrite(output_path, np.zeros_like(mask_img))
        else:
            print(f"[{100 - class_prob*100:.1f}% WIEŚ] Czyszczenie maski -> {filename}")
            _, thresh = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros(mask_img.shape, dtype=np.uint8)
            for cnt in contours:
                if cv2.contourArea(cnt) >= min_area:
                    cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
            cv2.imwrite(output_path, filled)
