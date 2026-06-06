import torch
import cv2
import numpy as np
import os
from src.unet_model import UNet

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

    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    building_count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            building_count += 1

    print(f"Znaleziono {building_count} budynków na zdjęciu.")
    
    out_path = f"prediction_{os.path.basename(image_path)}"
    cv2.imwrite(out_path, binary_mask)
    print(f"Zapisano wizualizację maski predykcyjnej jako {out_path}")