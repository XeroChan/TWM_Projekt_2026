import os
import sys
import math
import random
import cv2
import torch
import matplotlib.pyplot as plt

from src.unet_model import UNet
from src.utils import image_to_tensor, load_model
from src.postprocessing import build_clean_mask, draw_instances

IMG_DIR = "data/processed/images"
FITTED_DIR = "data/fitted/images"      # kafle użyte do treningu
MODEL_PATH = "models/unet_weights.pth"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12

    if not os.path.isdir(IMG_DIR):
        sys.exit(f"Brak {IMG_DIR}.")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Brak modelu: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(UNet(in_channels=3, out_channels=1), MODEL_PATH, device)

    # tylko kafle spoza fitted (zbiór walidacyjny)
    trained = set(os.listdir(FITTED_DIR)) if os.path.isdir(FITTED_DIR) else set()
    files = []
    for f in os.listdir(IMG_DIR):
        if f.endswith(".png") and f not in trained:
            files.append(f)
    random.seed(4)
    random.shuffle(files)
    files = files[:n]
    print(f"Kafli walidacyjnych (spoza fitted): {len(files)} | urządzenie: {device}")

    cols = min(4, len(files))
    rows = math.ceil(len(files) / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, name in enumerate(files, start=1):
        img = cv2.imread(os.path.join(IMG_DIR, name))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            prob = torch.sigmoid(model(image_to_tensor(img).to(device))).squeeze().cpu().numpy()
        mask, count = build_clean_mask(prob, threshold=0.5, min_area=30)

        plt.subplot(rows, cols, i)
        plt.imshow(draw_instances(img, mask))
        plt.title(f"{count} bud.", fontsize=10)
        plt.axis("off")

    plt.suptitle(f"Predykcje U-Net na danych walidacyjnych — {len(files)} kafli",
                 fontsize=13)
    plt.tight_layout()
    out = "predictions_grid.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Zapisano: {out}")


if __name__ == "__main__":
    main()
