import os
import random
import cv2
import numpy as np
import torch
import rasterio
import plotly.express as px

from src.unet_model import UNet
from src.utils import image_to_tensor, load_model

# Fixed palette for consistent bbox colors (cycles if more buildings than colors)
_BBOX_PALETTE = [
    (0,   255,  0),    # green
    (255, 165,  0),    # orange
    (0,   200, 255),   # cyan
    (255,  50, 255),   # magenta
    (255, 255,  0),    # yellow
    (50,  150, 255),   # blue
    (255,  80,  80),   # red
    (180, 255, 100),   # lime
]


def interactive_instance_viewer(original_img_path: str, prediction_mask_path: str):
    img = cv2.imread(original_img_path)
    mask = cv2.imread(prediction_mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print("Błąd: Nie można wczytać obrazu lub maski!")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 10:
            continue

        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        cv2.drawContours(img_rgb, [cnt], -1, color, thickness=1)

        x, y, w, h = cv2.boundingRect(cnt)
        label = f"Budynek {i + 1}"
        font_scale, font_thickness = 0.4, 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        padding = 5

        if y > text_h + padding * 2:
            bg_y1 = y - text_h - padding * 2
            bg_y2 = y - padding
            text_y = y - padding - baseline
        else:
            bg_y1 = y + h + padding
            bg_y2 = y + h + text_h + padding * 2
            text_y = bg_y2 - padding - baseline

        cv2.rectangle(img_rgb, (x, bg_y1), (x + text_w, bg_y2), (0, 0, 0), thickness=cv2.FILLED)
        cv2.putText(img_rgb, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, font_thickness)

    fig = px.imshow(img_rgb, title="Detekcja Instancji (Tylko kontury)")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(dragmode="pan", margin=dict(l=0, r=0, b=0, t=30))
    fig.show()


def bbox_detection_viewer(image_path: str, mask: np.ndarray, min_area: int = 30,
                          title: str = "Detekcja Budynków", save_path: str | None = None):
    """
    Military recon style bounding box visualization.
    Each detected building gets a colored rectangle + label with ID and area.
    Shows result in interactive Plotly window and optionally saves to file.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Błąd: Nie można wczytać obrazu: {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    building_idx = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        building_idx += 1
        color = _BBOX_PALETTE[(building_idx - 1) % len(_BBOX_PALETTE)]

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # bounding box
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, thickness=2)

        # label background + text
        label = f"#{building_idx}  {area}px"
        font_scale, font_thickness = 0.45, 1
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        pad = 3

        label_y = y - pad if y - th - pad * 2 >= 0 else y + h + th + pad
        bg_y1 = label_y - th - pad
        bg_y2 = label_y + pad

        cv2.rectangle(img_rgb, (x, bg_y1), (x + tw + pad, bg_y2), color, thickness=cv2.FILLED)
        cv2.putText(img_rgb, label, (x + pad, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"Zapisano wizualizację: {save_path}")

    fig = px.imshow(img_rgb, title=f"{title}  —  wykryto: {building_idx} budynków")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(dragmode="pan", margin=dict(l=0, r=0, b=0, t=40))
    fig.show()


def build_clean_mask(pred_mask: np.ndarray, threshold: float = 0.5,
                     min_area: int = 30, kernel_size: int = 3) -> tuple[np.ndarray, int]:
    """Return (cleaned binary mask, building count)."""
    binary = (pred_mask >= threshold).astype(np.uint8) * 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
            count += 1

    return cleaned, count


def predict_and_count(image_path: str, model_path: str = "models/unet_weights.pth",
                      threshold: float = 0.3, min_area: int = 30,
                      output_dir: str = "data/predictions", visualize: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        print("Błąd: Nie znaleziono wytrenowanego modelu. Najpierw uruchom trening.")
        return

    model = load_model(UNet(in_channels=3, out_channels=1), model_path, device)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Błąd: Nie można wczytać obrazu: {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = image_to_tensor(img_rgb).to(device)

    with torch.no_grad():
        pred_mask = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()

    binary_mask, building_count = build_clean_mask(pred_mask, threshold=threshold, min_area=min_area)
    print(f"Znaleziono {building_count} budynków na zdjęciu.")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"prediction_{os.path.basename(image_path)}")
    viz_path = os.path.join(output_dir, f"viz_{os.path.splitext(os.path.basename(image_path))[0]}.png")

    try:
        with rasterio.open(image_path) as src:
            profile = src.profile
            profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(binary_mask, 1)
        print(f"Zapisano maskę predykcyjną (GeoTIFF) jako {out_path}")
    except Exception:
        png_path = out_path.rsplit('.', 1)[0] + '.png'
        cv2.imwrite(png_path, binary_mask)
        print(f"Zapisano wizualizację maski predykcyjnej jako {png_path} (bez georeferencji)")

    if visualize:
        bbox_detection_viewer(image_path, binary_mask, min_area=min_area,
                              save_path=viz_path)
