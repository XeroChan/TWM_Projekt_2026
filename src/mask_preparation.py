import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

INPUT_DIR = "../data/raw/masks"
OUTPUT_DIR = "../data/processed/masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# obsługiwane rozszerzenia
EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

for filename in os.listdir(INPUT_DIR):

    if not filename.lower().endswith(EXTENSIONS):
        continue

    input_path = os.path.join(INPUT_DIR, filename)

    print(f"Processing: {filename}")

    # wczytanie grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Cannot load: {filename}")
        continue

    # binaryzacja
    _, thresh = cv2.threshold(
        img,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # domknięcie małych przerw
    kernel = np.ones((3, 3), np.uint8)

    closed = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # znalezienie konturów
    contours, hierarchy = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # pusty obraz wynikowy
    filled = np.zeros(
        img.shape,
        dtype=np.uint8
    )

    # minimalna powierzchnia obiektu
    MIN_AREA = 20

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < MIN_AREA:
            continue

        cv2.drawContours(
            filled,
            [cnt],
            -1,
            255,
            thickness=cv2.FILLED
        )

    # zapis
    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )
    TARGET_SIZE = 512

    h, w = filled.shape[:2]

    start_x = (w - TARGET_SIZE) // 2
    start_y = (h - TARGET_SIZE) // 2

    cropped = filled[
        start_y:start_y + TARGET_SIZE,
        start_x:start_x + TARGET_SIZE
    ]
    
    cv2.imwrite(output_path, cropped)

    print(f"Saved: {output_path}")

print("\nDone!")