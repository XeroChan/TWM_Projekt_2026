import os
import cv2
from glob import glob

def center_crop(image, target_size=512):
    """
    Wycina centralny obszar z obrazu/maski.
    """
    h, w = image.shape[:2]
    start_x = (w - target_size) // 2
    start_y = (h - target_size) // 2

    return image[start_y:start_y + target_size, start_x:start_x + target_size]

def tile_images_and_masks(img_raw_dir="data/raw/images", mask_interim_dir="data/interim/masks", 
                          processed_dir="data/processed", target_size=512, tile_size=256):
    
    img_out_dir = os.path.join(processed_dir, "images")
    mask_out_dir = os.path.join(processed_dir, "masks")
    img_cities_out_dir = os.path.join(processed_dir, "images_city")
    mask_cities_out_dir = os.path.join(processed_dir, "masks_city")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(img_cities_out_dir, exist_ok=True)
    os.makedirs(mask_cities_out_dir, exist_ok=True)

    raw_images = glob(os.path.join(img_raw_dir, "*_img.tif"))

    for img_path in raw_images:
        base_name = os.path.basename(img_path).replace("_img.tif", "")
        mask_path = os.path.join(mask_interim_dir, f"{base_name}_mask.tif")

        if not os.path.exists(mask_path):
            print(f"Brak wyczyszczonej maski dla {base_name}, pomijam cięcie.")
            continue

        # Wczytanie obrazu i maski (zwykle 1024x1024)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        # --- TUTAJ WCHODZI TWOJA LOGIKA ---
        # 1. Przycięcie ZDJĘCIA i MASKI do 512x512
        img_cropped = center_crop(img, target_size=target_size)
        mask_cropped = center_crop(mask, target_size=target_size)

        # 2. Cięcie przyciętego obrazu (512x512) na kafle (256x256)
        print(f"Cięcie centralnego obszaru dla: {base_name}...")
        h, w = img_cropped.shape[:2]
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                img_tile = img_cropped[y:y+tile_size, x:x+tile_size]
                mask_tile = mask_cropped[y:y+tile_size, x:x+tile_size]

                if img_tile.shape[0] == tile_size and img_tile.shape[1] == tile_size:
                    # Dodajemy kordynaty do nazwy, żeby kafle się nie nadpisały
                    tile_name = f"{base_name}_{y}_{x}.png"
                    if base_name.lower().startswith("miasto"):
                        cv2.imwrite(os.path.join(img_cities_out_dir, tile_name), img_tile)
                        cv2.imwrite(os.path.join(mask_cities_out_dir, tile_name), mask_tile)
                    else:
                        cv2.imwrite(os.path.join(img_out_dir, tile_name), img_tile)
                        cv2.imwrite(os.path.join(mask_out_dir, tile_name), mask_tile)
                    
    print("Pre-processing (Center Crop -> Tiling) zakończony!")