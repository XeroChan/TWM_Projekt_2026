import os
import time
import requests
import numpy as np
import pyproj
import leafmap
from PIL import Image
from io import BytesIO

def fetch_all_data(center_points, offset_meters=300, size=1024):
    img_dir = "data/raw/images"
    mask_dir = "data/raw/masks"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)

    wms_ortho = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"
    wms_kieg = "https://integracja.gugik.gov.pl/cgi-bin/KrajowaIntegracjaEwidencjiGruntow"

    for point in center_points:
        name = point["name"]
        lon, lat = point["pos"]
        
        cx, cy = transformer.transform(lon, lat)
        min_x = cx - offset_meters
        min_y = cy - offset_meters
        max_x = cx + offset_meters
        max_y = cy + offset_meters
        
        bbox_2180 = [min_x, min_y, max_x, max_y]
        bbox_str = f"{min_x},{min_y},{max_x},{max_y}"

        img_path = os.path.join(img_dir, f"{name}_img.tif")
        mask_path = os.path.join(mask_dir, f"{name}_mask.tif")

        downloaded_anything = False

        if not os.path.exists(img_path):
            print(f"[{name}] Pobieranie ortofotomapy...")
            try:
                leafmap.wms_to_geotiff(
                    url=wms_ortho, 
                    bbox=bbox_2180, 
                    layers="Raster",
                    output=img_path, 
                    width=size, 
                    height=size, 
                    CRS="EPSG:2180"
                )
                downloaded_anything = True
            except Exception as e:
                print(f"[{name}] Błąd ortofotomapy: {e}")
        else:
            print(f"[{name}] Zdjęcie gotowe. Pomijam.")

        if not os.path.exists(mask_path):
            print(f"[{name}] Pobieranie masek budynków...")
            params = {
                "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
                "LAYERS": "budynki", "SRS": "EPSG:2180",
                "BBOX": bbox_str,
                "WIDTH": size, "HEIGHT": size, "FORMAT": "image/png", "TRANSPARENT": "TRUE"
            }

            try:
                response = requests.get(wms_kieg, params=params, timeout=60)
                if b"ServiceException" not in response.content:
                    img_rgba = Image.open(BytesIO(response.content)).convert('RGBA')
                    alpha = np.array(img_rgba)[:, :, 3]
                    binary_mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
                    Image.fromarray(binary_mask).save(mask_path)
                    downloaded_anything = True
                else:
                    print(f"[{name}] Serwer masek odrzucił zapytanie.")
            except Exception as e:
                print(f"[{name}] Błąd maski: {e}")
        else:
            print(f"[{name}] Maska gotowa. Pomijam.")

        if downloaded_anything:
            time.sleep(2)
