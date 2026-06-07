import os
import time
import requests
import numpy as np
import pyproj
import leafmap
from PIL import Image
from io import BytesIO

def fetch_all_data(center_points, offset=0.0030, size=1024):
    img_dir = "data/raw/images"
    mask_dir = "data/raw/masks"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)

    for point in center_points:
        name = point["name"]
        lon, lat = point["pos"]
        bbox = [lon - offset, lat - offset, lon + offset, lat + offset]
        
        img_path = os.path.join(img_dir, f"{name}_img.tif")
        mask_path = os.path.join(mask_dir, f"{name}_mask.tif")

        downloaded = False

        if not os.path.exists(img_path):
            print(f"[{name}] Pobieranie ortofotomapy...")
            wms_ortho = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"
            leafmap.wms_to_geotiff(
                url=wms_ortho, bbox=bbox, layers="Raster",
                output=img_path, width=size, height=size, CRS="EPSG:4326"
            )
            downloaded = True

        if not os.path.exists(mask_path):
            print(f"[{name}] Pobieranie masek budynków...")
            min_x, min_y = transformer.transform(bbox[0], bbox[1])
            max_x, max_y = transformer.transform(bbox[2], bbox[3])
            
            wms_kieg = "https://integracja.gugik.gov.pl/cgi-bin/KrajowaIntegracjaEwidencjiGruntow"
            params = {
                "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
                "LAYERS": "budynki", "SRS": "EPSG:2180",
                "BBOX": f"{min_x},{min_y},{max_x},{max_y}",
                "WIDTH": size, "HEIGHT": size, "FORMAT": "image/png", "TRANSPARENT": "TRUE"
            }

            try:
                response = requests.get(wms_kieg, params=params, timeout=60)
                if b"ServiceException" not in response.content:
                    img_rgba = Image.open(BytesIO(response.content)).convert('RGBA')
                    alpha = np.array(img_rgba)[:, :, 3]
                    binary_mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
                    Image.fromarray(binary_mask).save(mask_path)
                    downloaded = True
            except Exception as e:
                print(f"[{name}] Błąd: {e}")

        if downloaded:
            time.sleep(10)