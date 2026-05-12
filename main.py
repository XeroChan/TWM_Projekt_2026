import leafmap
import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import pyproj
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

BASE_RAW_DIR = os.path.join("data", "raw")

center_points = [
    {"name": "strozewo", "pos": [ 	18.6458107,  	52.57286282]},
    {"name": "krakow_center", "pos": [19.9367, 50.0619]},
    {"name": "warszawa_outskirts", "pos": [21.0394, 52.3706]},
    {"name": "podlasie_village", "pos": [23.1567, 52.8234]},
    {"name": "bieszczady_rural", "pos": [22.6967, 49.2514]},
    {"name": "lubelskie_farms", "pos": [23.1000, 51.2000]}
]

OFFSET = 0.0030
SIZE = 1024

def download_data(name, lon, lat):
    bbox = [lon - OFFSET, lat - OFFSET, lon + OFFSET, lat + OFFSET]
    
    img_path = f"data/raw/{name}_img.tif"
    wms_ortho = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"
    
    print(f"Downloading image for: {name}...")
    leafmap.wms_to_geotiff(
        url=wms_ortho,
        bbox=bbox,
        layers="Raster",
        output=img_path,
        width=SIZE,
        height=SIZE,
        CRS="EPSG:4326"
    )

    mask_path = os.path.join(BASE_RAW_DIR, f"{name}_mask.tif")
    
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    min_x, min_y = transformer.transform(bbox[0], bbox[1])
    max_x, max_y = transformer.transform(bbox[2], bbox[3])

    wms_kieg = "https://integracja.gugik.gov.pl/cgi-bin/KrajowaIntegracjaEwidencjiGruntow"
    
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": "budynki",
        "SRS": "EPSG:2180",
        "BBOX": f"{min_x},{min_y},{max_x},{max_y}",
        "WIDTH": SIZE,
        "HEIGHT": SIZE,
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE"
    }

    try:
        response = requests.get(wms_kieg, params=params, timeout=60)
        if b"ServiceException" in response.content:
            print(f"Server reported an error for mask {name}.")
            return

        img_rgba = Image.open(BytesIO(response.content)).convert('RGBA')
        img_array = np.array(img_rgba)
        
        alpha = img_array[:, :, 3]
        binary_mask = np.where(alpha > 0, 255, 0).astype(np.uint8)

        Image.fromarray(binary_mask).save(mask_path)
        print(f"Saved complete set for: {name}")

    except Exception as e:
        print(f"Error with mask {name}: {e}")

for point in center_points:
    download_data(point["name"], point["pos"][0], point["pos"][1])
    time.sleep(3)
print("\nDone!")