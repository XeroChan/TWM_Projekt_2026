import sys
from src.postprocessing import interactive_instance_viewer

DEFAULT_IMG = "data/processed/images/024_wies_rataje_3_256_0.png"
DEFAULT_MASK = "data/predictions/prediction_024_wies_rataje_3_256_0.png"

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    mask = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MASK
    interactive_instance_viewer(img, mask)
