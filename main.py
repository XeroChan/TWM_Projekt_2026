import argparse
from src.locations import CENTER_POINTS
from src.data_fetcher import fetch_all_data
from src.mask_preparation import prepare_all_masks
from src.preprocessing import tile_images_and_masks
from src.trainer import train_model, train_classifier
from src.postprocessing import predict_and_count

def main():
    parser = argparse.ArgumentParser(description="Detekcja zabudowań (Auto-Labeling Pipeline)")
    parser.add_argument('--step', type=str, choices=['fetch', 'train_classifier', 'prepare_masks', 'preprocess', 'train_unet', 'predict', 'all'], required=True)
    parser.add_argument('--image', type=str, help='Ścieżka do zdjęcia testowego (wymagane dla predict)')
    args = parser.parse_args()

    if args.step in ['fetch', 'all']:
        print(f"\n--- KROK 1: POBIERANIE DANYCH ({len(CENTER_POINTS)} obszarów) ---")
        fetch_all_data(CENTER_POINTS)
        
    if args.step in ['train_classifier', 'all']:
        print("\n--- KROK 2: TRENING KLASYFIKATORA (Bramkarz danych) ---")
        train_classifier(epochs=5)
        
    if args.step in ['prepare_masks', 'all']:
        print("\n--- KROK 3: INTELIGENTNE CZYSZCZENIE MASEK ---")
        prepare_all_masks()
        
    if args.step in ['preprocess', 'all']:
        print("\n--- KROK 4: CIĘCIE NA KAFLE (Z wyzerowanymi miastami) ---")
        tile_images_and_masks()
        
    if args.step in ['train_unet', 'all']:
        print("\n--- KROK 5: TRENING GŁÓWNEGO MODELU U-NET ---")
        train_model(epochs=30)
        
    if args.step == 'predict':
        if not args.image:
            print("Błąd: Podaj ścieżkę do zdjęcia")
            return
        print(f"\n--- PREDYKCJA U-NET DLA: {args.image} ---")
        predict_and_count(args.image)

if __name__ == "__main__":
    main()