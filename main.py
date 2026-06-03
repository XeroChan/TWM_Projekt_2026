import argparse
from src.data_fetcher import fetch_all_data
from src.mask_preparation import prepare_all_masks
from src.preprocessing import tile_images_and_masks
from src.trainer import train_model, train_classifier # DODANO train_classifier
from src.postprocessing import predict_and_count

CENTER_POINTS = [
    # =========================================================
    # MIASTA (6 miast x 3 zdjęcia = 18 obszarów)
    # =========================================================
    # 1. Warszawa (Śródmieście, Wola, Praga - różne typy gęstej zabudowy)
    {"name": "miasto_waw_1", "pos": [21.0122, 52.2297]},
    {"name": "miasto_waw_2", "pos": [20.9822, 52.2350]},
    {"name": "miasto_waw_3", "pos": [21.0422, 52.2500]},

    # 2. Kraków (Kazimierz, Grzegórzki, Podgórze)
    {"name": "miasto_kra_1", "pos": [19.9450, 50.0510]},
    {"name": "miasto_kra_2", "pos": [19.9650, 50.0600]},
    {"name": "miasto_kra_3", "pos": [19.9400, 50.0350]},

    # 3. Łódź (Piotrkowska, Bałuty, Widzew)
    {"name": "miasto_lod_1", "pos": [19.4560, 51.7600]},
    {"name": "miasto_lod_2", "pos": [19.4460, 51.7800]},
    {"name": "miasto_lod_3", "pos": [19.4960, 51.7500]},

    # 4. Wrocław (Stare Miasto, Nadodrze, Krzyki)
    {"name": "miasto_wro_1", "pos": [17.0322, 51.1100]},
    {"name": "miasto_wro_2", "pos": [17.0222, 51.1250]},
    {"name": "miasto_wro_3", "pos": [17.0122, 51.0850]},

    # 5. Poznań (Rataje - blokowiska, Wilda, Łazarz)
    {"name": "miasto_poz_1", "pos": [16.9550, 52.3850]},
    {"name": "miasto_poz_2", "pos": [16.9250, 52.3900]},
    {"name": "miasto_poz_3", "pos": [16.8950, 52.3950]},

    # 6. Gdańsk (Stocznia/Przemysł, Wrzeszcz, Przymorze)
    {"name": "miasto_gda_1", "pos": [18.6530, 54.3600]},
    {"name": "miasto_gda_2", "pos": [18.6030, 54.3800]},
    {"name": "miasto_gda_3", "pos": [18.5830, 54.4050]},


    # =========================================================
    # WSIE (20 wsi x 3 zdjęcia = 60 obszarów)
    # Zgrupowane według typów architektonicznych w Polsce
    # =========================================================
    
    # --- TYP 1: Wielkopolska / Kujawy (Zabudowa zwarta, rolnicza) ---
    {"name": "wies_strozewo_1",  "pos": [18.6458, 52.5728]},
    {"name": "wies_strozewo_2",  "pos": [18.6558, 52.5728]},
    {"name": "wies_strozewo_3",  "pos": [18.6358, 52.5728]},

    {"name": "wies_rataje_1",    "pos": [16.9150, 52.9800]},
    {"name": "wies_rataje_2",    "pos": [16.9250, 52.9800]},
    {"name": "wies_rataje_3",    "pos": [16.9050, 52.9800]},

    {"name": "wies_pierzchno_1", "pos": [17.1350, 52.2150]},
    {"name": "wies_pierzchno_2", "pos": [17.1450, 52.2150]},
    {"name": "wies_pierzchno_3", "pos": [17.1250, 52.2150]},

    {"name": "wies_osieczna_1",  "pos": [16.6800, 51.8950]},
    {"name": "wies_osieczna_2",  "pos": [16.6900, 51.8950]},
    {"name": "wies_osieczna_3",  "pos": [16.6700, 51.8950]},


    # --- TYP 2: Podlasie (Ulicówki, długie pasy pól, domy przy drodze) ---
    # *Przesunięcie północ/południe, bo wsie ciągną się wzdłuż osi drogi
    {"name": "wies_trzescianka_1", "pos": [23.4560, 52.9370]},
    {"name": "wies_trzescianka_2", "pos": [23.4560, 52.9470]},
    {"name": "wies_trzescianka_3", "pos": [23.4560, 52.9270]},

    {"name": "wies_soce_1",      "pos": [23.4900, 52.9200]},
    {"name": "wies_soce_2",      "pos": [23.4900, 52.9280]},
    {"name": "wies_soce_3",      "pos": [23.4900, 52.9120]},

    {"name": "wies_puchly_1",    "pos": [23.5000, 52.9000]},
    {"name": "wies_puchly_2",    "pos": [23.5000, 52.9080]},
    {"name": "wies_puchly_3",    "pos": [23.5000, 52.8920]},

    {"name": "wies_tokary_1",    "pos": [23.2700, 52.3350]},
    {"name": "wies_tokary_2",    "pos": [23.2700, 52.3430]},
    {"name": "wies_tokary_3",    "pos": [23.2700, 52.3270]},


    # --- TYP 3: Lubelskie (Łańcuchówki wielodrożne) ---
    {"name": "wies_sulow_1",     "pos": [22.9500, 50.7800]},
    {"name": "wies_sulow_2",     "pos": [22.9600, 50.7800]},
    {"name": "wies_sulow_3",     "pos": [22.9400, 50.7800]},

    {"name": "wies_tworyczow_1", "pos": [22.9700, 50.7600]},
    {"name": "wies_tworyczow_2", "pos": [22.9800, 50.7600]},
    {"name": "wies_tworyczow_3", "pos": [22.9600, 50.7600]},

    {"name": "wies_bodaczow_1",  "pos": [23.0000, 50.7100]},
    {"name": "wies_bodaczow_2",  "pos": [23.0100, 50.7100]},
    {"name": "wies_bodaczow_3",  "pos": [22.9900, 50.7100]},

    {"name": "wies_goraj_1",     "pos": [22.6650, 50.7150]},
    {"name": "wies_goraj_2",     "pos": [22.6750, 50.7150]},
    {"name": "wies_goraj_3",     "pos": [22.6550, 50.7150]},


    # --- TYP 4: Kaszuby (Zabudowa rozproszona między jeziorami i lasami) ---
    {"name": "wies_chmielno_1",  "pos": [18.1020, 54.3250]},
    {"name": "wies_chmielno_2",  "pos": [18.1120, 54.3250]},
    {"name": "wies_chmielno_3",  "pos": [18.0920, 54.3250]},

    {"name": "wies_ostrzyce_1",  "pos": [18.1300, 54.2800]},
    {"name": "wies_ostrzyce_2",  "pos": [18.1400, 54.2800]},
    {"name": "wies_ostrzyce_3",  "pos": [18.1200, 54.2800]},

    {"name": "wies_szymbark_1",  "pos": [18.1000, 54.2200]},
    {"name": "wies_szymbark_2",  "pos": [18.1100, 54.2200]},
    {"name": "wies_szymbark_3",  "pos": [18.0900, 54.2200]},

    {"name": "wies_wdzydze_1",   "pos": [17.9350, 54.0050]},
    {"name": "wies_wdzydze_2",   "pos": [17.9450, 54.0050]},
    {"name": "wies_wdzydze_3",   "pos": [17.9250, 54.0050]},


    # --- TYP 5: Podhale / Bieszczady (Głębokie doliny, domy w liniach u podnóża gór) ---
    {"name": "wies_wetlina_1",   "pos": [22.4670, 49.1500]},
    {"name": "wies_wetlina_2",   "pos": [22.4770, 49.1450]},
    {"name": "wies_wetlina_3",   "pos": [22.4570, 49.1550]},

    {"name": "wies_smerek_1",    "pos": [22.4350, 49.1700]},
    {"name": "wies_smerek_2",    "pos": [22.4450, 49.1650]},
    {"name": "wies_smerek_3",    "pos": [22.4250, 49.1750]},

    {"name": "wies_chocholow_1", "pos": [19.8200, 49.3650]},
    {"name": "wies_chocholow_2", "pos": [19.8200, 49.3750]},
    {"name": "wies_chocholow_3", "pos": [19.8200, 49.3550]},

    {"name": "wies_bialka_1",    "pos": [20.1050, 49.3900]},
    {"name": "wies_bialka_2",    "pos": [20.1050, 49.4000]},
    {"name": "wies_bialka_3",    "pos": [20.1050, 49.3800]}
]

def main():
    parser = argparse.ArgumentParser(description="Detekcja zabudowań (Auto-Labeling Pipeline)")
    parser.add_argument('--step', type=str, choices=['fetch', 'train_classifier', 'prepare_masks', 'preprocess', 'train_unet', 'predict', 'all'], required=True)
    parser.add_argument('--image', type=str, help='Ścieżka do zdjęcia testowego (wymagane dla predict)')
    args = parser.parse_args()

    if args.step in ['fetch', 'all']:
        print("\n--- KROK 1: POBIERANIE DANYCH ---")
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
        train_model(epochs=10)
        
    if args.step == 'predict':
        if not args.image:
            print("Błąd: Podaj ścieżkę do zdjęcia")
            return
        print(f"\n--- PREDYKCJA U-NET DLA: {args.image} ---")
        predict_and_count(args.image)

if __name__ == "__main__":
    main()