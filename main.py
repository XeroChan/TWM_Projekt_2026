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

    # # 2. Kraków (Kazimierz, Grzegórzki, Podgórze)
    {"name": "miasto_kra_1", "pos": [19.9450, 50.0510]},
    {"name": "miasto_kra_2", "pos": [19.9650, 50.0600]},
    {"name": "miasto_kra_3", "pos": [19.9400, 50.0350]},

    # # 3. Łódź (Piotrkowska, Bałuty, Widzew)
    {"name": "miasto_lod_1", "pos": [19.4560, 51.7600]},
    {"name": "miasto_lod_2", "pos": [19.4460, 51.7800]},
    {"name": "miasto_lod_3", "pos": [19.4960, 51.7500]},

    # # 4. Wrocław (Stare Miasto, Nadodrze, Krzyki)
    {"name": "miasto_wro_1", "pos": [17.0322, 51.1100]},
    {"name": "miasto_wro_2", "pos": [17.0222, 51.1250]},
    {"name": "miasto_wro_3", "pos": [17.0122, 51.0850]},

    # # 5. Poznań (Rataje - blokowiska, Wilda, Łazarz)
    {"name": "miasto_poz_1", "pos": [16.9550, 52.3850]},
    {"name": "miasto_poz_2", "pos": [16.9250, 52.3900]},
    {"name": "miasto_poz_3", "pos": [16.8950, 52.3950]},

    # # 6. Gdańsk (Stocznia/Przemysł, Wrzeszcz, Przymorze)
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
    {"name": "wies_tokary_1",    "pos": [23.1915, 52.4287]},  #[52 25 43.63   23 11 29.63]},
    {"name": "wies_tokary_2",    "pos": [23.1007, 52.4576]},  # [52 27 27.64    23 06 02 81]},
    {"name": "wies_tokary_3",    "pos": [23.2258, 52.5069]},  #[52 30 24.85    23 13 33 14]},
    {"name": "wies_chmielno_1",  "pos": [16.678103, 54.344153]},  #54 20 38 95    16 40 41 17
    {"name": "wies_chmielno_2",  "pos": [19.827028, 53.730656]},  #53 43 50 36    19 49  37 30
    {"name": "wies_chmielno_3",  "pos": [19.821178, 53.664986]},  #53 39 53 95    19 49 16 24

    {"name": "wies_ostrzyce_1",  "pos": [14.779503, 53.344675]},  #53 20 40 83    14 46 46 21
    {"name": "wies_ostrzyce_2",  "pos": [14.562742, 53.345833]},  #53 20 45 00    14 33 45 87
    {"name": "wies_ostrzyce_3",  "pos": [14.747861, 53.347944]},  #53 20 52 60    14 44 52 30

    {"name": "wies_szymbark_1",  "pos": [14.77385, 53.396561]},  #53 23 47 62    14 46 25 86
    {"name": "wies_szymbark_2",  "pos": [15.480072, 52.925086]},  #52 55 30 31    15 28 48 26
    {"name": "wies_szymbark_3",  "pos": [16.842231, 53.586472]},  #53 35 11 30    16 50 32 03

    {"name": "wies_wdzydze_1",   "pos": [18.118806, 53.740683]},    #53 44 26 46   18 07 07 70
    {"name": "wies_wdzydze_2",   "pos": [17.795333, 53.597819]},     #53 35 52 15   17 47 43 20
    {"name": "wies_wdzydze_3",   "pos": [17.791497, 53.644261]},    #53 38 39 34    17 47 29 39

    {"name": "wies_chocholow_1", "pos": [18.867142, 49.534122]},    #49 32 02 84    18 52 01 71
    {"name": "wies_chocholow_2", "pos": [19.886614, 49.2798]},     #49 16 47 28    19 53 11 81
    {"name": "wies_chocholow_3", "pos": [20.115903, 49.301358]},    #49 18 04 89    20 06 57 25

    {"name": "wies_bialka_1",    "pos": [20.070728, 49.304961]},    #49 18 17 86    20 04 14 62
    {"name": "wies_bialka_2",    "pos": [20.906103, 49.337156]},    #49 20 13 76    20 54 21 97
    {"name": "wies_bialka_3",    "pos": [21.985056, 49.391269]},     #49 23 28 57    21 59 06 20
    # --- TYP 5: Podhale / Bieszczady (Głębokie doliny, domy w liniach u podnóża gór) ---
    {"name": "wies_wetlina_1",   "pos": [22.4670, 49.1500]},
    {"name": "wies_wetlina_2",   "pos": [22.4770, 49.1450]},
    {"name": "wies_wetlina_3",   "pos": [22.4570, 49.1550]},

    {"name": "wies_smerek_1",    "pos": [22.4350, 49.1700]},
    {"name": "wies_smerek_2",    "pos": [22.4450, 49.1650]},
    {"name": "wies_smerek_3",    "pos": [22.4250, 49.1750]}]

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
        train_model(epochs=18)
        
    if args.step == 'predict':
        if not args.image:
            print("Błąd: Podaj ścieżkę do zdjęcia")
            return
        print(f"\n--- PREDYKCJA U-NET DLA: {args.image} ---")
        predict_and_count(args.image)

if __name__ == "__main__":
    main()