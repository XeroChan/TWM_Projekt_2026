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
    {"name": "wies_smerek_3",    "pos": [22.4250, 49.1750]},
    {"name": "wies_a_1", "pos": [22.1015, 51.0228]}, # 51°01'22.17" N, 22°06'05.53" E
    {"name": "wies_a_2", "pos": [22.0819, 51.0268]}, # 51°01'36.60" N, 22°04'54.71" E
    {"name": "wies_a_3", "pos": [22.0538, 51.0326]}, # 51°01'57.30" N, 22°03'13.84" E
    {"name": "wies_b_1", "pos": [22.0721, 51.0553]}, # 51°03'19.06" N, 22°04'19.48" E
    {"name": "wies_b_2", "pos": [21.4585, 52.6001]}, # 52°36'00.47" N, 21°27'30.62" E
    {"name": "wies_b_3", "pos": [18.0639, 52.8595]}, # 52°51'34.07" N, 18°03'50.21" E
    {"name": "wies_c_1", "pos": [18.0024, 52.8902]}, # 52°53'24.85" N, 18°00'08.59" E
    {"name": "wies_c_2", "pos": [17.6007, 52.9924]}, # 52°59'32.47" N, 17°36'02.41" E
    {"name": "wies_c_3", "pos": [17.5964, 52.9956]}, # 52°59'44.00" N, 17°35'46.88" E
    {"name": "wies_d_1", "pos": [16.1957, 54.1909]}, # 54°11'27.08" N, 16°11'44.67" E
    {"name": "wies_d_2", "pos": [16.1922, 54.1926]}, # 54°11'33.42" N, 16°11'32.00" E
    {"name": "wies_d_3", "pos": [17.0485, 54.4619]}, # 54°27'42.79" N, 17°02'54.48" E
    {"name": "wies_e_1", "pos": [17.1045, 54.4708]}, # 54°28'14.92" N, 17°06'16.31" E
    {"name": "wies_e_2", "pos": [18.5573, 54.3643]}, # 54°21'51.54" N, 18°33'26.40" E
    {"name": "wies_e_3", "pos": [18.5567, 54.3654]}, # 54°21'55.30" N, 18°33'23.96" E
    {"name": "wies_f_1", "pos": [18.5503, 54.3671]}, # 54°22'01.66" N, 18°33'01.12" E
    {"name": "wies_f_2", "pos": [18.5504, 54.3691]}, # 54°22'08.89" N, 18°33'01.54" E
    {"name": "wies_f_3", "pos": [18.5713, 54.3649]}, # 54°21'53.46" N, 18°34'16.80" E
    {"name": "wies_g_1", "pos": [18.5951, 54.3651]}, # 54°21'54.30" N, 18°35'42.27" E
    {"name": "wies_g_2", "pos": [19.8381, 53.4955]}, # 53°29'43.67" N, 19°50'17.12" E
    {"name": "wies_g_3", "pos": [19.8299, 53.4988]}, # 53°29'55.81" N, 19°49'47.49" E
    {"name": "wies_h_1", "pos": [19.8234, 53.5016]}, # 53°30'05.91" N, 19°49'24.40" E
    {"name": "wies_h_2", "pos": [19.7965, 53.5098]}, # 53°30'35.37" N, 19°47'47.22" E
    {"name": "wies_h_3", "pos": [19.7068, 53.4991]}, # 53°29'56.74" N, 19°42'24.60" E
    {"name": "wies_i_1", "pos": [19.6838, 53.4995]}, # 53°29'58.05" N, 19°41'01.51" E
    {"name": "wies_i_2", "pos": [19.6803, 53.5025]}, # 53°30'09.17" N, 19°40'49.15" E
    {"name": "wies_i_3", "pos": [16.6756, 51.0104]}, # 51°00'37.55" N, 16°40'32.30" E
    {"name": "wies_j_1", "pos": [19.6502, 53.4999]}, # 53°29'59.80" N, 19°39'00.81" E
    {"name": "wies_j_2", "pos": [19.5915, 53.4178]}, # 53°25'04.01" N, 19°35'29.47" E
    {"name": "wies_j_3", "pos": [19.6990, 53.3532]}, # 53°21'11.41" N, 19°41'56.30" E
    {"name": "wies_k_1", "pos": [19.7111, 53.3606]}, # 53°21'38.28" N, 19°42'39.96" E
    {"name": "wies_k_2", "pos": [16.6799, 51.0040]}, # 51°00'14.33" N, 16°40'47.58" E
    {"name": "wies_k_3", "pos": [16.7565, 51.0342]}, # 51°02'02.94" N, 16°45'23.28" E
    {"name": "wies_l_1", "pos": [16.7576, 51.0370]}, # 51°02'13.29" N, 16°45'27.29" E
    {"name": "wies_l_2", "pos": [16.7660, 51.0411]}, # 51°02'27.84" N, 16°45'57.51" E
    {"name": "wies_l_3", "pos": [22.4002, 50.6006]}, # 50°36'02.16" N, 22°24'00.69" E
    {"name": "wies_m_1", "pos": [22.4108, 50.5981]}, # 50°35'53.11" N, 22°24'38.81" E
    {"name": "wies_m_2", "pos": [22.4254, 50.5962]}  # 50°35'46.35" N, 22°25'31.61" E
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
        train_model(epochs=18)
        
    if args.step == 'predict':
        if not args.image:
            print("Błąd: Podaj ścieżkę do zdjęcia")
            return
        print(f"\n--- PREDYKCJA U-NET DLA: {args.image} ---")
        predict_and_count(args.image)

if __name__ == "__main__":
    main()