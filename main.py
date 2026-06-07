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
    
    # =========================================================
    # DODATKOWE PUNKTY (Z GEOPORTALU)
    # =========================================================
    {"name": "nowy_obszar_1", "pos": [22.1015, 51.0228]}, # 51°01'22.17" N, 22°06'05.53" E
    {"name": "nowy_obszar_2", "pos": [22.0819, 51.0268]}, # 51°01'36.60" N, 22°04'54.71" E
    {"name": "nowy_obszar_3", "pos": [22.0538, 51.0326]}, # 51°01'57.30" N, 22°03'13.84" E
    {"name": "nowy_obszar_4", "pos": [22.0721, 51.0553]}, # 51°03'19.06" N, 22°04'19.48" E
    {"name": "nowy_obszar_5", "pos": [21.4585, 52.6001]}, # 52°36'00.47" N, 21°27'30.62" E
    {"name": "nowy_obszar_6", "pos": [18.0639, 52.8595]}, # 52°51'34.07" N, 18°03'50.21" E
    {"name": "nowy_obszar_7", "pos": [18.0024, 52.8902]}, # 52°53'24.85" N, 18°00'08.59" E
    {"name": "nowy_obszar_8", "pos": [17.6007, 52.9924]}, # 52°59'32.47" N, 17°36'02.41" E
    {"name": "nowy_obszar_9", "pos": [17.5964, 52.9956]}, # 52°59'44.00" N, 17°35'46.88" E
    {"name": "nowy_obszar_10", "pos": [16.1957, 54.1909]}, # 54°11'27.08" N, 16°11'44.67" E
    {"name": "nowy_obszar_11", "pos": [16.1922, 54.1926]}, # 54°11'33.42" N, 16°11'32.00" E
    {"name": "nowy_obszar_12", "pos": [17.0485, 54.4619]}, # 54°27'42.79" N, 17°02'54.48" E
    {"name": "nowy_obszar_13", "pos": [17.1045, 54.4708]}, # 54°28'14.92" N, 17°06'16.31" E
    {"name": "nowy_obszar_14", "pos": [18.5573, 54.3643]}, # 54°21'51.54" N, 18°33'26.40" E
    {"name": "nowy_obszar_15", "pos": [18.5567, 54.3654]}, # 54°21'55.30" N, 18°33'23.96" E
    {"name": "nowy_obszar_16", "pos": [18.5503, 54.3671]}, # 54°22'01.66" N, 18°33'01.12" E
    {"name": "nowy_obszar_17", "pos": [18.5504, 54.3691]}, # 54°22'08.89" N, 18°33'01.54" E
    {"name": "nowy_obszar_18", "pos": [18.5713, 54.3649]}, # 54°21'53.46" N, 18°34'16.80" E
    {"name": "nowy_obszar_19", "pos": [18.5951, 54.3651]}, # 54°21'54.30" N, 18°35'42.27" E
    {"name": "nowy_obszar_20", "pos": [19.8381, 53.4955]}, # 53°29'43.67" N, 19°50'17.12" E
    {"name": "nowy_obszar_21", "pos": [19.8299, 53.4988]}, # 53°29'55.81" N, 19°49'47.49" E
    {"name": "nowy_obszar_22", "pos": [19.8234, 53.5016]}, # 53°30'05.91" N, 19°49'24.40" E
    {"name": "nowy_obszar_23", "pos": [19.7965, 53.5098]}, # 53°30'35.37" N, 19°47'47.22" E
    {"name": "nowy_obszar_24", "pos": [19.7068, 53.4991]}, # 53°29'56.74" N, 19°42'24.60" E
    {"name": "nowy_obszar_25", "pos": [19.6838, 53.4995]}, # 53°29'58.05" N, 19°41'01.51" E
    {"name": "nowy_obszar_26", "pos": [19.6803, 53.5025]}, # 53°30'09.17" N, 19°40'49.15" E
    {"name": "nowy_obszar_27", "pos": [16.6756, 51.0104]}, # 51°00'37.55" N, 16°40'32.30" E
    {"name": "nowy_obszar_28", "pos": [19.6502, 53.4999]}, # 53°29'59.80" N, 19°39'00.81" E
    {"name": "nowy_obszar_29", "pos": [19.5915, 53.4178]}, # 53°25'04.01" N, 19°35'29.47" E
    {"name": "nowy_obszar_30", "pos": [19.6990, 53.3532]}, # 53°21'11.41" N, 19°41'56.30" E
    {"name": "nowy_obszar_31", "pos": [19.7111, 53.3606]}, # 53°21'38.28" N, 19°42'39.96" E
    {"name": "nowy_obszar_32", "pos": [16.6799, 51.0040]}, # 51°00'14.33" N, 16°40'47.58" E
    {"name": "nowy_obszar_33", "pos": [16.7565, 51.0342]}, # 51°02'02.94" N, 16°45'23.28" E
    {"name": "nowy_obszar_34", "pos": [16.7576, 51.0370]}, # 51°02'13.29" N, 16°45'27.29" E
    {"name": "nowy_obszar_35", "pos": [16.7660, 51.0411]}, # 51°02'27.84" N, 16°45'57.51" E
    {"name": "nowy_obszar_36", "pos": [22.4002, 50.6006]}, # 50°36'02.16" N, 22°24'00.69" E
    {"name": "nowy_obszar_37", "pos": [22.4108, 50.5981]}, # 50°35'53.11" N, 22°24'38.81" E
    {"name": "nowy_obszar_38", "pos": [22.4254, 50.5962]}, # 50°35'46.35" N, 22°25'31.61" E
    {"name": "nowy_obszar_39", "pos": [22.4338, 50.5951]}, # 50°35'42.40" N, 22°26'01.74" E
    {"name": "nowy_obszar_40", "pos": [22.4398, 50.5942]}, # 50°35'38.99" N, 22°26'23.38" E
    {"name": "nowy_obszar_41", "pos": [22.1404, 50.1668]}, # 50°10'00.45" N, 22°08'25.33" E
    {"name": "nowy_obszar_42", "pos": [22.1427, 50.1695]}, # 50°10'10.37" N, 22°08'33.80" E
    {"name": "nowy_obszar_43", "pos": [22.1469, 50.1815]}, # 50°10'53.51" N, 22°08'49.01" E
    {"name": "nowy_obszar_44", "pos": [22.1460, 50.1823]}, # 50°10'56.21" N, 22°08'45.42" E
    {"name": "nowy_obszar_45", "pos": [22.1468, 50.1832]}, # 50°10'59.48" N, 22°08'48.53" E
    {"name": "nowy_obszar_46", "pos": [22.1490, 50.1951]}, # 50°11'42.19" N, 22°08'56.40" E
    {"name": "nowy_obszar_47", "pos": [22.1465, 50.1985]}, # 50°11'54.57" N, 22°08'47.45" E
    {"name": "nowy_obszar_48", "pos": [22.1458, 50.1995]}, # 50°11'58.28" N, 22°08'45.04" E
    {"name": "nowy_obszar_49", "pos": [22.1465, 50.2051]}, # 50°12'18.23" N, 22°08'47.46" E
    {"name": "nowy_obszar_50", "pos": [22.1470, 50.2069]}, # 50°12'24.96" N, 22°08'49.26" E
    {"name": "nowy_obszar_51", "pos": [22.1492, 50.2078]}, # 50°12'28.09" N, 22°08'57.05" E
    {"name": "nowy_obszar_52", "pos": [22.1487, 50.2088]}, # 50°12'31.61" N, 22°08'55.36" E
    {"name": "nowy_obszar_53", "pos": [22.1484, 50.2106]}, # 50°12'38.29" N, 22°08'54.20" E
    {"name": "nowy_obszar_54", "pos": [22.1478, 50.2128]}, # 50°12'46.17" N, 22°08'51.96" E
    {"name": "nowy_obszar_55", "pos": [22.1493, 50.2138]}, # 50°12'49.84" N, 22°08'57.41" E
    {"name": "nowy_obszar_56", "pos": [22.1503, 50.2147]}, # 50°12'52.85" N, 22°09'00.97" E
    {"name": "nowy_obszar_57", "pos": [22.1537, 50.2168]}, # 50°13'00.34" N, 22°09'13.37" E
    {"name": "nowy_obszar_58", "pos": [22.1582, 50.2181]}, # 50°13'05.18" N, 22°09'29.44" E
    {"name": "nowy_obszar_59", "pos": [22.1572, 50.2191]}, # 50°13'08.70" N, 22°09'26.00" E
    {"name": "nowy_obszar_60", "pos": [22.1600, 50.2213]}, # 50°13'16.85" N, 22°09'36.16" E
    {"name": "nowy_obszar_61", "pos": [22.1637, 50.2212]}, # 50°13'16.27" N, 22°09'49.14" E
    {"name": "nowy_obszar_62", "pos": [22.1659, 50.2223]}, # 50°13'20.43" N, 22°09'57.39" E
    {"name": "nowy_obszar_63", "pos": [22.1670, 50.2245]}, # 50°13'28.37" N, 22°10'01.30" E
    {"name": "nowy_obszar_64", "pos": [22.1678, 50.2247]}, # 50°13'28.89" N, 22°10'04.13" E
    {"name": "nowy_obszar_65", "pos": [22.1667, 50.2259]}, # 50°13'33.15" N, 22°10'00.10" E
    {"name": "nowy_obszar_66", "pos": [22.1710, 50.2275]}, # 50°13'39.13" N, 22°10'15.72" E
    {"name": "nowy_obszar_67", "pos": [22.3591, 49.3492]}, # 49°20'57.07" N, 22°21'32.79" E
    {"name": "nowy_obszar_68", "pos": [22.3539, 49.3408]}, # 49°20'26.71" N, 22°21'13.89" E
    {"name": "nowy_obszar_69", "pos": [22.2985, 49.3280]}, # 49°19'40.86" N, 22°17'54.50" E
    {"name": "nowy_obszar_70", "pos": [22.2966, 49.3287]}, # 49°19'43.35" N, 22°17'47.84" E
    {"name": "nowy_obszar_71", "pos": [22.2740, 49.3161]}, # 49°18'57.81" N, 22°16'26.19" E
    {"name": "nowy_obszar_72", "pos": [22.2783, 49.3117]}, # 49°18'42.14" N, 22°16'41.94" E
    {"name": "nowy_obszar_73", "pos": [22.2785, 49.3106]}, # 49°18'37.98" N, 22°16'42.59" E
    {"name": "nowy_obszar_74", "pos": [22.2796, 49.3063]}, # 49°18'22.60" N, 22°16'46.51" E
    {"name": "nowy_obszar_75", "pos": [22.2830, 49.2780]}, # 49°16'40.85" N, 22°16'58.60" E
    {"name": "nowy_obszar_76", "pos": [20.7248, 49.6115]}, # 49°36'41.40" N, 20°43'29.19" E
    {"name": "nowy_obszar_77", "pos": [20.7189, 49.6197]}, # 49°37'10.92" N, 20°43'08.08" E
    {"name": "nowy_obszar_78", "pos": [20.7188, 49.6207]}, # 49°37'14.58" N, 20°43'07.76" E
    {"name": "nowy_obszar_79", "pos": [20.3896, 49.7197]}, # 49°43'10.97" N, 20°23'22.44" E
    {"name": "nowy_obszar_80", "pos": [20.3838, 49.7241]}, # 49°43'26.58" N, 20°23'01.61" E
    {"name": "nowy_obszar_81", "pos": [19.2383, 50.2242]}, # 50°13'27.05" N, 19°14'17.98" E
    {"name": "nowy_obszar_82", "pos": [19.2378, 50.2296]}, # 50°13'46.54" N, 19°14'16.16" E
    {"name": "nowy_obszar_83", "pos": [19.2725, 50.2621]}, # 50°15'43.62" N, 19°16'20.98" E
    {"name": "nowy_obszar_84", "pos": [19.2761, 50.2893]}, # 50°17'21.61" N, 19°16'34.08" E
    {"name": "nowy_obszar_85", "pos": [19.2714, 50.2947]}, # 50°17'40.89" N, 19°16'16.89" E
    {"name": "nowy_obszar_86", "pos": [20.0622, 50.0446]}, # 50°02'40.50" N, 20°03'44.04" E
    {"name": "nowy_obszar_87", "pos": [20.0658, 50.0469]}, # 50°02'48.67" N, 20°03'56.81" E
    {"name": "nowy_obszar_88", "pos": [20.4536, 49.9657]}, # 49°57'56.53" N, 20°27'13.11" E
    {"name": "nowy_obszar_89", "pos": [20.4565, 49.9677]}, # 49°58'03.62" N, 20°27'23.24" E
    {"name": "nowy_obszar_90", "pos": [20.8926, 49.8912]}, # 49°53'28.43" N, 20°53'33.21" E
    {"name": "nowy_obszar_91", "pos": [20.9378, 49.5621]}, # 49°33'43.59" N, 20°56'16.15" E
    {"name": "nowy_obszar_92", "pos": [20.9424, 49.5632]}, # 49°33'47.46" N, 20°56'32.70" E
    {"name": "nowy_obszar_93", "pos": [20.9535, 49.5638]}, # 49°33'49.64" N, 20°57'12.63" E
    {"name": "nowy_obszar_94", "pos": [20.9558, 49.5653]}, # 49°33'55.04" N, 20°57'21.04" E
    {"name": "nowy_obszar_95", "pos": [20.9706, 49.5633]}, # 49°33'47.93" N, 20°58'14.13" E
    {"name": "nowy_obszar_96", "pos": [20.9777, 49.5633]}, # 49°33'47.81" N, 20°58'39.71" E
    {"name": "nowy_obszar_97", "pos": [20.9798, 49.5658]}, # 49°33'56.86" N, 20°58'47.13" E
    {"name": "nowy_obszar_98", "pos": [20.9768, 49.5780]}, # 49°34'40.82" N, 20°58'36.39" E
    {"name": "nowy_obszar_99", "pos": [20.9785, 49.5790]}, # 49°34'44.28" N, 20°58'42.50" E
    {"name": "nowy_obszar_100", "pos": [21.0698, 49.7927]} # 49°47'33.73" N, 21°04'11.15" E
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