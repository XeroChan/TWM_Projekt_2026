# Detekcja i zliczanie zabudowań na zdjęciach lotniczych

Projekt wykorzystujący architekturę U-Net do segmentacji semantycznej i detekcji budynków na niezurbanizowanych obszarach Polski.
Dane pobierane są automatycznie z usług WMS Głównego Urzędu Geodezji i Kartografii.

## Struktura potoku danych (Data Pipeline)

Dane przechodzą przez kolejne foldery — na każdym etapie zostają zachowane, więc nic nie ginie:

1. **Raw (`data/raw/`)** — surowe ortofotomapy i maski budynków pobrane z WMS (1024x1024).
2. **Interim (`data/interim/`)** — maski po wstępnym czyszczeniu morfologicznym i wyzerowaniu miast przez klasyfikator (1024x1024).
3. **Processed (`data/processed/`)** — obrazy i maski pocięte na kafle 256x256 (pełna, nieprzefiltrowana pula).
4. **Fitted (`data/fitted/`)** — **ręcznie wyselekcjonowane, dobre kafle** (przez `review_masks.py`). To **na nich** trenuje się U-Net.

```
raw  ──►  interim  ──►  processed  ──►  fitted  ──►  trening U-Net
(WMS)    (czyszcz.)     (kafle)      (ręczny filtr)
```

### Dlaczego istnieje krok `fitted`?

Maski generowane automatycznie z danych BDOT10k bywają błędne — np. siedzą na obiektach, których fizycznie jeszcze nie ma (budynki w budowie, plac z piaskiem) albo są przesunięte względem zdjęcia. Trening na takich sprzecznych etykietach ogranicza skuteczność modelu. Ręczny przegląd (`review_masks.py`) odsiewa złe pary obraz–maska i zostawia tylko te, gdzie maska faktycznie pokrywa dach. Dzięki temu U-Net uczy się na czystych danych.

**Zbiór walidacyjny** powstaje naturalnie: kafle z `processed`, których **nie ma** w `fitted`, to dane niewidziane przez model — `visualize_predictions.py` pokazuje predykcje właśnie na nich.

## Uruchamianie potoku

Kroki `main.py` (zalecane uruchamianie pojedynczo — krok `fetch` może chwilowo zablokować dostęp do Geoportalu):

```bash
python main.py --step fetch              # 1. pobranie ortofoto + masek z WMS  -> data/raw
python main.py --step train_classifier   # 2. trening klasyfikatora miasto/wieś
python main.py --step prepare_masks       # 3. czyszczenie masek + zerowanie miast -> data/interim
python main.py --step preprocess          # 4. cięcie na kafle 256x256          -> data/processed
python review_masks.py                    # 4b. RĘCZNY filtr dobrych kafli       -> data/fitted
python main.py --step train_unet          # 5. trening U-Net na data/fitted      -> unet_weights.pth
python main.py --step predict --image <ścieżka>   # predykcja + zliczenie budynków
```

Krok `4b` (`review_masks.py`) jest manualny i wykonuje się między cięciem na kafle a treningiem.

## Struktura i architektura projektu

Projekt realizuje pełen proces: od pobrania danych, przez przygotowanie i ręczną kontrolę masek oraz wykluczenie miast (klasyfikator), aż po segmentację budynków na obszarach wiejskich (U-Net) i ich zliczenie.

### **`main.py`** (Plik startowy)

Entrypoint całego projektu. Zarządza przepływem działania (pipeline) przez `argparse` — pozwala uruchomić pojedyncze etapy albo cały proces naraz (flaga `--step all`). Zawiera predefiniowane współrzędne (`CENTER_POINTS`) obszarów, z których pobierane są dane.

---

## Folder `src/` (Kluczowe moduły)

* **`data_fetcher.py` (Pobieranie danych)**
Automatyczne pobieranie danych na podstawie współrzędnych geograficznych.
    * Łączy się z WMS Geoportalu (GUGiK), aby pobrać ortofotomapy wysokiej rozdzielczości.
    * Łączy się z WMS Krajowej Integracji Ewidencji Gruntów, aby pobrać maski budynków (gdzie obiekty faktycznie stoją).

* **`classifier_model.py` (Klasyfikator Miasto vs. Wieś)**
Model oparty na architekturze **ResNet-18** (z wagami pre-trenowanymi), z ostatnią warstwą zmodyfikowaną do klasyfikacji binarnej. Ocenia, czy zdjęcie przedstawia miasto, czy wieś.

* **`mask_preparation.py` (Inteligentne czyszczenie masek)**
Działa jak "bramkarz" (*Data Gatekeeper*) z użyciem wytrenowanego klasyfikatora.
    * Jeśli zdjęcie to **MIASTO** — maska jest w całości zerowana (cel projektu to budynki na wsi).
    * Jeśli **WIEŚ** — maska jest czyszczona operacjami morfologicznymi (OpenCV), zachowując kontury budynków.

* **`preprocessing.py` (Cięcie na kafle)**
    * Wycięcie centralne (*Center Crop*) z dużych zdjęć (1024x1024 → 512x512).
    * Rozcięcie zdjęć i masek na kafle 256x256 (*Tiling*). Kafle z pustą maską (miasta) trafiają do osobnego folderu i nie wchodzą do treningu.

* **`datasets.py` (Wczytywanie danych)**
Klasy `Dataset` dla PyTorch: `RawClassifierDataset` (klasyfikator) i `BuildingDataset` (U-Net, z augmentacją i opcjonalnym cache w RAM).

* **`trainer.py` (Zarządzanie procesem uczenia)**
Pętle uczące dla obu modeli:
    * `train_classifier()` — uczy ResNet-18 odróżniać miasta (`1.0`) od wsi (`0.0`); zapisuje `classifier_weights.pth`.
    * `train_model()` — uczy U-Net segmentacji budynków na kaflach z **`data/fitted`** (early stopping na Val Dice); zapisuje `unet_weights.pth`.

* **`unet_model.py` (Architektura segmentacji)**
Klasyczna architektura **U-Net** — ścieżka kodująca wyciąga cechy, dekodująca odbudowuje obraz do precyzyjnej maski pikseli budynków.

* **`postprocessing.py` (Predykcja i Zliczanie)**
    * Przepuszcza zdjęcie przez wytrenowany U-Net, generując maskę budynków.
    * `build_clean_mask` czyści maskę i liczy osobne obiekty (`connectedComponentsWithStats`).
    * `draw_instances` rysuje kontury i etykiety budynków; `interactive_instance_viewer` pokazuje wynik interaktywnie (Plotly).

---

## Skrypty pomocnicze (katalog główny)

* **`review_masks.py` (Ręczny filtr masek)**
Przegląd par obraz+maska z `data/processed` z nałożoną maską. Akceptowane (maska na dachu) trafiają do `data/fitted`, odrzucane są pomijane. Obsługuje wznawianie — można robić na raty.

* **`visualize_predictions.py` (Masowa wizualizacja)**
Składa predykcje na wielu kaflach **walidacyjnych** (spoza `fitted`) w jedną siatkę z liczbą wykrytych budynków. Wynik → `predictions_grid.png`.
