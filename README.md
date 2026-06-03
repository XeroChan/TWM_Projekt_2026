# Detekcja i zliczanie zabudowań na zdjęciach lotniczych

Projekt wykorzystujący architekturę U-Net do segmentacji semantycznej i detekcji budynków na niezurbanizowanych obszarach Polski. 
Dane pobierane są automatycznie z usług WMS Głównego Urzędu Geodezji i Kartografii.

## Struktura potoku danych (Data Pipeline)
1. **Raw (`data/raw/`)**: Surowe ortofotomapy i maski pobrane z WMS (1024x1024).
2. **Interim (`data/interim/`)**: Maski wyczyszczone z szumów algorytmami morfologicznymi (1024x1024).
3. **Processed (`data/processed/`)**: Obrazy i wyczyszczone maski pocięte na kafle (np. 256x256) gotowe do treningu U-Net.

## Struktura i architektura projektu

Projekt składa się z kilku powiązanych ze sobą modułów, które realizują pełen proces: od pobrania danych, przez przygotowanie masek i wykluczenie miast (klasyfikator), aż po segmentację budynków na obszarach wiejskich (model U-Net) i ich zliczenie.

---

### **`main.py`** (Plik startowy)

Plik startowy (entrypoint) całego projektu. Zarządza przepływem działania aplikacji (pipeline). Wykorzystuje bibliotekę `argparse` do przyjmowania argumentów z linii komend, co pozwala na uruchomienie poszczególnych etapów (np. pobierania danych, treningu czy predykcji) lub całego procesu naraz (flaga `--step all`). Plik ten zawiera również predefiniowane współrzędne (`CENTER_POINTS`) dla różnych typów miast i wsi, z których pobierane są dane satelitarne. Kroki wykonywania potoku są opisane tekstowo w pliku. Zalecane jest uruchamianie pojedynczo od kroku 2 do ostatniego kroku predict z uwagi na to, że wykonanie kroku 1 fetch może zablokować użytkownika na geoportal.

---

## Folder `src/` (Kluczowe moduły)

* **`data_fetcher.py` (Pobieranie danych)**
Odpowiada za automatyczne pobieranie danych na podstawie podanych współrzędnych geograficznych.
* Łączy się z serwisem WMS Geoportalu (GUGiK), aby pobrać zdjęcia satelitarne (ortofotomapy) o wysokiej rozdzielczości.
* Łączy się z serwisem WMS Krajowej Integracji Ewidencji Gruntów, aby pobrać maski budynków (kształty, gdzie obiekty faktycznie stoją).


* **`classifier_model.py` (Klasyfikator Miasto vs. Wieś)**
Definiuje model sieci neuronowej oparty na architekturze **ResNet-18** (z wgranymi domyślnymi wagami pre-trenowanymi na obrazach). Model ten został zmodyfikowany na końcu (ostatnia warstwa) w taki sposób, aby zamiast 1000 klas wyrzucał jedną wartość – dokonuje klasyfikacji binarnej. Jego jedynym celem jest ocena, czy dane zdjęcie satelitarne przedstawia miasto, czy wieś.
* **`mask_preparation.py` (Inteligentne czyszczenie masek)**
Ten moduł działa jako "bramkarz" (*Data Gatekeeper*). Wykorzystuje wytrenowany model z `classifier_model.py`.
* Ocenia każde zdjęcie. Jeśli model uzna, że zdjęcie to **MIASTO**, maska budynków dla tego zdjęcia jest w całości zerowana (czyszczona do czerni), ponieważ celem projektu jest szukanie budynków tylko na wsi.
* Jeśli to **WIEŚ**, algorytm "czyści" maskę (np. używając operacji morfologicznych z biblioteki OpenCV), aby była wyraźniejsza i wolna od małych zakłóceń, zachowując kontury budynków.


* **`preprocessing.py` (Przygotowanie danych do U-Net)**
Przygotowuje zdjęcia i oczyszczone maski dla głównego modelu do segmentacji.
* Dokonuje wycięcia centralnego (*Center Crop*) z dużych zdjęć (np. z 1024x1024 na 512x512).
* Rozcina te zdjęcia oraz maski na mniejsze "kafle" o wymiarach 256x256 pikseli (*Tiling*), co jest optymalnym rozmiarem ułatwiającym nauczanie modelu U-Net.


* **`trainer.py` (Zarządzanie procesem uczenia)**
Zawiera logikę ładowania danych (`Dataset` i `DataLoader` z biblioteki PyTorch) oraz pętle uczące dla obu modeli:
* `train_classifier()` - uczy model ResNet-18 odróżniać miasta (Label: `1.0`) od wsi (Label: `0.0`) na surowych danych. Zapisuje jego "mózg" do `classifier_weights.pth`.
* `train_model()` - uczy główny model U-Net (na pociętych kaflach), jak wygląda zarys budynków na wsi. Zapisuje wagi do `unet_weights.pth`.


* **`unet_model.py` (Architektura modelu segmentacji)**
Definiuje główny model sieci neuronowej o architekturze **U-Net**. Jest to klasyczna architektura, która z jednej strony "zgniata" obraz, aby wyciągnąć z niego cechy, a następnie odbudowuje go do oryginalnego rozmiaru, zwracając precyzyjną maskę wskazującą dokładne piksele, gdzie na obrazku wejściowym znajduje się budynek.
* **`postprocessing.py` (Predykcja i Zliczanie)**
Używany na samym końcu do testowania wytrenowanego modelu na nowych zdjęciach.
* Odbiera polecenie predykcji konkretnego zdjęcia.
* Przepuszcza zdjęcie przez wytrenowany model U-Net, który generuje nową, wyestymowaną maskę budynków.
* Analizuje stworzoną maskę przy pomocy OpenCV (`connectedComponentsWithStats`), by policzyć osobne "plamy" i zwraca informację o ilości wykrytych budynków na zdjęciu, zapisując podgląd do pliku.
