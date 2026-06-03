# Detekcja i zliczanie zabudowań na zdjęciach lotniczych

Projekt wykorzystujący architekturę U-Net do segmentacji semantycznej i detekcji budynków na niezurbanizowanych obszarach Polski. 
Dane pobierane są automatycznie z usług WMS Głównego Urzędu Geodezji i Kartografii.

## Struktura potoku danych (Data Pipeline)
1. **Raw (`data/raw/`)**: Surowe ortofotomapy i maski pobrane z WMS (1024x1024).
2. **Interim (`data/interim/`)**: Maski wyczyszczone z szumów algorytmami morfologicznymi (1024x1024).
3. **Processed (`data/processed/`)**: Obrazy i wyczyszczone maski pocięte na kafle (np. 256x256) gotowe do treningu U-Net.