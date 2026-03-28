# Deployment / uruchamianie pipeline'u

Ten projekt składa się z trzech głównych procesów:

- Day Collector – day_collector.py (alias do main.py)
- Night Trainer – night_trainer.py (korzysta z nightly.py)
- Train Classifier – train_classifier.py
- (opcjonalnie) Day Recognizer – do zaimplementowania osobno

## A) Ręczne uruchamianie (najprostsze)

1. Rano – start Day Collector:

```bash
python day_collector.py
```

2. Wieczorem – zatrzymaj proces dzienny, odpal nocne batch'e:

```bash
# zatrzymanie procesu day_collector (sposób zależny od systemu)
# przykładowo na Linux można użyć ps/kill lub systemd

python night_trainer.py
python train_classifier.py
```

3. (Opcjonalnie) Następnego dnia – jeśli istnieje Day Recognizer:

```bash
python day_recognizer.py
```

## B) Automatyzacja (systemd / crontab na Raspberry Pi)

Przykładowa koncepcja:

- Day Collector jako usługa systemd z `Restart=always`, uruchamiana przy starcie systemu, wywołująca `python day_collector.py`.
- Night Trainer jako systemd timer (np. godzina 23:00), wywołujący `python night_trainer.py`.
- Train Classifier jako drugi timer (np. 23:10), wywołujący `python train_classifier.py`.
- (Opcjonalnie) Day Recognizer jako osobna usługa, jeśli ma działać w czasie rzeczywistym.

Katalogi danych używane w pipeline'ie:

- data/faces_raw/   – surowe cropy twarzy z Day Collector
- data/faces_clean/ – dodatkowo oczyszczone i posegregowane twarze z Night Trainer
- data/embeddings/  – embeddingi twarzy (NumPy .npy)
- data/clusters/    – etykiety klastrów i ścieżki
- data/models/      – centroidy embeddingów oraz pliki modelu klasyfikatora

Upewnij się, że procesy mają prawa zapisu do katalogu data/ oraz że używasz tego samego środowiska Pythona (venv) dla wszystkich kroków.

## C) Ubuntu 24 na Raspberry Pi — systemd z oknem 07:00–15:00

Poniższa konfiguracja zakłada, że kod projektu jest w katalogu `/opt/vision-pipeline` i używasz systemowego Pythona (`/usr/bin/python`). W razie potrzeby zmień ścieżki w plikach service.

### 1. Pliki systemd (w repozytorium)

W katalogu `config/systemd` znajdują się przykładowe jednostki:

- `day_collector.service` – długi proces akwizycji (Day Collector)
- `day_collector_start.timer` – start o 07:00
- `day_collector_stop.service` + `day_collector_stop.timer` – zatrzymanie o 15:00
- `night_trainer.service` + `night_trainer.timer` – Night Trainer o 15:01
- `train_classifier.service` + `train_classifier.timer` – Train Classifier o 15:10

### 2. Instalacja na Ubuntu 24 (jako root / z sudo)

1. Skompletuj projekt na RPi, np.:

```bash
sudo mkdir -p /opt/vision-pipeline
sudo chown -R $USER:$USER /opt/vision-pipeline
cp -r * /opt/vision-pipeline/
```

2. Skopiuj jednostki systemd:

```bash
cd /opt/vision-pipeline/config/systemd
sudo cp day_collector.service /etc/systemd/system/
sudo cp day_collector_start.timer /etc/systemd/system/
sudo cp day_collector_stop.service /etc/systemd/system/
sudo cp day_collector_stop.timer /etc/systemd/system/
sudo cp night_trainer.service /etc/systemd/system/
sudo cp night_trainer.timer /etc/systemd/system/
sudo cp train_classifier.service /etc/systemd/system/
sudo cp train_classifier.timer /etc/systemd/system/
```

3. Przeładuj systemd i włącz timery:

```bash
sudo systemctl daemon-reload

sudo systemctl enable day_collector_start.timer
sudo systemctl enable day_collector_stop.timer
sudo systemctl enable night_trainer.timer
sudo systemctl enable train_classifier.timer

sudo systemctl start day_collector_start.timer
sudo systemctl start day_collector_stop.timer
sudo systemctl start night_trainer.timer
sudo systemctl start train_classifier.timer
```

### 3. Co się wtedy dzieje

- O 07:00: `day_collector_start.timer` uruchamia `day_collector.service`, który startuje `python day_collector.py` i zbiera dane.
- O 15:00: `day_collector_stop.timer` uruchamia `day_collector_stop.service`, który wykonuje `systemctl stop day_collector.service`.
- O 15:01: `night_trainer.timer` uruchamia jednorazowo `night_trainer.service` (`python night_trainer.py`).
- O 15:10: `train_classifier.timer` uruchamia jednorazowo `train_classifier.service` (`python train_classifier.py`).

W efekcie akwizycja danych działa tylko między 07:00 a 15:00, a po 15:00 automatycznie rozpoczyna się pipeline analiz nocnych.
