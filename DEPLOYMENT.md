# Deployment / uruchamianie pipeline'u

Ten projekt składa się z trzech głównych procesów:

- Day Collector – day_collector.py (alias do main.py)
- Night Trainer – night_trainer.py (korzysta z nightly.py)
- Train Classifier – train_classifier.py
- (opcjonalnie) Day Recognizer – do zaimplementowania osobno

## A) Ręczne uruchamianie (najprostsze)

Przed pierwszym uruchomieniem zainstaluj zależności (szczegóły w README.md):

```bash
python -m venv .venv
source .venv/bin/activate    # Linux / Raspberry Pi
pip install -r requirements.txt
```

Na Windows użyj odpowiedniego skryptu aktywującego środowisko wirtualne.

1. Rano – start Day Collector (alias do głównego pipeline'u):

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

3. (Opcjonalnie) Następnego dnia – jeśli istnieje moduł rozpoznawania (Day Recognizer),
   możesz uruchomić osobny proces odpowiedzialny za rozpoznawanie w czasie rzeczywistym.

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
- `night_trainer.service` – Night Trainer (uruchamiany automatycznie po zatrzymaniu Day Collector)
- `train_classifier.service` – Train Classifier (uruchamiany automatycznie po Night Trainer)

Możesz je zainstalować ręcznie (jak poniżej) albo użyć skryptu `scripts/deploy_rpi_systemd.sh`,
który wykonuje za Ciebie kopiowanie projektu do `/opt/vision-pipeline`, instalację zależności
i podłączenie timerów systemd (tryb "fire and forget").

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
sudo cp train_classifier.service /etc/systemd/system/
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

### 2a. Szybki start na Raspberry Pi ("fire and forget")

Zamiast wykonywać ręcznie kroki z punktu 2, możesz z poziomu katalogu projektu
uruchomić przygotowany skrypt wdrożeniowy:

```bash
cd CubeEdgeRecognition
sudo bash scripts/deploy_rpi_systemd.sh
```

Skrypt:
- kopiuje projekt do `/opt/vision-pipeline`,
- (opcjonalnie) tworzy `.venv` i instaluje `requirements.txt`,
- kopiuje unity i timery z `config/systemd` do `/etc/systemd/system`,
- wykonuje `systemctl daemon-reload`, `enable` i `start` wszystkich timerów.

### 3. Co się wtedy dzieje

- O 07:00: `day_collector_start.timer` uruchamia `day_collector.service`, który startuje `python day_collector.py` i zbiera dane.
- O 15:00: `day_collector_stop.timer` uruchamia `day_collector_stop.service`, który wykonuje `systemctl stop day_collector.service`.
- Po poprawnym wykonaniu `day_collector_stop.service` systemd automatycznie uruchamia `night_trainer.service` (`python night_trainer.py`).
- Po poprawnym zakończeniu `night_trainer.service` systemd automatycznie uruchamia `train_classifier.service` (`python train_classifier.py`).

W efekcie akwizycja danych działa tylko między 07:00 a 15:00, a po 15:00 automatycznie,
sekwencyjnie wykonują się kolejne kroki pipeline'u (Night Trainer → Train Classifier),
bez sztywnych, "na rympał" timerów czasowych.

### 4. Podgląd logów i czasów wykonania

Night Trainer i Train Classifier logują początek i koniec pracy z dokładnym znacznikiem czasu.
Przykładowe polecenia diagnostyczne na Raspberry Pi / Ubuntu:

```bash
journalctl -u night_trainer.service -e
journalctl -u train_classifier.service -e
```

W logach (format `[asctime] [LEVEL] logger: message`) zobaczysz m.in. wpisy
`Start Night Trainer` / `Night Trainer zakończony` oraz
`Start Train Classifier` / `Train Classifier zakończony`, co pozwala śledzić realne czasy
trwania poszczególnych kroków pipeline'u.
