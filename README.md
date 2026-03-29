# Raspberry Pi Face Capture Pipeline

Lekki, modularny pipeline do dziennego zbierania wycinków twarzy na Raspberry Pi / PC
oraz nocnego przetwarzania i trenowania klasyfikatora twarzy.

W ciągu dnia działa **Day Collector** (kamera → wykrywanie twarzy → filtrowanie → deduplikacja → zapis cropów),
wieczorem uruchamiane są **Night Trainer** (embeddingi + clustering) i **Train Classifier** (trening modelu).

Główne katalogi danych:

- data/faces_raw/   – surowe cropy twarzy z Day Collector,
- data/faces_clean/ – dodatkowo oczyszczone twarze (Night Trainer),
- data/embeddings/  – embeddingi twarzy,
- data/clusters/    – etykiety klastrów i ścieżki,
- data/models/      – centroidy i/lub klasyfikator.

Szczegóły wdrożenia (systemd, Raspberry Pi) znajdziesz w pliku DEPLOYMENT.md.

## Instalacja

W katalogu projektu:

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# Linux / Raspberry Pi
source .venv/bin/activate

pip install -r requirements.txt
```

Na Raspberry Pi możesz też użyć systemowego Pythona zamiast wirtualnego środowiska,
o ile wszystkie wymagane pakiety są zainstalowane globalnie.

## Szybki start (tryb demo)

Proste uruchomienie tylko dziennego zbierania twarzy:

```bash
python main.py
```

Domyślnie wycinki twarzy zapisywane są w katalogu `data/faces_raw`.
Ustawienia (interwał czasowy, progi jakości itp.) można zmieniać w `src/raspi_face_capture/config.py`.

## Pełny pipeline (ręcznie)

1. **Day Collector – zbieranie w dzień**

	```bash
	python day_collector.py
	```

2. **Night Trainer – nocne przetwarzanie**

	```bash
	python night_trainer.py
	```

3. **Train Classifier – trenowanie klasyfikatora**

	```bash
	python train_classifier.py
	```

Po konfiguracji usług systemowych na Raspberry Pi te kroki mogą być
uruchamiane automatycznie – zobacz DEPLOYMENT.md.

## Monitorowanie i logi (systemd)

Jeżeli korzystasz z automatyzacji opartej o systemd, logi Night Trainer
i Train Classifier możesz podejrzeć za pomocą:

```bash
journalctl -u night_trainer.service -e
journalctl -u train_classifier.service -e
```

W logach znajdziesz m.in. wpisy `Start Night Trainer` / `Night Trainer zakończony`
oraz `Start Train Classifier` / `Train Classifier zakończony`, co pozwala śledzić
realne czasy działania poszczególnych kroków pipeline'u.
