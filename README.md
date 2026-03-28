# Raspberry Pi Face Capture Pipeline

Lekki pipeline do dziennego zbierania wycinków twarzy na Raspberry Pi:

- wykrywanie twarzy (Haar Cascade / OpenCV),
- zapisywanie tylko wykrojonych twarzy (crop),
- odrzucanie zbyt małych, rozmazanych i "profilowych" ujęć,
- odrzucanie duplikatów przy użyciu perceptual hash.

## Instalacja

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

Na Raspberry Pi możesz użyć systemowego Pythona / wirtualnego środowiska i dostosować ścieżki.

## Uruchomienie

```bash
python main.py
```

Domyślnie wycinki twarzy zapisywane są w katalogu `data/faces_raw`. Ustawienia (interwał czasowy, progi jakości itp.) można zmieniać w `src/raspi_face_capture/config.py`.
