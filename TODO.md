# TODO / dalszy rozwój pipeline'u

## Integracja śledzenia eksperymentów

- Dodać prostą integrację z MLflow w:
  - `nightly.py` (logowanie parametrów DBSCAN, liczby twarzy, liczby klastrów),
  - `train_classifier.py` (logowanie typu klasyfikatora, liczby klas, prostych metryk jakości, zapis/artefakt modelu).
- Zdefiniować lokalny backend MLflow (na RPi lub na maszynie developerskiej) i ścieżkę artefaktów.

## Wersjonowanie danych i modeli

- Włączyć DVC w repozytorium (np. `dvc init`).
- Dodać pod kontrolę DVC co najmniej katalog `data/models/` (modele klasyfikatora) i ewentualnie `data/embeddings/`, `data/clusters/`.
- Skonfigurować zdalny storage DVC (S3 / Azure / SSH / dysk sieciowy) zgodny z wymaganiami bezpieczeństwa i RODO.

## Organizacja R&D

- Utrzymywać ten plik TODO w katalogu głównym repo jako prostą listę zadań rozwojowych.
- Gdy lista się rozrośnie, rozważyć przeniesienie szczegółowych notatek do osobnego katalogu dokumentacji (np. `docs/` lub `r_and_d/`), zostawiając tutaj jedynie skrócony roadmap.
