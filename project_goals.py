# GitHub Copilot — Finalna organizacja projektu i instrukcja wdrożenia

# Cel projektu:
# Zbudować modularny, stabilny pipeline do wykrywania, grupowania i rozpoznawania osób
# na podstawie twarzy, działający na Raspberry Pi lub PC, z pełnym podziałem na procesy:
# 1) Day Collector
# 2) Night Trainer
# 3) Train Classifier
# 4) (opcjonalnie) Day Recognizer

# Struktura repo:
# project/
#   day_collector.py
#   night_trainer.py
#   train_classifier.py
#   day_recognizer.py   (opcjonalnie)
#   config/
#   data/
#       faces_raw/
#       faces_clean/
#       embeddings/
#       clusters/
#       models/
#   DEPLOYMENT.md
#   README.md

# Opis procesów:

# 1. Day Collector (proces dzienny)
# - działa cały dzień na Raspberry Pi / PC
# - pobiera klatki z kamery
# - wykrywa twarze Haar Cascade
# - filtruje (rozmiar, blur, pozycja)
# - deduplikuje (phash)
# - zapisuje cropy twarzy do data/faces_raw/
# - jest maksymalnie lekki i stabilny

# 2. Night Trainer (proces nocny)
# - działa offline, batchowo
# - wczytuje data/faces_raw/
# - filtruje artefakty
# - liczy embeddingi (HOG + średnie kanałów)
# - wykonuje clustering DBSCAN
# - zapisuje embeddingi, etykiety, centroidy i posegregowane obrazy
# - nie dotyka kamery

# 3. Train Classifier (supervised learning)
# - wczytuje embeddingi i etykiety klastrów
# - ignoruje outliery (label = -1)
# - trenuje lekki klasyfikator:
#       * LinearSVC
#       * k-NN
#       * MLPClassifier
# - zapisuje model do data/models/classifier.pkl
# - zapisuje mapowanie cluster_id → person_id
# - pomija trening, jeśli jest za mało danych

# 4. Day Recognizer (opcjonalnie)
# - działa w czasie rzeczywistym
# - wykrywa twarz
# - liczy embedding
# - używa classifier.pkl lub centroidów
# - zwraca person_id lub "unknown"

# Finalne wdrożenie — jak uruchamiać:

# A) Ręcznie (najprostszy tryb)
# Rano:
#   python day_collector.py
# Wieczorem:
#   pkill -f day_collector.py
#   python night_trainer.py
#   python train_classifier.py
# Następnego dnia:
#   python day_recognizer.py

# B) Automatycznie (rekomendowane)
# - Day Collector jako systemd service (Restart=always)
# - Night Trainer jako systemd timer (np. 23:00)
# - Train Classifier jako drugi timer (np. 23:10)
# - Day Recognizer jako osobny proces, jeśli potrzebny

# Zasady dla Copilota:
# - sprawdzaj spójność nowych funkcji z powyższą architekturą
# - nie mieszaj odpowiedzialności między procesami
# - generuj kod modularny, lekki, CPU-friendly
# - utrzymuj kompatybilność z istniejącymi katalogami i formatami danych
# - nie przenoś logiki detekcji do Night Trainer ani logiki embeddingów do Day Collector
# - dbaj o stabilność i odporność na błędy

# Ten prompt definiuje finalną organizację projektu i sposób wdrożenia.
# Copilot powinien używać go jako kontekstu przy generowaniu dalszego kodu.
