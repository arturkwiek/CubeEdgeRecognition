# Eksperyment: system akwizycji i analizy twarzy

Ten projekt realizuje **eksperyment badawczy** polegający na automatycznej akwizycji oraz analizie obrazów twarzy przy użyciu kamery podłączonej do Raspberry Pi.

---

## Komunikat dla uczestników

**Stając przed tą kamerą wyrażasz zgodę na uczestnictwo w eksperymencie opisanym poniżej.**

Jeżeli nie wyrażasz zgody, **nie stawaj w polu widzenia kamery**.

---

## Cel projektu

- Zbadanie możliwości **lokalnego** (on-device) wykrywania, grupowania i rozpoznawania twarzy na niskomocowym urządzeniu (Raspberry Pi).
- Opracowanie lekkiego, modularnego pipeline’u, który:
  - w ciągu dnia zbiera wycinki twarzy (Day Collector),
  - po zakończeniu akwizycji wykonuje offline’owe przetwarzanie (Night Trainer, Train Classifier),
  - **nie wysyła danych poza urządzenie**.

---

## Funkcjonalności systemu

1. **Dzienne zbieranie danych (Day Collector)**
  - automatyczne przechwytywanie obrazu z kamery w godzinach **07:00–15:00**, zgodnie z harmonogramem opisanym w DEPLOYMENT.md,
  - wykrywanie twarzy (Haar Cascade),
  - zapisywanie wycinków twarzy (cropów) do katalogu `data/faces_raw/`,
  - odrzucanie zbyt małych, rozmazanych lub „profilowych” ujęć,
  - odrzucanie prawie identycznych kolejnych ujęć (deduplikacja).

2. **Nocne przetwarzanie (Night Trainer)**
  - dodatkowe czyszczenie z artefaktów,
  - wyznaczanie wektorów cech (embeddingów) dla każdej twarzy,
  - grupowanie podobnych twarzy metodą klasteryzacji,
  - zapisywanie posegregowanych twarzy oraz wyników obliczeń (embeddingi, etykiety klastrów, centroidy) wyłącznie lokalnie.

3. **Trenowanie klasyfikatora (Train Classifier)**
  - uczenie lekkiego modelu rozpoznającego zgrupowane osoby na podstawie embeddingów,
  - zapisywanie wytrenowanego modelu lokalnie na Raspberry Pi,
  - możliwość wykorzystania modelu w kolejnym, osobnym kroku (np. moduł rozpoznawania w czasie rzeczywistym).

4. **Przetwarzanie tylko lokalne**
  - żaden z procesów nie wysyła danych (obrazy, embeddingi, modele) poza urządzenie,
  - wszystkie operacje odbywają się na tym samym Raspberry Pi.

---

## Zakres i miejsce przetwarzania danych

- Obrazy twarzy oraz wszystkie pochodne dane (embeddingi, klastry, modele) są **gromadzone wyłącznie lokalnie na Raspberry Pi**.
- Całe przetwarzanie (trenowanie modeli, walidacja, analizy) odbywa się **na tym samym urządzeniu**.
- Dane **nie są przesyłane do chmury ani na żadne zewnętrzne serwery**.

---

## Czas trwania i usuwanie danych

- Dane gromadzone są **tylko na czas trwania eksperymentu**.
- Po zakończeniu eksperymentu wszystkie:
  - obrazy twarzy,
  - zbiory danych pośrednich (faces_raw, faces_clean, embeddings, clusters),
  - wytrenowane modele (models),

zostaną **trwale usunięte z Raspberry Pi** zgodnie z przyjętą procedurą.

---

## Uwagi organizacyjne

- Ten system jest przeznaczony **wyłącznie do celów eksperymentalnych / badawczych** w ramach pracy.
- Przed uruchomieniem w środowisku firmowym upewnij się, że:
  - spełnione są wewnętrzne wymogi prawne i RODO,
  - link w komunikacie prowadzi do pełnego opisu eksperymentu (regulamin / zgoda),
  - fizycznie w pobliżu kamery umieszczona jest informacja z powyższym komunikatem.
