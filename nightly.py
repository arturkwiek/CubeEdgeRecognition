"""Night Trainer: offline'owy pipeline do wieczornego przetwarzania twarzy.

Ten moduł realizuje drugi proces projektu:

1. Day Collector (już zaimplementowany) — działa na Raspberry Pi / PC,
   pobiera klatki z kamery, wykrywa twarze Haar Cascade,
   filtruje je (rozmiar, ostrość, pozycja), deduplikuje (phash),
   zapisuje cropy twarzy do katalogu data/faces_raw/.

2. Night Trainer (ten plik) — osobny proces uruchamiany wieczorem.
   Zadania:
   - wczytać obrazy z data/faces_raw/
   - odfiltrować artefakty i błędne detekcje
   - policzyć lekkie embeddingi twarzy (CPU-friendly, HOG + statystyki)
   - wykonać clustering (DBSCAN), aby pogrupować twarze według osób
   - zapisać wyniki:
		* data/faces_clean/ — poprawne twarze (skopiowane pliki, nie nowe zdjęcia)
		* data/embeddings/embeddings.npy — embeddingi (NumPy)
		* data/clusters/labels.npy + paths.txt — etykiety klastrów i ścieżki plików
		* data/models/centroids.npy + cluster_ids.npy — centroidy embeddingów

Wymagania spełnione przez ten plik:
- brak kodu kamery / VideoCapture (tylko offline batch processing),
- modularny kod: load_raw_faces(), filter_artifacts(), compute_embeddings(),
  cluster_faces(), save_results(),
- ostrożna obsługa błędów (pomijanie uszkodzonych plików, brak danych itp.),
- brak ciężkich modeli i GPU — tylko lekkie obliczenia na CPU,
- dodane logowanie przebiegu pipeline'u.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


LOGGER = logging.getLogger("night_trainer")


@dataclass
class NightConfig:
	"""Konfiguracja pipeline'u Night Trainer."""

	# Ścieżki wejścia/wyjścia
	input_dir: Path = Path("data/faces_raw")
	faces_clean_dir: Path = Path("data/faces_clean")
	embeddings_dir: Path = Path("data/embeddings")
	clusters_dir: Path = Path("data/clusters")
	models_dir: Path = Path("data/models")

	# Filtry jakości (drugi etap, można ustawić ostrzej niż w Day Collector)
	min_size: int = 80
	blur_threshold: float = 150.0  # wyższy próg = bardziej wymagający

	# Embedding (HOG)
	resize_width: int = 128
	resize_height: int = 128

	# Clustering (DBSCAN)
	dbscan_eps: float = 0.6
	dbscan_min_samples: int = 5


def setup_logging() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
	)


def load_raw_faces(cfg: NightConfig) -> List[Path]:
	"""Zbiera listę wszystkich obrazów z katalogu wejściowego.

	Obsługiwane są rozszerzenia .jpg, .jpeg, .png.
	"""

	if not cfg.input_dir.exists():
		LOGGER.warning("Katalog wejściowy nie istnieje: %s", cfg.input_dir)
		return []

	image_paths: List[Path] = []
	for ext in ("*.jpg", "*.jpeg", "*.png"):
		image_paths.extend(sorted(cfg.input_dir.glob(ext)))

	LOGGER.info("Znaleziono %d surowych wycinków twarzy", len(image_paths))
	return image_paths


def is_blurry(image_gray: np.ndarray, blur_threshold: float) -> bool:
	variance = cv2.Laplacian(image_gray, cv2.CV_64F).var()
	return variance < blur_threshold


def filter_artifacts(cfg: NightConfig, image_paths: Sequence[Path]) -> List[Path]:
	"""Drugi etap czyszczenia: odrzucanie rozmazanych / zbyt małych wycinków.

	Nie zapisuje nowych plików – tylko zwraca listę ścieżek zaakceptowanych obrazów.
	"""

	kept: List[Path] = []
	for path in image_paths:
		try:
			img_bgr = cv2.imread(str(path))
			if img_bgr is None:
				LOGGER.warning("Nie udało się wczytać obrazu: %s", path)
				continue

			h, w = img_bgr.shape[:2]
			if h < cfg.min_size or w < cfg.min_size:
				continue

			img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
			if is_blurry(img_gray, cfg.blur_threshold):
				continue

			kept.append(path)
		except Exception as exc:  # pylint: disable=broad-except
			LOGGER.error("Błąd przy przetwarzaniu %s: %s", path, exc)

	LOGGER.info(
		"Po dodatkowym czyszczeniu pozostało %d/%d twarzy",
		len(kept),
		len(image_paths),
	)
	return kept


def _compute_single_embedding(cfg: NightConfig, img_bgr: np.ndarray) -> np.ndarray:
	"""Prosty embedding oparty na HOG + średnich kolorów.

	Celowo lekki, CPU-friendly i bez zewnętrznych modeli.
	"""

	resized = cv2.resize(
		img_bgr,
		(cfg.resize_width, cfg.resize_height),
		interpolation=cv2.INTER_AREA,
	)
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

	# HOG z OpenCV (ustawienia rozsądne dla twarzy o małej rozdzielczości)
	hog = cv2.HOGDescriptor()
	hog_vec = hog.compute(gray)
	if hog_vec is None:
		hog_vec = np.zeros((0, 1), dtype=np.float32)

	# Proste statystyki koloru (średnie po kanałach)
	means = resized.mean(axis=(0, 1))  # B, G, R

	emb = np.concatenate([hog_vec.flatten(), means.astype(np.float32)])
	return emb.astype(np.float32)


def compute_embeddings(cfg: NightConfig, image_paths: Sequence[Path]) -> np.ndarray:
	"""Liczy embeddingi dla każdej ścieżki.

	Zwraca tablicę o kształcie (N, D). Błędne obrazy są pomijane.
	"""

	embeddings: List[np.ndarray] = []
	valid_paths: List[Path] = []

	for path in image_paths:
		try:
			img_bgr = cv2.imread(str(path))
			if img_bgr is None:
				LOGGER.warning("Nie udało się ponownie wczytać obrazu: %s", path)
				continue

			emb = _compute_single_embedding(cfg, img_bgr)
			embeddings.append(emb)
			valid_paths.append(path)
		except Exception as exc:  # pylint: disable=broad-except
			LOGGER.error("Błąd embeddingu dla %s: %s", path, exc)

	if not embeddings:
		LOGGER.warning("Nie udało się policzyć żadnych embeddingów")
		return np.empty((0, 0), dtype=np.float32)

	# Upewnij się, że ścieżki są zgodne
	if len(valid_paths) != len(image_paths):
		LOGGER.info(
			"Embeddingi policzono dla %d/%d obrazów (część odrzucona)",
			len(valid_paths),
			len(image_paths),
		)

	return np.vstack(embeddings)


def cluster_faces(cfg: NightConfig, embeddings: np.ndarray) -> np.ndarray:
	"""Klasteryzuje embeddingi metodą DBSCAN.

	Zwraca tablicę etykiet o długości N. Wartość -1 oznacza szum / outlier.
	"""

	if embeddings.size == 0:
		LOGGER.warning("Brak embeddingów do klasteryzacji")
		return np.empty((0,), dtype=int)

	dbscan = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples, metric="euclidean")
	labels: np.ndarray = dbscan.fit_predict(embeddings)

	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	LOGGER.info("DBSCAN: znaleziono %d klastrów (etykiet nie-ujemnych)", n_clusters)
	return labels


def _compute_centroids(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Liczy centroidy embeddingów dla każdego klastra (bez szumu)."""

	unique_labels = sorted(l for l in set(labels) if l != -1)
	if not unique_labels:
		return np.empty((0, embeddings.shape[1]), dtype=np.float32), np.empty((0,), dtype=int)

	centroids: List[np.ndarray] = []
	cluster_ids: List[int] = []

	for label in unique_labels:
		mask = labels == label
		cluster_embs = embeddings[mask]
		centroid = cluster_embs.mean(axis=0)
		centroids.append(centroid.astype(np.float32))
		cluster_ids.append(label)

	return np.vstack(centroids), np.asarray(cluster_ids, dtype=int)


def save_results(
	cfg: NightConfig,
	image_paths: Sequence[Path],
	embeddings: np.ndarray,
	labels: np.ndarray,
) -> None:
	"""Zapisuje wyniki: embeddingi, etykiety, centroidy, skopiowane obrazy."""

	if embeddings.size == 0 or labels.size == 0:
		LOGGER.warning("Brak danych do zapisania wyników")
		return

	cfg.faces_clean_dir.mkdir(parents=True, exist_ok=True)
	cfg.embeddings_dir.mkdir(parents=True, exist_ok=True)
	cfg.clusters_dir.mkdir(parents=True, exist_ok=True)
	cfg.models_dir.mkdir(parents=True, exist_ok=True)

	# 1) Embeddingi
	np.save(cfg.embeddings_dir / "embeddings.npy", embeddings)

	# 2) Etykiety + ścieżki
	np.save(cfg.clusters_dir / "labels.npy", labels)
	paths_txt = cfg.clusters_dir / "paths.txt"
	with paths_txt.open("w", encoding="utf-8") as f:
		for path in image_paths:
			f.write(str(path) + "\n")

	# 3) Centroidy / "lekki model"
	centroids, cluster_ids = _compute_centroids(embeddings, labels)
	if centroids.size > 0:
		np.save(cfg.models_dir / "centroids.npy", centroids)
		np.save(cfg.models_dir / "cluster_ids.npy", cluster_ids)

	# 4) Kopie czystych twarzy do faces_clean/ z podziałem na klastry
	#    (bez generowania nowych obrazów z kamery)
	for path, label in zip(image_paths, labels):
		if label == -1:
			# Szum można pominąć lub wrzucić do osobnego katalogu; tutaj pomijamy
			continue

		target_dir = cfg.faces_clean_dir / f"cluster_{label}"
		target_dir.mkdir(parents=True, exist_ok=True)
		target_path = target_dir / path.name

		# Kopiujemy plik binarnie; nie przetwarzamy obrazu
		try:
			data = path.read_bytes()
			target_path.write_bytes(data)
		except Exception as exc:  # pylint: disable=broad-except
			LOGGER.error("Błąd kopiowania %s -> %s: %s", path, target_path, exc)

	LOGGER.info("Zapisano wyniki Night Trainer w katalogu data/*")


def run_night_trainer() -> None:
	"""Główna funkcja uruchamiająca cały pipeline Night Trainer."""

	setup_logging()
	cfg = NightConfig()

	LOGGER.info("Start Night Trainer")

	raw_paths = load_raw_faces(cfg)
	if not raw_paths:
		LOGGER.warning("Brak surowych twarzy do przetworzenia")
		return

	clean_paths = filter_artifacts(cfg, raw_paths)
	if not clean_paths:
		LOGGER.warning("Po czyszczeniu nie zostały żadne twarze")
		return

	embeddings = compute_embeddings(cfg, clean_paths)
	if embeddings.size == 0:
		LOGGER.warning("Brak embeddingów – przerywam")
		return

	labels = cluster_faces(cfg, embeddings)
	if labels.size == 0:
		LOGGER.warning("Brak etykiet klastrów – przerywam")
		return

	save_results(cfg, clean_paths, embeddings, labels)
	LOGGER.info("Night Trainer zakończony")


if __name__ == "__main__":
	run_night_trainer()

