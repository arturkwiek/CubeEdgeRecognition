# GitHub Copilot — pełny kontekst projektu (Train Classifier)
#
# Ten plik implementuje Krok 3 pipeline’u: trenowanie lekkiego klasyfikatora
# na embeddingach twarzy wygenerowanych przez Night Trainer.
#
# Architektura projektu:
# - Day Collector: zbiera cropy twarzy do faces_raw/
# - Night Trainer: filtruje, liczy embeddingi (HOG + średnie kanałów), robi clustering DBSCAN
# - Train Classifier (ten plik): trenuje klasyfikator wieloklasowy na embeddingach
#
# Ten moduł działa jako osobny proces. Nie dotyka kamery, nie liczy embeddingów,
# nie wykonuje clusteringu. Korzysta wyłącznie z wyników Night Trainer.
#
# Wejście:
# - data/embeddings/embeddings.npy  → macierz (N, D)
# - data/clusters/labels.npy        → etykiety klastrów (DBSCAN)
# - data/clusters/paths.txt         → ścieżki obrazów (opcjonalnie)
#
# Zasady:
# - ignorujemy outliery (label = -1)
# - każdy klaster != -1 to jedna osoba
# - embeddingi są cechami wejściowymi X
# - etykiety klastrów są y
#
# Zadania modułu:
#
# 1. load_training_data(cfg)
#    - wczytuje embeddings.npy i labels.npy
#    - odrzuca outliery (label = -1)
#    - zwraca X (float32) i y (int)
#    - loguje liczbę klas i próbek
#
# 2. train_classifier(cfg, X, y)
#    - trenuje lekki klasyfikator wieloklasowy
#    - wspierane modele:
#         * SVM (LinearSVC)
#         * k-NN (KNeighborsClassifier)
#         * MLPClassifier (mała sieć)
#    - wybór modelu przez cfg.classifier_type
#    - zwraca wytrenowany model
#
# 3. save_classifier(cfg, model, label_mapping)
#    - zapisuje model do data/models/classifier.pkl
#    - zapisuje mapowanie klas (cluster_id → person_id) do JSON
#    - loguje rozmiar modelu i liczbę klas
#
# 4. run_train_classifier()
#    - ustawia logging (INFO)
#    - tworzy ClassifierConfig
#    - wywołuje load_training_data → train_classifier → save_classifier
#    - obsługuje warunki brzegowe:
#         brak embeddingów
#         brak klastrów != -1
#         tylko jedna klasa (opcjonalnie pomija trenowanie)
#
# Wymagania:
# - numpy
# - scikit-learn
# - pickle / joblib
# - logging
#
# Wskazówki dla Copilota:
# - generuj kod modularny, z funkcjami dokładnie jak opisane powyżej
# - nie mieszaj logiki z Night Trainer ani Day Collector
# - nie licz embeddingów — używaj tylko plików z dysku
# - klasyfikator ma być lekki i CPU-friendly
# - dbaj o czytelne logowanie i obsługę błędów
# - nie twórz klas tam, gdzie wystarczą funkcje
# - ClassifierConfig ma zawierać ścieżki i parametry modelu
#
# Cel:
# Copilot ma wygenerować czysty, stabilny moduł supervised learning
# do trenowania klasyfikatora wieloklasowego na embeddingach twarzy.

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


LOGGER = logging.getLogger("train_classifier")


ClassifierType = Literal["svm", "knn", "mlp"]


@dataclass
class ClassifierConfig:
	"""Konfiguracja treningu klasyfikatora na embeddingach twarzy."""

	embeddings_path: Path = Path("data/embeddings/embeddings.npy")
	labels_path: Path = Path("data/clusters/labels.npy")

	# Wyjścia
	models_dir: Path = Path("data/models")
	classifier_filename: str = "classifier.pkl"
	label_mapping_filename: str = "label_mapping.json"

	# Typ klasyfikatora: "svm", "knn" lub "mlp"
	classifier_type: ClassifierType = "svm"

	# Parametry modeli
	knn_n_neighbors: int = 5
	mlp_hidden_layer_sizes: Tuple[int, ...] = (64,)
	mlp_max_iter: int = 500


def setup_logging() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
	)


def load_training_data(cfg: ClassifierConfig) -> Tuple[np.ndarray, np.ndarray]:
	"""Wczytuje embeddingi i etykiety, odrzuca outliery (label = -1).

	Zwraca X (float32) oraz y (int64).
	"""

	if not cfg.embeddings_path.exists() or not cfg.labels_path.exists():
		LOGGER.warning(
			"Brak plików embeddings/labels (%s, %s)",
			cfg.embeddings_path,
			cfg.labels_path,
		)
		return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

	try:
		X = np.load(cfg.embeddings_path)
		y = np.load(cfg.labels_path)
	except Exception as exc:  # pylint: disable=broad-except
		LOGGER.error("Błąd wczytywania danych treningowych: %s", exc)
		return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

	if X.shape[0] != y.shape[0]:
		LOGGER.error(
			"Niezgodne liczby próbek: embeddings=%d, labels=%d",
			X.shape[0],
			y.shape[0],
		)
		return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

	mask = y != -1
	X_clean = X[mask].astype(np.float32, copy=False)
	y_clean = y[mask].astype(np.int64, copy=False)

	if X_clean.size == 0:
		LOGGER.warning("Po odrzuceniu outlierów (label=-1) nie ma danych treningowych")
		return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

	unique_labels = np.unique(y_clean)
	LOGGER.info(
		"Załadowano dane treningowe: %d próbek, %d klas (etykiety: %s)",
		X_clean.shape[0],
		unique_labels.size,
		unique_labels,
	)
	return X_clean, y_clean


def _build_classifier(cfg: ClassifierConfig) -> object:
	"""Tworzy niewytrenowany klasyfikator na podstawie typu w konfiguracji."""

	if cfg.classifier_type == "svm":
		# LinearSVC jest lekki i dobrze działa na embeddingach
		return LinearSVC()

	if cfg.classifier_type == "knn":
		return KNeighborsClassifier(n_neighbors=cfg.knn_n_neighbors, n_jobs=-1)

	if cfg.classifier_type == "mlp":
		return MLPClassifier(
			hidden_layer_sizes=cfg.mlp_hidden_layer_sizes,
			max_iter=cfg.mlp_max_iter,
		)

	raise ValueError(f"Nieobsługiwany typ klasyfikatora: {cfg.classifier_type}")


def train_classifier(cfg: ClassifierConfig, X: np.ndarray, y: np.ndarray) -> object:
	"""Trenuje wybrany klasyfikator wieloklasowy.

	Zwraca wytrenowany model lub None, jeśli trening został pominięty.
	"""

	if X.size == 0 or y.size == 0:
		LOGGER.warning("Brak danych do treningu klasyfikatora")
		return None

	unique_labels = np.unique(y)
	if unique_labels.size < 2:
		LOGGER.warning(
			"Znaleziono tylko jedną klasę (%s) – pomijam trening klasyfikatora",
			unique_labels,
		)
		return None

	clf = _build_classifier(cfg)
	LOGGER.info(
		"Trenowanie klasyfikatora typu '%s' na %d próbkach, %d cechach",
		cfg.classifier_type,
		X.shape[0],
		X.shape[1],
	)

	clf.fit(X, y)
	LOGGER.info("Trenowanie zakończone")
	return clf


def _build_label_mapping(y: np.ndarray) -> dict:
	"""Buduje mapowanie cluster_id → person_id (0..K-1)."""

	unique_labels = sorted(int(l) for l in set(y) if l != -1)
	mapping = {cluster_id: idx for idx, cluster_id in enumerate(unique_labels)}
	return mapping


def save_classifier(
	cfg: ClassifierConfig,
	model: object,
	label_mapping: dict,
) -> None:
	"""Zapisuje wytrenowany model i mapowanie etykiet na dysk."""

	if model is None:
		LOGGER.warning("Brak modelu do zapisania – pomijam save_classifier")
		return

	cfg.models_dir.mkdir(parents=True, exist_ok=True)

	model_path = cfg.models_dir / cfg.classifier_filename
	mapping_path = cfg.models_dir / cfg.label_mapping_filename

	try:
		with model_path.open("wb") as f:
			pickle.dump(model, f)
	except Exception as exc:  # pylint: disable=broad-except
		LOGGER.error("Nie udało się zapisać modelu do %s: %s", model_path, exc)
		return

	try:
		with mapping_path.open("w", encoding="utf-8") as f:
			json.dump(label_mapping, f, ensure_ascii=False, indent=2)
	except Exception as exc:  # pylint: disable=broad-except
		LOGGER.error("Nie udało się zapisać mapowania etykiet do %s: %s", mapping_path, exc)
		return

	try:
		size_bytes = model_path.stat().st_size
	except OSError:
		size_bytes = -1

	LOGGER.info(
		"Zapisano klasyfikator do %s (rozmiar: %d B), liczba klas: %d",
		model_path,
		size_bytes,
		len(label_mapping),
	)


def run_train_classifier() -> None:
	"""Główna funkcja orkiestrująca trening klasyfikatora."""

	setup_logging()
	cfg = ClassifierConfig()

	LOGGER.info("Start Train Classifier")

	X, y = load_training_data(cfg)
	if X.size == 0 or y.size == 0:
		LOGGER.warning("Brak danych treningowych – kończę")
		return

	label_mapping = _build_label_mapping(y)
	if not label_mapping:
		LOGGER.warning("Brak więcej niż jednej klasy – kończę")
		return

	model = train_classifier(cfg, X, y)
	if model is None:
		LOGGER.warning("Model nie został wytrenowany – kończę")
		return

	save_classifier(cfg, model, label_mapping)
	LOGGER.info("Train Classifier zakończony")


if __name__ == "__main__":
	run_train_classifier()

