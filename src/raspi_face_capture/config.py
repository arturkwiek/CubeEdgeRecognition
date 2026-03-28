from pathlib import Path


class Settings:
    """Konfiguracja pipeline'u zbierania twarzy na Raspberry Pi."""

    # Kamera
    CAMERA_INDEX: int = 0
    FRAME_INTERVAL_S: float = 1.5  # co ile sekund przetwarzamy klatkę

    # Detekcja twarzy
    HAAR_CASCADE_PATH: str = "haarcascade_frontalface_default.xml"  # jeśli pusty, użyj domyślnego z cv2
    MIN_FACE_SIZE: int = 80  # minimalna szerokość/wysokość twarzy w px

    # Filtry jakości
    BLUR_THRESHOLD: float = 100.0  # próg ostrości (Laplace variance)
    PROFILE_MARGIN_RATIO: float = 0.05  # margines od krawędzi kadru dla odrzucania profili

    # Duplikaty (perceptual hash)
    DUPLICATE_HASH_DISTANCE: int = 5

    # Zapisywanie
    OUTPUT_DIR: Path = Path("data/faces_raw")


settings = Settings()  # instancja domyślna
