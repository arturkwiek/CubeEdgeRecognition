from __future__ import annotations

from pathlib import Path
import time

import cv2

from .config import settings


class FaceStorage:
    def __init__(self) -> None:
        self.output_dir: Path = settings.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_face(self, crop_bgr) -> Path:
        ts = int(time.time())
        filename = self.output_dir / f"face_{ts}.jpg"
        cv2.imwrite(str(filename), crop_bgr)
        return filename
