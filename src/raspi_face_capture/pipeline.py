from __future__ import annotations

import time

import cv2

from .config import settings
from .detector import FaceDetector
from .duplicates import DuplicateFilter
from .filters import FaceQualityFilter
from .storage import FaceStorage


class CapturePipeline:
    def __init__(self) -> None:
        self.detector = FaceDetector()
        self.quality = FaceQualityFilter()
        self.duplicates = DuplicateFilter()
        self.storage = FaceStorage()

    def run(self) -> None:
        cap = cv2.VideoCapture(settings.CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Nie można otworzyć kamery")

        last_time = 0.0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                now = time.time()
                if now - last_time < settings.FRAME_INTERVAL_S:
                    continue
                last_time = now

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_h, frame_w = gray.shape

                faces = self.detector.detect(gray)
                for face in faces:
                    if self.quality.is_too_small(face):
                        continue

                    crop_bgr = face.crop(frame)
                    gray_crop = face.crop(gray)

                    if self.quality.is_blurry(gray_crop):
                        continue

                    if self.quality.is_profile_like(face, frame_w, frame_h):
                        continue

                    if self.duplicates.is_duplicate(crop_bgr):
                        continue

                    self.storage.save_face(crop_bgr)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
