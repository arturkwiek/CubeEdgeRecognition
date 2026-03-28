from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2

from .config import settings


@dataclass
class DetectedFace:
    x: int
    y: int
    w: int
    h: int

    def crop(self, frame):
        return frame[self.y : self.y + self.h, self.x : self.x + self.w]


class FaceDetector:
    def __init__(self) -> None:
        cascade_path = settings.HAAR_CASCADE_PATH
        if not cascade_path:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, gray_frame) -> List[DetectedFace]:
        faces = self._cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(settings.MIN_FACE_SIZE, settings.MIN_FACE_SIZE),
        )
        return [DetectedFace(x=int(x), y=int(y), w=int(w), h=int(h)) for (x, y, w, h) in faces]
