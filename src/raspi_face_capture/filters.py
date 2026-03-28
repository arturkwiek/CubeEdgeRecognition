from __future__ import annotations

from dataclasses import dataclass

import cv2

from .config import settings
from .detector import DetectedFace


@dataclass
class FaceQualityFilter:
    """Zestaw filtrów jakości dla wyciętych twarzy."""

    blur_threshold: float = settings.BLUR_THRESHOLD
    min_face_size: int = settings.MIN_FACE_SIZE
    profile_margin_ratio: float = settings.PROFILE_MARGIN_RATIO

    def is_too_small(self, face: DetectedFace) -> bool:
        return face.w < self.min_face_size or face.h < self.min_face_size

    def is_blurry(self, gray_crop) -> bool:
        fm = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
        return fm < self.blur_threshold

    def is_profile_like(self, face: DetectedFace, frame_width: int, frame_height: int) -> bool:
        margin = int(self.profile_margin_ratio * frame_width)
        too_left = face.x <= margin
        too_right = face.x + face.w >= frame_width - margin
        return too_left or too_right
