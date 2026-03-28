from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
from PIL import Image
import imagehash

from .config import settings


@dataclass
class DuplicateFilter:
    """Filtrowanie prawie identycznych twarzy przy użyciu perceptual hash."""

    distance_threshold: int = settings.DUPLICATE_HASH_DISTANCE
    _last_hash: Optional[imagehash.ImageHash] = field(default=None, init=False, repr=False)

    def is_duplicate(self, crop_bgr) -> bool:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        current_hash = imagehash.phash(pil_img)

        if self._last_hash is None:
            self._last_hash = current_hash
            return False

        distance = abs(current_hash - self._last_hash)
        if distance <= self.distance_threshold:
            return True

        self._last_hash = current_hash
        return False
