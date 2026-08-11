"""
Webcam capture and FPS tracking.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, TARGET_FPS


class Camera:
    """Wraps OpenCV VideoCapture with resolution and FPS helpers."""

    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps_start = time.perf_counter()
        self._fps_frame_count = 0
        self.fps: float = 0.0

    def open(self) -> None:
        """Open the webcam and apply resolution settings."""
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback without DirectShow (Linux/macOS)
            self._cap = cv2.VideoCapture(self.index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self.index}. "
                "Check that a camera is connected and not in use by another app."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        # Read back actual dimensions (driver may adjust)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame and update the FPS estimate."""
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ok, frame = self._cap.read()
        if ok:
            self._update_fps()
        return ok, frame

    def _update_fps(self) -> None:
        self._fps_frame_count += 1
        elapsed = time.perf_counter() - self._fps_start
        if elapsed >= 0.5:
            self.fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_start = time.perf_counter()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
