"""
Background reference capture and persistence.

Uses median stacking across multiple frames so transient motion
(e.g. a person briefly in frame) does not pollute the reference.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from camera import Camera
from config import (
    BACKGROUND_CAPTURE_DELAY_MS,
    BACKGROUND_CAPTURE_FRAMES,
    BACKGROUND_COUNTDOWN_SECONDS,
    BACKGROUND_FILE,
)


class BackgroundManager:
    """Captures, stores, and reloads the static background image."""

    def __init__(self) -> None:
        self.background: Optional[np.ndarray] = None

    @property
    def is_ready(self) -> bool:
        return self.background is not None

    def capture_with_countdown(
        self,
        camera: Camera,
        window_name: str,
        countdown: int = BACKGROUND_COUNTDOWN_SECONDS,
    ) -> bool:
        """
        Show a countdown, then collect frames and build a median background.

        Returns True if capture succeeded.
        """
        for remaining in range(countdown, 0, -1):
            ok, frame = camera.read()
            if not ok or frame is None:
                return False

            display = frame.copy()
            self._draw_countdown(display, remaining)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False

            time.sleep(1.0)

        frames = self._collect_frames(camera, BACKGROUND_CAPTURE_FRAMES)
        if len(frames) < 3:
            return False

        self.background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
        self.save(BACKGROUND_FILE)
        return True

    def _collect_frames(self, camera: Camera, count: int) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        for _ in range(count):
            ok, frame = camera.read()
            if ok and frame is not None:
                frames.append(frame.copy())
            time.sleep(BACKGROUND_CAPTURE_DELAY_MS / 1000.0)
        return frames

    def recapture(self, camera: Camera, window_name: str) -> bool:
        """Recapture background with a short countdown."""
        return self.capture_with_countdown(camera, window_name, countdown=3)

    def build_from_frames(self, frames: List[np.ndarray]) -> bool:
        """Build median background from a list of BGR frames."""
        if len(frames) < 3:
            return False
        self.background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
        self.save()
        return True

    def save(self, path=BACKGROUND_FILE) -> None:
        if self.background is not None:
            cv2.imwrite(str(path), self.background)

    def load(self, path=BACKGROUND_FILE) -> bool:
        img = cv2.imread(str(path))
        if img is None:
            return False
        self.background = img
        return True

    def resize_to_match(self, frame: np.ndarray) -> np.ndarray:
        """Ensure background matches the current frame size."""
        if self.background is None:
            raise RuntimeError("Background has not been captured yet.")

        if self.background.shape[:2] != frame.shape[:2]:
            return cv2.resize(
                self.background,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return self.background

    @staticmethod
    def _draw_countdown(frame: np.ndarray, seconds: int) -> None:
        text = f"Capturing background in {seconds}..."
        cv2.putText(
            frame,
            text,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Keep the scene clear of the green cloth.",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
