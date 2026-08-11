"""
Mask denoising, background gating, feathering, and green spill suppression.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config import (
    BLUR_SIZE,
    FEATHER_BLUR_SIZE,
    MASK_FEATHERING,
    MIN_CONTOUR_AREA,
    MORPH_KERNEL_SIZE,
)


class MaskProcessor:
    """Clean raw color masks and blend foreground/background smoothly."""

    def __init__(
        self,
        blur_size: int = BLUR_SIZE,
        morph_kernel_size: int = MORPH_KERNEL_SIZE,
        min_contour_area: int = MIN_CONTOUR_AREA,
        feathering: bool = MASK_FEATHERING,
        feather_blur_size: int = FEATHER_BLUR_SIZE,
    ) -> None:
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self.morph_kernel_size = (
            morph_kernel_size if morph_kernel_size % 2 == 1 else morph_kernel_size + 1
        )
        self.min_contour_area = min_contour_area
        self.feathering = feathering
        self.feather_blur_size = (
            feather_blur_size if feather_blur_size % 2 == 1 else feather_blur_size + 1
        )
        self.background_gating = True
        self.bg_diff_threshold = 25
        self.spill_suppression = True
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        self._kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def process(
        self,
        raw_mask: np.ndarray,
        frame: Optional[np.ndarray] = None,
        background: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Denoise mask and optionally gate with background difference.

        Background gating: ignore green pixels that match the stored background
        (e.g. a green plant that was present during background capture).
        """
        if raw_mask.size == 0:
            return raw_mask

        mask = raw_mask.copy()

        if self.background_gating and frame is not None and background is not None:
            mask = self._apply_background_gate(mask, frame, background)

        blurred = cv2.GaussianBlur(mask, (self.blur_size, self.blur_size), 0)
        opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, self._kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._kernel_large)
        cleaned = self._remove_small_contours(closed)

        # Fill small holes inside the cloak silhouette
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self._kernel_large)
        return cleaned

    def _apply_background_gate(
        self,
        mask: np.ndarray,
        frame: np.ndarray,
        background: np.ndarray,
    ) -> np.ndarray:
        diff = cv2.absdiff(frame, background)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, changed = cv2.threshold(gray_diff, self.bg_diff_threshold, 255, cv2.THRESH_BINARY)
        changed = cv2.dilate(changed, self._kernel, iterations=2)
        return cv2.bitwise_and(mask, changed)

    def _remove_small_contours(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_contour_area:
                cv2.drawContours(output, [contour], -1, 255, cv2.FILLED)
        return output

    def apply_invisibility(
        self,
        frame: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        if not self.feathering:
            output = frame.copy()
            output[mask > 0] = background[mask > 0]
            return output

        alpha = cv2.GaussianBlur(mask, (self.feather_blur_size, self.feather_blur_size), 0)
        alpha = alpha.astype(np.float32) / 255.0
        alpha_3 = cv2.merge([alpha, alpha, alpha])

        frame_f = frame.astype(np.float32)
        bg_f = background.astype(np.float32)
        blended = bg_f * alpha_3 + frame_f * (1.0 - alpha_3)
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def suppress_green_spill(result: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Reduce green fringing on edges by desaturating near-boundary pixels.
        """
        edge = cv2.Canny(mask, 50, 150)
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=2)
        if not np.any(edge):
            return result

        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        edge_f = (edge > 0).astype(np.float32)
        # Pull saturation down on edge pixels to kill green spill
        hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - 0.35 * edge_f)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def mask_to_bgr(self, mask: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
