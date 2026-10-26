"""
Unified invisibility engine — ties detection, masking, and compositing together.

Used by both the Streamlit web app and the legacy OpenCV desktop app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from background import BackgroundManager
from color_detector import ColorDetector
from mask_processor import MaskProcessor


@dataclass
class EngineSettings:
    """Runtime toggles synced from the Streamlit sidebar."""

    effect_enabled: bool = False
    mode: str = "cloak"  # cloak | mask | debug
    feathering: bool = True
    background_gating: bool = True
    dual_color_space: bool = True
    spill_suppression: bool = True
    bg_diff_threshold: int = 25
    hue_tolerance: int = 15
    adaptive_mode: bool = True
    min_contour_area: int = 500
    morph_kernel: int = 5
    feather_blur: int = 15


@dataclass
class CloakEngine:
    """Stateful processor shared between UI and video callback."""

    detector: ColorDetector = field(default_factory=ColorDetector)
    mask_processor: MaskProcessor = field(default_factory=MaskProcessor)
    background_mgr: BackgroundManager = field(default_factory=BackgroundManager)
    settings: EngineSettings = field(default_factory=EngineSettings)

    # Background capture buffer (filled by video callback)
    capture_buffer: List[np.ndarray] = field(default_factory=list)
    is_capturing_background: bool = False
    capture_target_frames: int = 30
    effect_enabled: bool = False

    # Latest frame for calibration snapshot
    last_frame: Optional[np.ndarray] = None
    fps: float = 0.0
    _fps_count: int = 0
    _fps_t0: float = 0.0

    @property
    def background_ready(self) -> bool:
        return self.background_mgr.is_ready

    def sync_settings(self, settings: EngineSettings) -> None:
        self.settings = settings
        self.effect_enabled = settings.effect_enabled
        self.detector.hue_tolerance = settings.hue_tolerance
        self.detector.adaptive_mode = settings.adaptive_mode
        if self.detector.samples:
            self.detector._recompute_range()
        self.mask_processor.feathering = settings.feathering
        self.mask_processor.min_contour_area = settings.min_contour_area
        self.mask_processor.morph_kernel_size = settings.morph_kernel
        self.mask_processor.feather_blur_size = settings.feather_blur
        self.mask_processor.background_gating = settings.background_gating
        self.mask_processor.bg_diff_threshold = settings.bg_diff_threshold
        self.mask_processor.spill_suppression = settings.spill_suppression

    def start_background_capture(self, target_frames: int = 30) -> None:
        self.capture_buffer.clear()
        self.is_capturing_background = True
        self.capture_target_frames = target_frames

    def finalize_background_capture(self) -> bool:
        self.is_capturing_background = False
        if len(self.capture_buffer) < 5:
            return False
        stacked = np.stack(self.capture_buffer, axis=0)
        self.background_mgr.background = np.median(stacked, axis=0).astype(np.uint8)
        self.background_mgr.save()
        self.capture_buffer.clear()
        return True

    def sample_patch(self, frame: np.ndarray, x: int, y: int, radius: int = 2) -> Optional[Tuple[int, int, int]]:
        return self.detector.sample_patch(frame, x, y, radius)

    def process(self, frame: np.ndarray) -> np.ndarray:
        import time

        self.last_frame = frame.copy()

        if self.is_capturing_background:
            self.capture_buffer.append(frame.copy())
            if len(self.capture_buffer) >= self.capture_target_frames:
                self.finalize_background_capture()
            return self._draw_banner(frame, "Capturing background...", (0, 200, 255))

        if not self.effect_enabled:
            return frame.copy()

        if not self.background_ready:
            return self._draw_banner(frame, "Capture background first (sidebar)", (0, 165, 255))

        background = self.background_mgr.resize_to_match(frame)
        raw_mask = self.detector.create_mask(
            frame,
            use_lab=self.settings.dual_color_space,
        )
        mask = self.mask_processor.process(raw_mask, frame, background)
        result = self.mask_processor.apply_invisibility(frame, background, mask)

        if self.settings.spill_suppression:
            result = self.mask_processor.suppress_green_spill(result, mask)

        # FPS
        now = time.perf_counter()
        if self._fps_t0 == 0.0:
            self._fps_t0 = now
        self._fps_count += 1
        if now - self._fps_t0 >= 0.5:
            self.fps = self._fps_count / (now - self._fps_t0)
            self._fps_count = 0
            self._fps_t0 = now

        if self.settings.mode == "mask":
            out = self.mask_processor.mask_to_bgr(mask)
        elif self.settings.mode == "debug":
            mask_bgr = self.mask_processor.mask_to_bgr(mask)
            h, w = frame.shape[:2]
            combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
            combined[:, :w] = frame
            combined[:, w : 2 * w] = mask_bgr
            combined[:, 2 * w :] = result
            out = combined
        else:
            out = result

        cv2.putText(
            out,
            f"FPS: {self.fps:.1f}",
            (10, out.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return out

    @staticmethod
    def _draw_banner(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> np.ndarray:
        out = frame.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(out, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        return out
