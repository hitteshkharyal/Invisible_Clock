"""
HSV + LAB green cloth detection with patch sampling and adaptive calibration.

Accuracy improvements over single-pixel HSV:
- 5x5 patch median sampling (robust to sensor noise)
- Dual HSV + LAB color-space intersection
- Bilateral pre-filter before color conversion
- Hue wrapping for OpenCV's 0-179 range
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    ADAPTIVE_HUE_MARGIN,
    ADAPTIVE_MODE,
    ADAPTIVE_SAT_MARGIN,
    ADAPTIVE_VAL_MARGIN,
    CALIBRATION_FILE,
    DEFAULT_LOWER_GREEN,
    DEFAULT_UPPER_GREEN,
    HUE_TOLERANCE,
    LAB_A_MAX,
    LAB_A_MIN,
    LAB_B_MAX,
    LAB_B_MIN,
    SATURATION_MARGIN,
    SATURATION_MIN,
    VALUE_MARGIN,
    VALUE_MIN,
)


HSVTuple = Tuple[int, int, int]
LABTuple = Tuple[int, int, int]


class ColorDetector:
    """Detect green pixels using configurable HSV bounds (+ optional LAB)."""

    def __init__(self) -> None:
        self.lower_green = np.array(DEFAULT_LOWER_GREEN, dtype=np.uint8)
        self.upper_green = np.array(DEFAULT_UPPER_GREEN, dtype=np.uint8)
        self.lower_lab = np.array([LAB_A_MIN, LAB_B_MIN, 0], dtype=np.uint8)
        self.upper_lab = np.array([LAB_A_MAX, LAB_B_MAX, 255], dtype=np.uint8)
        self.samples: List[HSVTuple] = []
        self.lab_samples: List[LABTuple] = []
        self.last_sample_bgr: Optional[Tuple[int, int, int]] = None
        self.hue_tolerance = HUE_TOLERANCE
        self.adaptive_mode = ADAPTIVE_MODE
        self._wraps_hue = False

        if CALIBRATION_FILE.exists():
            self.load_calibration()

    # ------------------------------------------------------------------
    # Pre-filter (reduces noise, keeps edges)
    # ------------------------------------------------------------------

    @staticmethod
    def _prefilter(frame: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)

    # ------------------------------------------------------------------
    # Calibration — patch-based sampling
    # ------------------------------------------------------------------

    def sample_patch(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        radius: int = 2,
    ) -> Optional[HSVTuple]:
        """
        Sample a (2*radius+1)^2 patch and use median HSV/LAB values.
        Much more stable than a single pixel click.
        """
        h, w = frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None

        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        patch = frame[y0:y1, x0:x1]

        if patch.size == 0:
            return None

        filtered = self._prefilter(patch)
        hsv_patch = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        lab_patch = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)

        h_med = int(np.median(hsv_patch[:, :, 0]))
        s_med = int(np.median(hsv_patch[:, :, 1]))
        v_med = int(np.median(hsv_patch[:, :, 2]))
        sample = (h_med, s_med, v_med)

        a_med = int(np.median(lab_patch[:, :, 1]))
        b_med = int(np.median(lab_patch[:, :, 2]))
        l_med = int(np.median(lab_patch[:, :, 0]))
        lab_sample = (l_med, a_med, b_med)

        self.samples.append(sample)
        self.lab_samples.append(lab_sample)
        self.last_sample_bgr = tuple(int(c) for c in frame[y, x])
        self._recompute_range()
        return sample

    def sample_at(self, frame: np.ndarray, x: int, y: int) -> Optional[HSVTuple]:
        """Backward-compatible single-point alias."""
        return self.sample_patch(frame, x, y, radius=2)

    def reset_samples(self) -> None:
        self.samples.clear()
        self.lab_samples.clear()
        self.last_sample_bgr = None
        self.lower_green = np.array(DEFAULT_LOWER_GREEN, dtype=np.uint8)
        self.upper_green = np.array(DEFAULT_UPPER_GREEN, dtype=np.uint8)
        self.lower_lab = np.array([LAB_A_MIN, LAB_B_MIN, 0], dtype=np.uint8)
        self.upper_lab = np.array([LAB_A_MAX, LAB_B_MAX, 255], dtype=np.uint8)
        self._wraps_hue = False

    def _recompute_range(self) -> None:
        if not self.samples:
            return
        if self.adaptive_mode:
            self._compute_adaptive_range()
        else:
            self._compute_tolerance_range(self.samples[-1])
        self._compute_lab_range()

    def _compute_tolerance_range(self, sample: HSVTuple) -> None:
        h, s, v = sample
        h_low = max(0, h - self.hue_tolerance)
        h_high = min(179, h + self.hue_tolerance)
        s_low = max(SATURATION_MIN, s - SATURATION_MARGIN)
        v_low = max(VALUE_MIN, v - VALUE_MARGIN)

        self.lower_green = np.array([h_low, s_low, v_low], dtype=np.uint8)
        self.upper_green = np.array([h_high, 255, 255], dtype=np.uint8)
        self._wraps_hue = False

    def _compute_adaptive_range(self) -> None:
        hues = [s[0] for s in self.samples]
        sats = [s[1] for s in self.samples]
        vals = [s[2] for s in self.samples]

        h_min, h_max = min(hues), max(hues)
        s_min, s_max = min(sats), max(sats)
        v_min, v_max = min(vals), max(vals)

        h_low = max(0, h_min - ADAPTIVE_HUE_MARGIN - self.hue_tolerance)
        h_high = min(179, h_max + ADAPTIVE_HUE_MARGIN + self.hue_tolerance)
        s_low = max(SATURATION_MIN, s_min - ADAPTIVE_SAT_MARGIN)
        s_high = min(255, s_max + ADAPTIVE_SAT_MARGIN)
        v_low = max(VALUE_MIN, v_min - ADAPTIVE_VAL_MARGIN)
        v_high = min(255, v_max + ADAPTIVE_VAL_MARGIN)

        hue_span = h_max - h_min
        if hue_span > 90 and len(hues) >= 2:
            self._wraps_hue = True
            shifted = [(h + 90) % 180 for h in hues]
            sh_min, sh_max = min(shifted), max(shifted)
            h_low = (sh_min - ADAPTIVE_HUE_MARGIN - self.hue_tolerance) % 180
            h_high = (sh_max + ADAPTIVE_HUE_MARGIN + self.hue_tolerance) % 180
            self.lower_green = np.array([h_low, s_low, v_low], dtype=np.uint8)
            self.upper_green = np.array([h_high, s_high, v_high], dtype=np.uint8)
        else:
            self._wraps_hue = False
            self.lower_green = np.array([h_low, s_low, v_low], dtype=np.uint8)
            self.upper_green = np.array([h_high, s_high, v_high], dtype=np.uint8)

    def _compute_lab_range(self) -> None:
        """Build LAB bounds from samples — green cloth has low a, moderate-high b."""
        if not self.lab_samples:
            return
        a_vals = [s[1] for s in self.lab_samples]
        b_vals = [s[2] for s in self.lab_samples]
        a_min, a_max = min(a_vals), max(a_vals)
        b_min, b_max = min(b_vals), max(b_vals)

        margin_a, margin_b = 15, 20
        self.lower_lab = np.array(
            [0, max(0, a_min - margin_a), max(0, b_min - margin_b)], dtype=np.uint8
        )
        self.upper_lab = np.array(
            [255, min(255, a_max + margin_a), min(255, b_max + margin_b)], dtype=np.uint8
        )

    def adjust_tolerance(self, delta: int) -> None:
        self.hue_tolerance = int(np.clip(self.hue_tolerance + delta, 2, 60))
        if self.samples:
            self._recompute_range()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _hsv_mask(self, hsv: np.ndarray) -> np.ndarray:
        if self._wraps_hue and self.lower_green[0] > self.upper_green[0]:
            mask1 = cv2.inRange(hsv, self.lower_green, np.array([179, 255, 255], dtype=np.uint8))
            mask2 = cv2.inRange(
                hsv,
                np.array([0, self.lower_green[1], self.lower_green[2]], dtype=np.uint8),
                self.upper_green,
            )
            return cv2.bitwise_or(mask1, mask2)
        return cv2.inRange(hsv, self.lower_green, self.upper_green)

    @staticmethod
    def _lab_mask(lab: np.ndarray) -> np.ndarray:
        """Green in LAB: 'a' channel below neutral (~128), 'b' above neutral."""
        a = lab[:, :, 1].astype(np.int16)
        b = lab[:, :, 2].astype(np.int16)
        # Green cloth cluster: a < 128 (more green than magenta), b can vary
        mask_a = (a < 128).astype(np.uint8) * 255
        mask_b = (b > 100).astype(np.uint8) * 255
        return cv2.bitwise_and(mask_a, mask_b)

    def _lab_range_mask(self, lab: np.ndarray) -> np.ndarray:
        if len(self.lab_samples) > 0:
            return cv2.inRange(lab, self.lower_lab, self.upper_lab)
        return self._lab_mask(lab)

    def create_mask(self, frame: np.ndarray, use_lab: bool = True) -> np.ndarray:
        """Return single-channel uint8 mask (255 = green cloth)."""
        filtered = self._prefilter(frame)
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        hsv_mask = self._hsv_mask(hsv)

        if not use_lab:
            return hsv_mask

        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        lab_mask = self._lab_range_mask(lab)

        # Intersection reduces false positives (e.g. green-ish walls, plants)
        combined = cv2.bitwise_and(hsv_mask, lab_mask)

        # If calibration samples exist, trust HSV more at very high saturation
        if self.samples:
            high_sat = (hsv[:, :, 1] > 80).astype(np.uint8) * 255
            strong_green = cv2.bitwise_and(hsv_mask, high_sat)
            combined = cv2.bitwise_or(combined, cv2.bitwise_and(strong_green, lab_mask))

        return combined

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_calibration(self, path=CALIBRATION_FILE) -> None:
        data = {
            "lower": self.lower_green.tolist(),
            "upper": self.upper_green.tolist(),
            "lower_lab": self.lower_lab.tolist(),
            "upper_lab": self.upper_lab.tolist(),
            "samples": self.samples,
            "lab_samples": self.lab_samples,
            "hue_tolerance": self.hue_tolerance,
            "adaptive_mode": self.adaptive_mode,
            "wraps_hue": self._wraps_hue,
        }
        path.write_text(json.dumps(data, indent=2))

    def load_calibration(self, path=CALIBRATION_FILE) -> bool:
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            self.lower_green = np.array(data["lower"], dtype=np.uint8)
            self.upper_green = np.array(data["upper"], dtype=np.uint8)
            self.lower_lab = np.array(
                data.get("lower_lab", [0, LAB_A_MIN, LAB_B_MIN]), dtype=np.uint8
            )
            self.upper_lab = np.array(
                data.get("upper_lab", [255, LAB_A_MAX, LAB_B_MAX]), dtype=np.uint8
            )
            self.samples = [tuple(s) for s in data.get("samples", [])]
            self.lab_samples = [tuple(s) for s in data.get("lab_samples", [])]
            self.hue_tolerance = data.get("hue_tolerance", HUE_TOLERANCE)
            self.adaptive_mode = data.get("adaptive_mode", ADAPTIVE_MODE)
            self._wraps_hue = data.get("wraps_hue", False)
            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def range_description(self) -> Tuple[str, str]:
        lo, hi = self.lower_green, self.upper_green
        range_str = f"H:{lo[0]}-{hi[0]} S:{lo[1]}-{hi[1]} V:{lo[2]}-{hi[2]}"
        if self.samples:
            last = self.samples[-1]
            sample_str = f"H:{last[0]} S:{last[1]} V:{last[2]} ({len(self.samples)} samples)"
        else:
            sample_str = "No samples yet"
        return sample_str, range_str
