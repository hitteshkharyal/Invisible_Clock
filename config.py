"""
Configuration for the Invisible Cloak application.

All tunable parameters live here. Adjust these values to match your
webcam, lighting, and green cloth without editing the core logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
BACKGROUND_FILE = PROJECT_ROOT / "background.jpg"
CALIBRATION_FILE = PROJECT_ROOT / "calibration.json"

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
TARGET_FPS: int = 30

# ---------------------------------------------------------------------------
# Background capture
# ---------------------------------------------------------------------------
BACKGROUND_COUNTDOWN_SECONDS: int = 5
BACKGROUND_CAPTURE_FRAMES: int = 30
BACKGROUND_CAPTURE_DELAY_MS: float = 50  # pause between sample frames

# ---------------------------------------------------------------------------
# Default HSV green range (OpenCV: H 0-179, S 0-255, V 0-255)
# Used until the user calibrates or loads saved values.
# ---------------------------------------------------------------------------
DEFAULT_LOWER_GREEN: Tuple[int, int, int] = (35, 80, 70)
DEFAULT_UPPER_GREEN: Tuple[int, int, int] = (85, 255, 255)

# ---------------------------------------------------------------------------
# Color calibration tolerances
# ---------------------------------------------------------------------------
HUE_TOLERANCE: int = 15
SATURATION_MARGIN: int = 90
VALUE_MARGIN: int = 100
SATURATION_MIN: int = 40
VALUE_MIN: int = 30

# Step size when pressing +/- to adjust hue tolerance
TOLERANCE_STEP: int = 2

# ---------------------------------------------------------------------------
# Adaptive detection (range from min/max of all samples + margins)
# ---------------------------------------------------------------------------
ADAPTIVE_MODE: bool = True
ADAPTIVE_HUE_MARGIN: int = 8
ADAPTIVE_SAT_MARGIN: int = 30
ADAPTIVE_VAL_MARGIN: int = 40

# ---------------------------------------------------------------------------
# LAB color space defaults (OpenCV LAB: L 0-255, a 0-255, b 0-255)
# Green cloth: a channel below ~128, b above ~100
# ---------------------------------------------------------------------------
LAB_A_MIN: int = 0
LAB_A_MAX: int = 120
LAB_B_MIN: int = 90
LAB_B_MAX: int = 255

# ---------------------------------------------------------------------------
# Mask processing
# ---------------------------------------------------------------------------
BLUR_SIZE: int = 5          # must be odd; applied to raw mask
MORPH_KERNEL_SIZE: int = 5  # must be odd
MIN_CONTOUR_AREA: int = 400
BG_DIFF_THRESHOLD: int = 25

# Optional soft edge blending around the cloak boundary
MASK_FEATHERING: bool = True
FEATHER_BLUR_SIZE: int = 15  # must be odd

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
WINDOW_NAME: str = "Invisible Cloak"
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: float = 0.55
FONT_THICKNESS: int = 1
UI_COLOR: Tuple[int, int, int] = (0, 255, 0)
UI_BG_ALPHA: float = 0.6

# ---------------------------------------------------------------------------
# Auto background refresh (off by default)
# ---------------------------------------------------------------------------
AUTO_RECAPTURE_ENABLED: bool = False
AUTO_RECAPTURE_INTERVAL_SEC: int = 60


@dataclass
class AppState:
    """Runtime flags shared across modules."""

    background_ready: bool = False
    calibration_mode: bool = False
    mask_preview: bool = False
    debug_view: bool = False
    invisible_active: bool = False
    hue_tolerance: int = HUE_TOLERANCE
    adaptive_mode: bool = ADAPTIVE_MODE
    calibration_samples: list = field(default_factory=list)
