"""
Invisible Cloak — real-time green-screen style invisibility using OpenCV.

Run: python main.py
"""

from __future__ import annotations

import sys
import time
from enum import Enum, auto
from typing import Tuple

import cv2
import numpy as np

from background import BackgroundManager
from camera import Camera
from color_detector import ColorDetector
from config import (
    AUTO_RECAPTURE_ENABLED,
    AUTO_RECAPTURE_INTERVAL_SEC,
    BACKGROUND_FILE,
    FONT,
    FONT_SCALE,
    FONT_THICKNESS,
    TOLERANCE_STEP,
    UI_COLOR,
    WINDOW_NAME,
    AppState,
)
from mask_processor import MaskProcessor


class Phase(Enum):
    INTRO = auto()
    BACKGROUND_WAIT = auto()
    BACKGROUND_CAPTURE = auto()
    POST_BACKGROUND = auto()
    CALIBRATION = auto()
    ACTIVE = auto()


class InvisibleCloakApp:
    """Main application orchestrating camera, detection, and UI."""

    def __init__(self) -> None:
        self.camera = Camera()
        self.background_mgr = BackgroundManager()
        self.detector = ColorDetector()
        self.mask_processor = MaskProcessor()
        self.state = AppState(
            adaptive_mode=self.detector.adaptive_mode,
            hue_tolerance=self.detector.hue_tolerance,
        )
        self.phase = Phase.INTRO
        self._mouse_x = 0
        self._mouse_y = 0
        self._last_recapture = time.time()
        self._status_message = ""
        self._status_until = 0.0
        self._calibration_clicks = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> int:
        try:
            self.camera.open()
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        try:
            while True:
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    print("Failed to read from webcam.", file=sys.stderr)
                    break

                display = self._process_frame(frame)
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break

                if (
                    AUTO_RECAPTURE_ENABLED
                    and self.phase == Phase.ACTIVE
                    and time.time() - self._last_recapture > AUTO_RECAPTURE_INTERVAL_SEC
                ):
                    self._start_background_recapture()
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

        return 0

    # ------------------------------------------------------------------
    # Frame pipeline
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.phase == Phase.INTRO:
            return self._draw_intro(frame)

        if self.phase == Phase.BACKGROUND_WAIT:
            return self._draw_background_wait(frame)

        if self.phase == Phase.BACKGROUND_CAPTURE:
            return frame  # countdown drawn inside capture_with_countdown

        if self.phase == Phase.POST_BACKGROUND:
            return self._draw_post_background(frame)

        if self.phase == Phase.CALIBRATION:
            return self._draw_calibration(frame)

        # ACTIVE
        return self._draw_active(frame)

    def _draw_active(self, frame: np.ndarray) -> np.ndarray:
        if not self.background_mgr.is_ready:
            return self._overlay_text(frame, ["Background not ready. Press B to capture."])

        background = self.background_mgr.resize_to_match(frame)
        raw_mask = self.detector.create_mask(frame)
        mask = self.mask_processor.process(raw_mask, frame, background)
        result = self.mask_processor.apply_invisibility(frame, background, mask)

        if self.state.debug_view:
            mask_bgr = self.mask_processor.mask_to_bgr(mask)
            panel_h = frame.shape[0]
            panel_w = frame.shape[1]
            combined = np.zeros((panel_h, panel_w * 3, 3), dtype=np.uint8)
            combined[:, 0:panel_w] = frame
            combined[:, panel_w : panel_w * 2] = mask_bgr
            combined[:, panel_w * 2 :] = result
            display = self._add_panel_labels(combined, ["Original", "Mask", "Result"])
        elif self.state.mask_preview:
            display = self.mask_processor.mask_to_bgr(mask)
        else:
            display = result

        return self._draw_hud(display, mask)

    def _draw_calibration(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        lines = [
            "CALIBRATION MODE",
            "Click on GREEN CLOTH (multiple clicks OK)",
            "Dark / light / shadow areas — sample them all",
            "Press ENTER when done, ESC to cancel last session",
        ]
        if self.detector.samples:
            sample_str, range_str = self.detector.range_description()
            lines.append(f"Samples: {len(self.detector.samples)}")
            lines.append(f"Last HSV — {sample_str}")
            lines.append(f"Range — {range_str}")
            lines.append(f"Hue tolerance: {self.detector.hue_tolerance} (+/- to adjust)")
        display = self._overlay_text(display, lines, origin_y=30)
        cv2.circle(display, (self._mouse_x, self._mouse_y), 6, (0, 0, 255), 2)
        return display

    def _draw_intro(self, frame: np.ndarray) -> np.ndarray:
        lines = [
            "INVISIBLE CLOAK",
            "",
            "1. Place camera in a fixed position.",
            "2. Make sure the room is visible.",
            "3. Remove the green cloth/person from the scene.",
            "4. Press SPACE to capture background.",
            "",
            "Q = Quit",
        ]
        return self._overlay_text(frame, lines)

    def _draw_background_wait(self, frame: np.ndarray) -> np.ndarray:
        return self._overlay_text(
            frame,
            ["Background captured successfully!", "Put on the green cloak.", "Press C to calibrate green color."],
        )

    def _draw_post_background(self, frame: np.ndarray) -> np.ndarray:
        # Brief transition screen; auto-advance to calibration prompt
        return self._draw_background_wait(frame)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        green_pixels = int(np.count_nonzero(mask))
        sample_str, range_str = self.detector.range_description()

        lines = [
            "---------------------------------",
            " INVISIBLE CLOAK",
            "---------------------------------",
            "",
            "Status:",
            "BACKGROUND CAPTURED" if self.background_mgr.is_ready else "NO BACKGROUND",
            "GREEN DETECTION: ACTIVE" if self.state.invisible_active else "STANDBY",
            "",
            f"FPS: {self.camera.fps:.1f}",
            f"Green pixels: {green_pixels}",
            f"Tolerance: {self.detector.hue_tolerance}",
            f"Adaptive: {'ON' if self.detector.adaptive_mode else 'OFF'}",
            "",
            f"Range: {range_str}",
        ]

        if self._status_message and time.time() < self._status_until:
            lines.insert(4, self._status_message)

        help_lines = [
            "B=Recapture  C=Calibrate  R=Reset  M=Mask  D=Debug",
            "+/-=Tolerance  L=Load BG  S=Screenshot  F=Fullscreen  Q=Quit",
        ]

        out = self._overlay_text(frame, lines, origin_y=20, line_height=22)
        out = self._overlay_text(
            out,
            help_lines,
            origin_y=out.shape[0] - 50,
            line_height=20,
            color=(200, 200, 200),
        )
        return out

    def _overlay_text(
        self,
        frame: np.ndarray,
        lines: list[str],
        origin_y: int = 80,
        line_height: int = 28,
        color: Tuple[int, int, int] = UI_COLOR,
    ) -> np.ndarray:
        overlay = frame.copy()
        y = origin_y
        for line in lines:
            if line.startswith("---"):
                cv2.line(overlay, (15, y - 10), (400, y - 10), color, 1)
                y += line_height // 2
                continue
            cv2.putText(
                overlay,
                line,
                (15, y),
                FONT,
                FONT_SCALE if not line.isupper() or len(line) < 20 else FONT_SCALE + 0.1,
                color,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )
            y += line_height
        return overlay

    @staticmethod
    def _add_panel_labels(combined: np.ndarray, labels: list[str]) -> np.ndarray:
        panel_w = combined.shape[1] // 3
        for i, label in enumerate(labels):
            cv2.putText(
                combined,
                label,
                (i * panel_w + 10, 25),
                FONT,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return combined

    def _set_status(self, message: str, duration: float = 3.0) -> None:
        self._status_message = message
        self._status_until = time.time() + duration

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        self._mouse_x, self._mouse_y = x, y
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.phase != Phase.CALIBRATION:
            return

        ok, frame = self.camera.read()
        if not ok or frame is None:
            return

        # Map click coords when debug view scales display width
        if self.state.debug_view:
            panel_w = frame.shape[1]
            x = x % panel_w

        sample = self.detector.sample_at(frame, x, y)
        if sample is not None:
            self._calibration_clicks += 1
            self.detector.save_calibration()
            self._set_status(f"Sampled HSV {sample}", 2.0)

    def _handle_key(self, key: int) -> bool:
        if key == ord("q"):
            return False

        if key == ord(" "):
            if self.phase in (Phase.INTRO, Phase.POST_BACKGROUND):
                self._start_background_capture()
            return True

        if key == ord("b"):
            if self.phase == Phase.ACTIVE:
                self._start_background_recapture()
            return True

        if key == ord("c"):
            self.phase = Phase.CALIBRATION
            self._calibration_clicks = 0
            self._set_status("Calibration: click green cloth", 4.0)
            return True

        if key == 13 and self.phase == Phase.CALIBRATION:  # ENTER
            if self.detector.samples:
                self.phase = Phase.ACTIVE
                self.state.invisible_active = True
                self.detector.save_calibration()
                self._set_status("INVISIBLE CLOAK ACTIVE", 3.0)
            else:
                self._set_status("Click the cloth first!", 2.0)
            return True

        if key == ord("r"):
            self.detector.reset_samples()
            self._set_status("Green range reset to defaults", 2.0)
            return True

        if key in (ord("+"), ord("=")):
            self.detector.adjust_tolerance(TOLERANCE_STEP)
            self.state.hue_tolerance = self.detector.hue_tolerance
            self._set_status(f"Hue tolerance: {self.detector.hue_tolerance}", 1.5)
            return True

        if key == ord("-"):
            self.detector.adjust_tolerance(-TOLERANCE_STEP)
            self.state.hue_tolerance = self.detector.hue_tolerance
            self._set_status(f"Hue tolerance: {self.detector.hue_tolerance}", 1.5)
            return True

        if key == ord("m"):
            self.state.mask_preview = not self.state.mask_preview
            return True

        if key == ord("d"):
            self.state.debug_view = not self.state.debug_view
            self.state.mask_preview = False
            return True

        if key == ord("a"):
            self.detector.adaptive_mode = not self.detector.adaptive_mode
            self.state.adaptive_mode = self.detector.adaptive_mode
            if self.detector.samples:
                self.detector._recompute_range()
            self._set_status(
                f"Adaptive mode: {'ON' if self.detector.adaptive_mode else 'OFF'}",
                2.0,
            )
            return True

        if key == ord("l"):
            if self.background_mgr.load():
                self.state.background_ready = True
                self.phase = Phase.ACTIVE
                self.state.invisible_active = True
                self._set_status(f"Loaded {BACKGROUND_FILE.name}", 2.0)
            else:
                self._set_status("No saved background found", 2.0)
            return True

        if key == ord("s"):
            self._save_screenshot()
            return True

        if key == ord("f"):
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
            return True

        return True

    # ------------------------------------------------------------------
    # Background flow
    # ------------------------------------------------------------------

    def _start_background_capture(self) -> None:
        self.phase = Phase.BACKGROUND_CAPTURE
        ok = self.background_mgr.capture_with_countdown(self.camera, WINDOW_NAME)
        if ok:
            self.state.background_ready = True
            self.phase = Phase.POST_BACKGROUND
            self._set_status("Background captured!", 4.0)
        else:
            self.phase = Phase.INTRO
            self._set_status("Background capture failed", 3.0)

    def _start_background_recapture(self) -> None:
        self._set_status("Recapturing background...", 2.0)
        ok = self.background_mgr.recapture(self.camera, WINDOW_NAME)
        self._last_recapture = time.time()
        if ok:
            self._set_status("Background recaptured!", 2.0)
        else:
            self._set_status("Recapture failed", 2.0)

    def _save_screenshot(self) -> None:
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        if self.background_mgr.is_ready:
            bg = self.background_mgr.resize_to_match(frame)
            raw = self.detector.create_mask(frame)
            mask = self.mask_processor.process(raw, frame, bg)
            result = self.mask_processor.apply_invisibility(frame, bg, mask)
            cv2.imwrite(f"screenshot_{ts}.jpg", result)
            cv2.imwrite(f"mask_{ts}.jpg", mask)
            self._set_status(f"Saved screenshot_{ts}.jpg", 2.0)


def main() -> int:
    app = InvisibleCloakApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
