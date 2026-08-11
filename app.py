"""
Invisible Cloak — Streamlit web app with real-time webcam processing.

Run: streamlit run app.py
"""

from __future__ import annotations

import time
from typing import Optional

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from cloak_engine import CloakEngine, EngineSettings
from config import BACKGROUND_CAPTURE_FRAMES, BACKGROUND_FILE, CALIBRATION_FILE

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Invisible Cloak",
    page_icon="🧥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Module-level engine shared with the WebRTC worker thread.
# Use a fallback that also works when the WebRTC worker thread does not have
# the same Streamlit session-state context as the main script.
_ENGINE: Optional[CloakEngine] = None


def get_engine() -> CloakEngine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    try:
        engine = st.session_state.get("engine")
    except Exception:
        engine = None

    if engine is None:
        engine = CloakEngine()
        try:
            st.session_state.engine = engine
        except Exception:
            pass

    _ENGINE = engine
    return _ENGINE


if "calibration_frame" not in st.session_state:
    st.session_state.calibration_frame: Optional[np.ndarray] = None
if "sample_x" not in st.session_state:
    st.session_state.sample_x = 320
if "sample_y" not in st.session_state:
    st.session_state.sample_y = 240


def _get_settings() -> EngineSettings:
    return EngineSettings(
        mode=st.session_state.get("view_mode", "cloak"),
        feathering=st.session_state.get("feathering", True),
        background_gating=st.session_state.get("bg_gating", True),
        dual_color_space=st.session_state.get("dual_cs", True),
        spill_suppression=st.session_state.get("spill", True),
        bg_diff_threshold=st.session_state.get("bg_diff", 25),
        hue_tolerance=st.session_state.get("hue_tol", 15),
        adaptive_mode=st.session_state.get("adaptive", True),
        min_contour_area=st.session_state.get("min_area", 400),
        morph_kernel=st.session_state.get("morph_k", 5),
        feather_blur=st.session_state.get("feather_b", 15),
    )


class CloakVideoProcessor(VideoProcessorBase):
    """Video callback for streamlit-webrtc."""

    def __init__(self) -> None:
        self.engine: CloakEngine = get_engine()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        out = self.engine.process(img)
        return av.VideoFrame.from_ndarray(out, format="bgr24")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.title("🧥 Invisible Cloak")
st.caption(
    "Camera-based visual effect — not physical invisibility. "
    "Keep the camera **stationary** and avoid changing the room after background capture."
)

engine: CloakEngine = get_engine()

with st.sidebar:
    st.header("Setup")

    # --- Step 1: Background ---
    st.subheader("1. Background")
    st.markdown("Remove the green cloth from view, then capture the empty room.")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("📸 Capture Background", use_container_width=True):
            engine.start_background_capture(BACKGROUND_CAPTURE_FRAMES)
            st.session_state.bg_msg = f"Collecting {BACKGROUND_CAPTURE_FRAMES} frames..."
    with col_b2:
        if st.button("📂 Load Saved", use_container_width=True):
            if engine.background_mgr.load():
                st.session_state.bg_msg = "Background loaded from file."
            else:
                st.session_state.bg_msg = "No background.jpg found."

    if engine.background_ready:
        st.success("✅ Background ready")
    elif engine.is_capturing_background:
        st.warning(f"⏳ Capturing... ({len(engine.capture_buffer)}/{engine.capture_target_frames})")
    else:
        st.info("⬜ Background not captured yet")

    if "bg_msg" in st.session_state:
        st.caption(st.session_state.bg_msg)

    st.divider()

    # --- Step 2: Calibration ---
    st.subheader("2. Calibrate Green")
    st.markdown("Freeze a frame, pick a point on the cloth, add samples.")

    if st.button("🎯 Freeze Frame for Calibration", use_container_width=True):
        if engine.last_frame is not None:
            st.session_state.calibration_frame = engine.last_frame.copy()
            h, w = engine.last_frame.shape[:2]
            st.session_state.sample_x = w // 2
            st.session_state.sample_y = h // 2
        else:
            st.warning("Start the webcam first.")

    cal_frame = st.session_state.calibration_frame
    if cal_frame is not None:
        h, w = cal_frame.shape[:2]
        sx = st.slider("Sample X", 0, w - 1, st.session_state.sample_x, key="sx")
        sy = st.slider("Sample Y", 0, h - 1, st.session_state.sample_y, key="sy")
        st.session_state.sample_x = sx
        st.session_state.sample_y = sy

        preview = cal_frame.copy()
        cv2.drawMarker(preview, (sx, sy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Calibration frame", use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Add Sample", use_container_width=True):
                sample = engine.sample_patch(cal_frame, sx, sy, radius=3)
                if sample:
                    engine.detector.save_calibration()
                    st.success(f"Sampled HSV {sample}")
        with c2:
            if st.button("🔄 Reset Samples", use_container_width=True):
                engine.detector.reset_samples()
                st.info("Samples cleared.")

    sample_str, range_str = engine.detector.range_description()
    st.text(f"Last: {sample_str}")
    st.text(f"Range: {range_str}")

    st.divider()

    # --- Detection tuning ---
    st.subheader("3. Detection Tuning")
    st.session_state.hue_tol = st.slider("Hue Tolerance", 2, 60, engine.detector.hue_tolerance)
    st.session_state.adaptive = st.toggle("Adaptive Range", value=engine.detector.adaptive_mode)
    st.session_state.dual_cs = st.toggle("Dual HSV + LAB", value=True)
    st.session_state.bg_gating = st.toggle("Background Gating", value=True)
    st.session_state.bg_diff = st.slider("BG Diff Threshold", 5, 80, 25)
    st.session_state.spill = st.toggle("Spill Suppression", value=True)
    st.session_state.min_area = st.slider("Min Contour Area", 100, 3000, 400, step=50)
    st.session_state.morph_k = st.selectbox("Morph Kernel", [3, 5, 7, 9], index=1)
    st.session_state.feathering = st.toggle("Edge Feathering", value=True)
    st.session_state.feather_b = st.selectbox("Feather Blur", [7, 11, 15, 21], index=2)

    st.divider()

    # --- View mode ---
    st.subheader("4. View Mode")
    st.session_state.view_mode = st.radio(
        "Display",
        options=["cloak", "mask", "debug"],
        format_func=lambda x: {"cloak": "🧥 Invisible", "mask": "⬛ Mask", "debug": "🔍 Debug (3-panel)"}[x],
        horizontal=True,
    )

# ---------------------------------------------------------------------------
# Main video stream
# ---------------------------------------------------------------------------
st.subheader("Live Feed")

webrtc_ctx = webrtc_streamer(
    key="invisible-cloak",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
    video_processor_factory=CloakVideoProcessor,
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.info("Webcam active — use the sidebar to capture background and calibrate.")
else:
    st.warning("Click **START** above to enable your webcam.")

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
with st.expander("How to use"):
    st.markdown(
        """
        ### Quick start
        1. Click **START** on the webcam widget.
        2. Fix your camera — do not move it later.
        3. Remove the green cloth → **Capture Background** in the sidebar.
        4. Put on the green cloak.
        5. **Freeze Frame** → move crosshair to cloth → **Add Sample** (repeat for dark/light/shadow areas).
        6. Watch the invisible effect in the live feed.

        ### Accuracy tips
        - Sample **5–10 points** across the cloth (dark, light, shadow, edges).
        - Enable **Dual HSV + LAB** — reduces false positives from green objects in the room.
        - Enable **Background Gating** — ignores green pixels that match the stored background.
        - Use **Debug** view to inspect original / mask / result side-by-side.
        - Increase **Hue Tolerance** if parts of the cloth are missed.
        - Decrease it if the background is incorrectly masked.

        ### Files
        - Background saved to `background.jpg`
        - Calibration saved to `calibration.json`
        """
    )

# Sync settings on every rerun
engine.sync_settings(_get_settings())
