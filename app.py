"""
Invisible Cloak — Streamlit web app with real-time webcam processing.

Run: streamlit run app.py
"""

from __future__ import annotations

from typing import Optional

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from cloak_engine import CloakEngine, EngineSettings
from config import BACKGROUND_CAPTURE_FRAMES

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
if "sampled_this_session" not in st.session_state:
    st.session_state.sampled_this_session = False
if "cloth_color_ready" not in st.session_state:
    st.session_state.cloth_color_ready = False
if "effect_enabled" not in st.session_state:
    st.session_state.effect_enabled = False


def _get_settings() -> EngineSettings:
    return EngineSettings(
        effect_enabled=st.session_state.get("effect_enabled", False),
        mode=st.session_state.get("view_mode", "cloak"),
        feathering=st.session_state.get("feathering", True),
        background_gating=st.session_state.get("bg_gating", True),
        dual_color_space=False,
        spill_suppression=False,
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
# Setup controls
# ---------------------------------------------------------------------------
st.title("🧥 Invisible Cloak")
st.caption(
    "Capture the empty scene, sample any cloth color, then start the effect. "
    "Keep the camera **stationary** for the best result."
)

engine: CloakEngine = get_engine()


@st.fragment(run_every="1s")
def render_setup() -> None:
    with st.sidebar:
        st.header("Setup")
        st.caption("Start the webcam below before capturing.")

        st.subheader("1. Capture the empty background")
        st.markdown("Keep the cloth out of view and the camera still.")
        if st.button("📸 Capture Background", width="stretch"):
            if engine.last_frame is None:
                st.session_state.bg_msg = "Start the webcam before capturing."
            else:
                st.session_state.effect_enabled = False
                engine.start_background_capture(BACKGROUND_CAPTURE_FRAMES)
                st.session_state.bg_msg = "Collecting background frames..."

        if engine.is_capturing_background:
            st.warning(
                f"Capturing background: {len(engine.capture_buffer)}"
                f"/{engine.capture_target_frames} frames"
            )
        elif engine.background_ready:
            st.success("Background captured.")
        else:
            st.info("Background not captured yet.")
        if st.session_state.get("bg_msg"):
            st.caption(st.session_state.bg_msg)

        st.divider()
        st.subheader("2. Sample the cloth color")
        st.markdown(
            "Show the cloth to the camera, freeze a frame, and place the crosshair "
            "on the cloth. Any color is supported."
        )
        if st.button("🎯 Freeze current frame", width="stretch"):
            if engine.last_frame is None:
                st.warning("Start the webcam first.")
            else:
                st.session_state.calibration_frame = engine.last_frame.copy()
                height, width = engine.last_frame.shape[:2]
                st.session_state.sample_x = width // 2
                st.session_state.sample_y = height // 2

        cal_frame = st.session_state.calibration_frame
        if cal_frame is not None:
            height, width = cal_frame.shape[:2]
            sx = st.slider(
                "Sample X", 0, width - 1, key="sample_x"
            )
            sy = st.slider(
                "Sample Y", 0, height - 1, key="sample_y"
            )
            preview = cal_frame.copy()
            cv2.drawMarker(preview, (sx, sy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                caption="Position the crosshair on a clear patch of cloth.",
                width="stretch",
            )

            sample_col, reset_col = st.columns(2)
            with sample_col:
                if st.button("➕ Add cloth sample", width="stretch"):
                    if not st.session_state.sampled_this_session:
                        engine.detector.reset_samples()
                        st.session_state.sampled_this_session = True
                    sample = engine.sample_patch(cal_frame, sx, sy, radius=3)
                    if sample is None:
                        st.error("Could not sample that point. Choose another spot.")
                    else:
                        engine.detector.save_calibration()
                        st.session_state.cloth_color_ready = True
                        st.session_state.effect_enabled = False
            with reset_col:
                if st.button("🔄 Reset color", width="stretch"):
                    engine.detector.reset_samples()
                    engine.detector.save_calibration()
                    st.session_state.sampled_this_session = False
                    st.session_state.cloth_color_ready = False
                    st.session_state.effect_enabled = False

        if st.session_state.cloth_color_ready and engine.detector.last_sample_bgr:
            blue, green, red = engine.detector.last_sample_bgr
            st.success(
                f"Cloth color sampled: RGB({red}, {green}, {blue}) · "
                f"{len(engine.detector.samples)} sample(s)"
            )
        else:
            st.info("No cloth color sampled in this session.")

        st.divider()
        st.subheader("3. Start the invisibility effect")
        can_start = (
            engine.background_ready
            and st.session_state.cloth_color_ready
            and bool(engine.detector.samples)
        )
        if st.session_state.effect_enabled:
            if st.button("⏹ Stop effect", width="stretch"):
                st.session_state.effect_enabled = False
        elif st.button(
            "▶ Start effect",
            width="stretch",
            disabled=not can_start,
        ):
            st.session_state.effect_enabled = True

        if not can_start and not st.session_state.effect_enabled:
            st.caption("Capture the background and sample the cloth color to enable Start.")
        elif st.session_state.effect_enabled:
            st.success("Effect is running.")

        with st.expander("Fine-tune detection"):
            st.slider(
                "Color tolerance", 2, 60, engine.detector.hue_tolerance, key="hue_tol"
            )
            st.toggle(
                "Adapt to additional samples",
                value=engine.detector.adaptive_mode,
                key="adaptive",
            )
            st.toggle(
                "Ignore colors already in background",
                value=True,
                key="bg_gating",
            )
            st.slider(
                "Background difference threshold", 5, 80, 25, key="bg_diff"
            )
            st.slider(
                "Minimum detected area", 100, 3000, 400, step=50, key="min_area"
            )
            st.selectbox(
                "Mask smoothing", [3, 5, 7, 9], index=1, key="morph_k"
            )
            st.toggle(
                "Soften edges", value=True, key="feathering"
            )
            st.selectbox(
                "Edge softness", [7, 11, 15, 21], index=2, key="feather_b"
            )
            st.radio(
                "Preview",
                options=["cloak", "mask", "debug"],
                format_func=lambda x: {
                    "cloak": "Invisible",
                    "mask": "Mask",
                    "debug": "Debug",
                }[x],
                horizontal=True,
                key="view_mode",
            )

        engine.sync_settings(_get_settings())


render_setup()

# ---------------------------------------------------------------------------
# Main video stream
# ---------------------------------------------------------------------------
st.subheader("Live Feed")
st.caption(
    "Click **START** to enable the webcam. Use the setup panel to capture the "
    "background and sample your cloth color, then press **Start effect**."
)

webrtc_ctx = webrtc_streamer(
    key="invisible-cloak",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
    video_processor_factory=CloakVideoProcessor,
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.info("Webcam active.")
else:
    st.warning("Click **START** above to enable your webcam and begin setup.")

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
with st.expander("How to use"):
    st.markdown(
        """
        ### Quick start
        1. Click **START** on the webcam widget and keep the camera still.
        2. Remove the cloth, then click **Capture Background** in the sidebar.
        3. Show the cloth in the camera, click **Freeze current frame**, and position the crosshair.
        4. Click **Add cloth sample**. Add more samples for different shades of the same cloth.
        5. Click **Start effect** after both setup steps are complete.

        ### Accuracy tips
        - Sample **5–10 points** across the cloth (dark, light, shadow, edges).
        - Any cloth color can be sampled; samples from a previous session are replaced by your first new sample.
        - Enable **Ignore colors already in background** to reduce false positives.
        - Use **Debug** view to inspect original / mask / result side-by-side.
        - Increase **Color tolerance** if parts of the cloth are missed.
        - Decrease it if the background is incorrectly masked.

        ### Files
        - Background saved to `background.jpg`
        - Calibration saved to `calibration.json`
        """
    )
