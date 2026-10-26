# Invisible Cloak

A real-time **camera-based visual effect** that makes a person or object covered in green cloth appear invisible on screen. This is **not** true physical invisibility — it works by replacing detected green pixels with a previously captured background image.

> **Important:** Keep the camera **stationary** and avoid changing the room after background capture.

## Web App (Recommended)

The Streamlit web app provides an easy UI with live tuning controls.

### Installation

```bash
cd invisible_cloak
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at **http://localhost:8501**

### Streamlit Community Cloud webcam connections

The app uses Google's public STUN server by default. Some networks block direct
WebRTC connections and require a TURN relay. If the webcam stays on
“Connection is taking longer than expected,” obtain a TURN URL and credentials
from a TURN provider, then add them in the app's **Streamlit Community Cloud →
Settings → Secrets**:

```toml
[webrtc]
turn_server = "turn:your-turn-server.example:3478"
turn_username = "your-turn-username"
turn_credential = "your-turn-credential"
```

Keep the TURN credentials in Streamlit secrets; do not commit them to GitHub.
The app adds this relay alongside STUN automatically. If the connection still
fails, check that the TURN service is active and permits the relay transports
required by your network.

### Web app workflow

1. Click **START** on the webcam widget (allow camera permission).
2. Fix the camera — do not move it later.
3. Remove the cloth → sidebar **Capture Background**.
4. Show the cloth in the camera → **Freeze current frame**.
5. Position the crosshair on the cloth and click **Add cloth sample**. Any cloth color is supported.
6. Click **Start effect**. Add more samples for different shades of the same cloth if needed.

### Sidebar controls

| Control | Purpose |
|---------|---------|
| Capture Background | Median-stack 30 frames into reference |
| Load Saved | Reload `background.jpg` |
| Freeze Frame | Snapshot for calibration |
| Add cloth sample | Sample a 7×7 patch of any cloth color at the crosshair |
| Color tolerance | Widen/narrow the sampled cloth-color range |
| Background gating | Ignore matching-color pixels that are already in the stored background |
| View Mode | Invisible / Mask / Debug (3-panel) |

## Desktop App (Legacy)

```bash
python main.py
```

Keyboard controls: SPACE, C, ENTER, B, M, D, +/-, Q — see original section below.

## How It Works

1. Capture a clean background (room without the green cloth).
2. Calibrate the cloth color by sampling patches on your cloth.
3. Detect matching pixels using **HSV** color ranges.
4. Gate mask with **background difference** to avoid false positives.
5. Replace masked pixels with the stored background.
6. Feather edges for smooth blending.

## Accuracy Improvements

| Technique | Benefit |
|-----------|---------|
| **Patch sampling (5×5 median)** | Stable calibration, ignores sensor noise |
| **Dual HSV + LAB** | Fewer false positives from walls/plants |
| **Background gating** | Won't mask static green objects in the BG |
| **Bilateral pre-filter** | Cleaner color conversion, keeps edges |
| **Adaptive multi-sample range** | Covers dark/light/shadow cloth areas |
| **Spill suppression** | Removes green fringing on boundaries |
| **Larger morph close kernel** | Fills holes inside the cloak silhouette |

## HSV vs RGB

| RGB | HSV + LAB |
|-----|-----------|
| Exact `(0,255,0)` fails on shadows | Hue isolates color from brightness |
| Lighting shifts all channels | LAB `a`/`b` channels separate green from neutral tones |
| One point in color space | **Range** + multi-sample covers variation |

OpenCV HSV: **H 0–179**, **S 0–255**, **V 0–255**

## Project Structure

```
invisible_cloak/
├── app.py              # Streamlit web app (recommended)
├── cloak_engine.py     # Unified processing engine
├── main.py             # Legacy OpenCV desktop app
├── config.py           # All parameters
├── camera.py           # Desktop webcam helper
├── background.py       # Median background capture
├── color_detector.py   # HSV + LAB detection
├── mask_processor.py   # Morphology, gating, feathering
├── requirements.txt
└── README.md
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Webcam won't start | Allow browser camera permission; use Chrome/Edge |
| Green edges visible | Enable spill suppression + feathering |
| Floor/objects turn invisible | Tighten hue tolerance; enable background gating |
| Background mismatch | Recapture background; keep camera still |
| Cloth not detected | Add more samples; increase hue tolerance |
| Dark green missed | Sample shadow areas; lower Value Min in config |

## Lighting Tips

- Even lighting on the green cloth helps.
- Avoid green objects in the background.
- Matte fabric works better than shiny material.

## Disclaimer

This creates a **video effect** similar to green-screen compositing. The camera must stay fixed and the background must not change significantly.

## License

MIT
