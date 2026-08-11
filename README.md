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

### Web app workflow

1. Click **START** on the webcam widget (allow camera permission).
2. Fix the camera — do not move it later.
3. Remove green cloth → sidebar **Capture Background**.
4. Put on the green cloak.
5. **Freeze Frame** → position crosshair on cloth → **Add Sample** (repeat for dark/light/shadow areas).
6. Watch the live invisible effect.

### Sidebar controls

| Control | Purpose |
|---------|---------|
| Capture Background | Median-stack 30 frames into reference |
| Load Saved | Reload `background.jpg` |
| Freeze Frame | Snapshot for calibration |
| Add Sample | 5×5 patch HSV+LAB sample at crosshair |
| Hue Tolerance | Widen/narrow green hue range |
| Dual HSV + LAB | Intersect color spaces for accuracy |
| Background Gating | Ignore green pixels matching stored BG |
| Spill Suppression | Remove green fringing on edges |
| View Mode | Invisible / Mask / Debug (3-panel) |

## Desktop App (Legacy)

```bash
python main.py
```

Keyboard controls: SPACE, C, ENTER, B, M, D, +/-, Q — see original section below.

## How It Works

1. Capture a clean background (room without the green cloth).
2. Calibrate the green color by sampling patches on your cloth.
3. Detect green pixels using **HSV + LAB** color ranges.
4. Gate mask with **background difference** to avoid false positives.
5. Replace masked pixels with the stored background.
6. Feather edges and suppress green spill for smooth blending.

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
