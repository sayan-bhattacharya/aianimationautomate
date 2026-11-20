# 🎬 SHE: Full-Song 1080p Film Master Render Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sayan-bhattacharya/aianimationautomate/blob/main/SHE_Full_Song_Master_Render_Colab.ipynb)

Autonomous computer-vision & procedural VFX pipeline for the full-length music film **"SHE"** (7:29 | 10,777 frames @ 24fps | 1080p).

---

## 🚀 One-Click Google Colab Render

Click the badge above or navigate to:
**[Open in Google Colab](https://colab.research.google.com/github/sayan-bhattacharya/aianimationautomate/blob/main/SHE_Full_Song_Master_Render_Colab.ipynb)**

Works on any Google account (including free, Pro, GPU, or TPU runtimes) with zero authentication required.

### Pipeline Highlights
1. **Zero-Defect Visual Gatekeeper**: Automated pre-render audit verifying all 7 film segments, reflections, and camera transitions before rendering begins.
2. **True Perspective Telemetry Suite (2-Screen Buddha Redesign)**:
   - **Screen A (Left Wall)**: *Between The Notes* Logo (3s) → Dinosaur Caravan March (19s)
   - **Screen B (Right Wall)**: Cheetah Sprint (5s) → Romcom in Rain (5s) → 90s Commercial (6s) → Analog White Noise with CRT sync bar (6s)
   - Calibrated planar homography (`cv2.warpPerspective`) with chromatic aberration, CRT vignette, and phosphor bloom.
   - 100% clean, verified mushroom botanical canopy with all blue spore artifacts removed.
3. **Multi-Process Accelerated Chunking**: Splits the 10,777 frames across CPU/GPU workers with idempotent caching and automatic ffmpeg loss-free audio muxing.
