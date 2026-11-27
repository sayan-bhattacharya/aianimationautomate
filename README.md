# 🎬 SHE: 4K Visual Music Film & Blender DoF Pipeline

[![Open Blender 4K DoF in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sayan-bhattacharya/aianimationautomate/blob/main/SHE_4K_Blender_DoF_Colab.ipynb)
[![Open 4K Master & Teaser in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sayan-bhattacharya/aianimationautomate/blob/main/SHE_4K_Master_And_Teaser_Colab.ipynb)

Autonomous computer-vision, Blender 3D, and procedural VFX pipeline for the full-length ambient visual music film **"SHE"** (7:25 | 10,680 frames @ 24fps | 4K `3840×2160`) by **Sayan Bhattacharya** and **Atandra Chakraborty (Between The Notes)**.

---

## ▶️ Watch the Official 4K Teaser First

> **Click the preview below to watch *SHE | Official Teaser (4K)* on YouTube:**

[![Watch SHE Official Teaser 4K](assets/teaser_stills/SHE_Teaser_Thumb_V2_Pagoda_Sun_LowAngle.jpg)](https://www.youtube.com/watch?v=AZGOTzeI6-g)

<p align="center">
  <img src="assets/teaser_preview.gif" alt="SHE 4K Teaser Animated Preview" width="640" />
</p>

### 🎞️ Teaser Stills & Hybrid Aerial Video Clips

| Dawn Coast Aerial | Pagoda Sun (Low Angle) | Aurora Night Drone | Crimson Moon Mirror |
| :---: | :---: | :---: | :---: |
| ![Dawn Coast](assets/teaser_stills/SHE_Teaser_Thumb_V1_Dawn_Coast_Aerial.jpg) | ![Pagoda Sun](assets/teaser_stills/SHE_Teaser_Thumb_V2_Pagoda_Sun_LowAngle.jpg) | ![Aurora Night](assets/teaser_stills/SHE_Teaser_Thumb_V3_Aurora_Night_Drone.jpg) | ![Crimson Moon](assets/teaser_stills/SHE_Teaser_Thumb_V4_Crimson_Moon_Mirror.jpg) |

* **Downloadable Aerial Video Plates:**
  * [`assets/flow_aerials/scene-01.mp4`](assets/flow_aerials/scene-01.mp4)
  * [`assets/flow_aerials/scene-02.mp4`](assets/flow_aerials/scene-02.mp4)
  * [`assets/flow_aerials/scene-03.mp4`](assets/flow_aerials/scene-03.mp4)

---

## 🧩 Cleaned 4K Image Assets (`assets/blender_clean/`)

All dinosaur sprites are exterior border flood-filled (zero interior `alpha=0` holes), and all botanical plates have dark studio floor pads stripped and feathered for seamless 3D dune grounding.

### Botanicals
| Pagoda Mushroom | Whorl Fir (Warm) | Whorl Fir (Deep) | Whorl Fir (Cool) |
| :---: | :---: | :---: | :---: |
| ![Pagoda Mushroom](assets/blender_clean/hero_pagoda_mushroom_master.png) | ![Pine Warm](assets/blender_clean/pine_fuller_warm.png) | ![Pine Deep](assets/blender_clean/pine_fuller_deep.png) | ![Pine Cool](assets/blender_clean/pine_fuller_cool.png) |

### Dinosaurs
| Sauropod | Hadrosaur | Ankylosaur | Raptor | Pterodactyl |
| :---: | :---: | :---: | :---: | :---: |
| ![Sauropod](assets/blender_clean/pokemon_sauropod.png) | ![Hadrosaur](assets/blender_clean/pokemon_hadrosaur.png) | ![Ankylosaur](assets/blender_clean/pokemon_ankylosaur.png) | ![Raptor](assets/blender_clean/pokemon_raptor.png) | ![Pterodactyl](assets/blender_clean/pokemon_pterodactyl.png) |

---

## 🎥 Blender 3D + Optical Depth-of-Field Previews

| Mushroom Scene (`1:01` — Sun Center) | Forest Scene (`3:21` — Sunrise Left) | Dinosaur Caravan (`4:22` — Sun High Right) |
| :---: | :---: | :---: |
| ![Mushrooms DoF](blender_previews/01_mushrooms_1m01_sun_center.png) | ![Forest DoF](blender_previews/02_forest_3m21_sun_low_left.png) | ![Dinos DoF](blender_previews/03_dinos_4m22_sun_high_right.png) |

---

## 🚀 One-Click Google Colab Render (Self-Healing)

* **Blender 3D + Optical DoF 4K Pipeline (Recommended):**
  **[Open `SHE_4K_Blender_DoF_Colab.ipynb` in Colab](https://colab.research.google.com/github/sayan-bhattacharya/aianimationautomate/blob/main/SHE_4K_Blender_DoF_Colab.ipynb)**
  * Per-frame & per-chunk Google Drive checkpointing (resumes at the exact frame if Colab disconnects)
  * Built-in 5-frame T4 GPU benchmark (`720p` & `1080p`)
  * 3D undulating sand dunes, Cycles GPU + OpenImageDenoise, optical Depth of Field (`f/2.0`), and 4K Lanczos upscale with `She.wav`
* **OpenCV Hybrid 4K Master & Teaser Pipeline:**
  **[Open `SHE_4K_Master_And_Teaser_Colab.ipynb` in Colab](https://colab.research.google.com/github/sayan-bhattacharya/aianimationautomate/blob/main/SHE_4K_Master_And_Teaser_Colab.ipynb)**
