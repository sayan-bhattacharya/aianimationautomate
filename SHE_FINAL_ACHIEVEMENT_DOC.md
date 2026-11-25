# "SHE" — Complete Film & Engineering Achievement Document

**Runtime:** `445.0s` (`7:25`) | **Frame Rate:** `24 fps` (`10,680` frames) | **Master Resolution:** `1920 × 1080`  
**Soundtrack:** `She.wav` (`audio_features_24fps.npy`, 10-band 24fps audio matrix)  
**Repositories:**
- Canonical SHE repository and Colab source: `sayan-bhattacharya/aianimationautomate` (`main`)
- Project boundary: SHE is independent of the separate NOIR film repository; no NOIR code, assets, or render steps are part of this pipeline.

---

## 1. Narrative & Musical Arc (5-Act Timeline)

Every visual transition in `she_film/clock.py` is calibrated against the musical phrasing of `She.wav`:

| Time | Musical Event (`audio_features_24fps.npy`) | Visual Narrative & Scene State |
| :--- | :--- | :--- |
| **`0:00 – 0:08`** (`0s–8s`) | Ambient dawn intro (`rms: 0.005`) | Branded intro logo bleach reveal crossfading into the coastal horizon at `4.6s–8.0s`. |
| **`0:08 – 0:27`** (`8s–27s`) | Gentle pad swell (`rms: 0.024 → 0.071`) | Editorial title sequence (`BETWEEN THE NOTES PRESENTS` / `S H E` / credits), clearing cleanly before sunrise bloom. |
| **`0:14 – 1:56`** (`14s–116s`) | First plucks (`79.875s–91.79s`) & bass build | **Act 1 Daylight**: First sun rises (`14s–122s`). Five slender pagoda mushroom stands grow (`20s–54s`), sway in a left-to-right passing wave, and modulate height with every drone swell (`60s`, `84s`). Acoustic ocean ripples react to musical onsets. |
| **`1:53 – 2:05`** (`113s–125s`) | Deep drone swell (`rms: 0.310 → 0.692`, drastic drop `122s`) | **Parasympathetic Exhale**: Headspace-style smooth height reduction (`1.00x → 0.68x` over `113.0s–121.5s`), releasing smoothly back to `1.00x` (`121.5s–125.0s`) as the swell drops. |
| **`1:56 – 2:10`** (`116s–130s`) | Warm dusk transition | **Act 1 Sunset**: Golden-hour glow transitions into deep night; stars emerge *after* night sky settles (`126s–136s`). |
| **`2:10 – 3:10`** (`130s–190s`) | Night drone swell (`148s`, `rms: 0.533`) | **Act 2 Night & Saucer Visit**: Exaggerated star-sheet time-lapse rotation + Northern Lights aurora ribbons (`_aurora_timelapse`). Mushrooms yield (`166s–176s`), horizon mirage refracts (`166s–190s`), five whorl-fir trees emerge rear-to-front (`176s–184s`), and the Vimana saucer descends and lands (`182s–185s`). |
| **`3:10 – 4:54`** (`190s–294s`) | Song Climax (`200s–225s`, `rms: 0.795`) & Tambura entrance (`259.5s`) | **Act 3 Sunrise 2 & Caravan**: Saucer ascends along `-Z` into the `200s–212s` musical climax. Four-species dinosaur caravan (`Hero Sauropod`, `Hadrosaur`, `Ankylosaur`, `Feathered Raptor`) marches East-to-West (`232s–405s`) with pterodactyls overhead (`242s–352s`). |
| **`4:54 – 7:08`** (`294s–428s`) | Meditative Tanpura & descending chimes (`rms: 0.337 → 0.035`) | **Act 4–5 Sunset 2, Stillness & Return**: Crimson moon rises (`306s+`). `water_stillness` calms ocean ripples to a crystal mirror (`318s–348s`). Gradual wind-chill breeze starts around `5m52s` (`352s`), and exiting dinosaurs brush past the leftmost fir (`x=340`). Forest decays (`390s–404s`), bioluminescent pagoda mushrooms return (`404s–428s`, clean caps with no glitter dots), and the Cosmic Buddha + audio-reactive heart waveform (`410s–428s`) emerge. |
| **`7:00 – 7:25`** (`420s–445s`) | Final crystalline fade (`rms: 0.035 → 0.001`) | **Telemetry Studio & Finale**: Cybernetic reality glitch (`420s–436s`) dissolves into the 1970s walnut telemetry studio with 4 moving CRT feeds (`scr_left`, `scr_mid_left`, `scr_mid_right`, `scr_right`), followed by the spotlight `"She"` outro card (`435s–445s`) and final 2-second fade to black (`443s–445s`). |

---

## 2. Key Technical & Artistic Achievements

### A. Grounded Slender Pagoda Mushrooms (`actors.py`, `video_sprites`)
- **Proportions & Sprite Curation**: Extracted from the Google Flow reference into a single contiguous master silhouette (`hero_pagoda_mushroom_master.png`), preserving the full three-point botanical foundation (**left root claw**, **central taproots**, and **restored right buttress root leg**) bridged by an authentic earth mound with zero transparent holes.
- **Calibrated Height**: Slender upright aspect (`340px × 292px` across `anim_mushroom_growth.npz` [192 frames] and `anim_mushroom_idle.npz` [48 frames]).
- **Seamless Dune Integration**: Dynamic diurnal sand encasing (`dune_sand_color` from `sky_weights(t)`), unclipped elliptical ambient occlusion constrained above `HORIZON`, and soft sand-drift lip cradles.

### B. Tight Audio-Visual Coupling (`actors.py`, `water.py`, `mirage.py`, `frame.py`)
- **Full-Track Drone Swell Coupling**: `_drone_swell_strength(audio_feats, t)` computes a jitter-free 3-second (`±1.5s`) envelope of `0.55 * rms + 0.45 * bass`. `_parasympathetic_swell_mod(t, audio_feats)` couples mushroom height across all appearances (`20s–176s` and `404s–428s`), including the deep `0.68x` Headspace exhale at `1m53s–2m01s` (`113.0s–121.5s`).
- **Bioluminescence, Stars & Saucer Reflections**: Returning mushroom cap radiance (`t >= 400s`), `shimmering_stars` halos, saucer ocean glitter, `mirage.soften_background`, and concentric `water.py` ripples all modulate with `audio_feats`.
- **Buddha Heart Oscilloscope**: `_render_buddha_waveform` drives dual glowing harmonic traces from `sub_bass` and `onset`, paired with a 43 BPM expanding pure-white heart aura (`410s–428s`).

### C. Reverse-Engineered Optical Depth & Sun Corona (`actors.py`, `frame.py`)
- **Depth-of-Field & Aerial Perspective**:
  - **Stand 3** (Hero Focal Plane): `σ = 0.0`, `haze = 0%` (pin-sharp micro-contrast).
  - **Stand 0** (Near Left Foreground): `σ = 1.25px` portrait softness, `haze = 4%`.
  - **Stand 1 & 2** (Mid-Ground / Mid-Distance): `σ = 1.6–1.8px`, `haze = 12%–16%`.
  - **Stand 4** (Far Background): `σ = 2.8px`, `haze = 26%` sky veil.
- **Sun Corona Bloom**: `_scaled_sun` applies `σ = 5.5px` edge feathering + `σ = 12.0px` radial corona glow.

### D. Botanical Wind & Dinosaur-Plant Contact Physics (`actors.py`)
- **`_tree_wind_and_contact`**: Starting gradually around `5m52s` (`344s → 362s`), upper whorl-fir foliage sways with travelling gusts and row-varying leaf flutter while bottom root rows remain pinned to soil contact.
- **Caravan Contact Interaction**: As exiting dinosaurs pass within `115px` of foreground trees (notably `FOREST_ANCHORS[0]` at `x = 340` around `356.9s` and `369.2s`), mid/lower foliage deflects leftward with an elastic bounce. Applied symmetrically to `forest_reflect`.

### E. Aurora Star-Sheet Time-Lapse & Clean Night Return (`actors.py`)
- **Removed `6m57s` Glitter**: Eliminated the ascending `cv2.circle` glitter dots around returning mushrooms (`t >= 400s`).
- **Exaggerated Time-Lapse Heavens**: `stars` and `shimmering_stars` sweep/rotate the entire star sheet across the night sky while `_aurora_timelapse` renders flowing emerald, cyan, and violet northern-lights curtains.

### F. Editorial Outro End Card & Colab Pipeline (`outro.py`, `generate_screens_cache.py`)
- **Fixed `7:22` Outro Visibility**: Resolved missing Linux/Colab font paths (`LiberationSerif`, `DejaVuSerif`, `DejaVuSans`, plus scalable `ImageFont.load_default(size=size)`), converted `outro_card` from RGB to **BGR** to restore champagne-gold `"She"` (`280pt`/`200pt`) and spotlight colors, and enlarged all credit lines (`26pt`–`46pt`) with drop-shadows.
- **Headless OpenCV Compatibility**: Fixed `CV_8U` depth casting in `generate_screens_cache.py` for Colab CRT loop generation.
- **Quality Gatekeeper (`pre_render_code_review.py`)**: All **34 automated checks passing** (`0` failures, `0` warnings).

---

## 3. Core Module Map (`drone_animation_project/she_film/`)

| Module | Responsibility |
| :--- | :--- |
| `clock.py` | Master `445.0s` / `10,680`-frame timeline, celestial arcs, sky weights, saucer path, dinosaur caravan, Vedantic `water_stillness`, and Buddha/outro fades. |
| `plates.py` | Pre-rendered sky/water plates, 4-layer sinusoidal dune ridges, titles, sprites, `audio_features_24fps.npy`, and `outro_card`. |
| `actors.py` | Pagoda mushrooms + drone swell modulation, depth-of-field/haze, aurora/star-sheet time-lapse, whorl-firs + wind/dino physics, dinosaurs, pterodactyls, and saucer. |
| `frame.py` | Master `render(plates, t)` compositor, sun corona, Buddha waveform, 4-screen perspective CRT studio, glitch transition, vignette, and outro fade. |
| `water.py` | Subpixel-antialiased wave crests, concentric audio-reactive ripples, lamp reflections, and Tanpura stillness mirror. |
| `mirage.py` | Horizon mirage refraction and music-breathing background softening during transitions. |
| `outro.py` | Volumetric conical spotlight end card with `"She"` serif emblem and editorial credits. |
| `generate_screens_cache.py` | Builds and validates the 4 moving 528-frame CRT feeds (`screens_90s.npz`). |
| `pre_render_code_review.py` | 34-point automated pre-render gatekeeper audit. |
