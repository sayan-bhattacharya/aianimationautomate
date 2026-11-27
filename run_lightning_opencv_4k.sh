#!/usr/bin/env bash
set -euo pipefail

BASE="/teamspace/studios/this_studio"
[ -d "$BASE" ] || BASE="$PWD"
REPO="$BASE/aianimationautomate"
OUT_DIR="$BASE/SHE_Film_4K_OpenCV"
CHUNKS_DIR="$OUT_DIR/chunks_1080p"
mkdir -p "$CHUNKS_DIR"

echo "=== 1. Installing OpenCV / Librosa / FFmpeg ==="
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg >/dev/null
pip install -q opencv-python pillow numpy librosa

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "=== 2. Generating Screens Cache ==="
python3 -m she_film.generate_screens_cache

echo "=== 3. Rendering 10 Chunks (10,680 frames) with Self-Healing Resume ==="
python3 - <<'PY'
import os, shutil, subprocess
import cv2, numpy as np
from she_film import frame, plates
from she_film.clock import FPS, HEIGHT, TOTAL_FRAMES, WIDTH

BASE = "/teamspace/studios/this_studio"
if not os.path.exists(BASE):
    BASE = os.getcwd()
REPO = os.path.join(BASE, "aianimationautomate")
OUT_DIR = os.path.join(BASE, "SHE_Film_4K_OpenCV")
CHUNKS_DIR = os.path.join(OUT_DIR, "chunks_1080p")

p = plates.load()
CHUNK_SIZE = 1068
n_chunks = (TOTAL_FRAMES + CHUNK_SIZE - 1) // CHUNK_SIZE
chunk_paths = []

for ci in range(n_chunks):
    f0 = ci * CHUNK_SIZE
    f1 = min(TOTAL_FRAMES, f0 + CHUNK_SIZE)
    c_path = os.path.join(CHUNKS_DIR, f"chunk_{ci+1:02d}.mp4")
    chunk_paths.append(c_path)
    if os.path.exists(c_path) and os.path.getsize(c_path) > 100_000:
        print(f"Skipping completed chunk {ci+1:02d}", flush=True)
        continue
    tmp_c = os.path.join(OUT_DIR, f"tmp_chunk_{ci+1:02d}.mp4")
    vw = cv2.VideoWriter(tmp_c, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for fi in range(f0, f1):
        rgb = frame.render(p, fi / FPS)
        vw.write(cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()
    shutil.move(tmp_c, c_path)
    print(f"✓ Saved chunk {ci+1:02d}/{n_chunks}: {c_path}", flush=True)

concat_txt = os.path.join(OUT_DIR, "concat_4k.txt")
with open(concat_txt, "w") as f:
    for cp in chunk_paths:
        f.write(f"file '{cp}'\n")

MASTER_1080 = os.path.join(OUT_DIR, "she_full_song_1080p_master.mp4")
if not (os.path.exists(MASTER_1080) and os.path.getsize(MASTER_1080) > 1_000_000):
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt,
        "-i", os.path.join(REPO, "She.wav"),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", MASTER_1080
    ], check=True)
    print("✓ Safe-point 1080p master saved:", MASTER_1080, flush=True)

MASTER_4K = os.path.join(OUT_DIR, "she_full_song_4K_crystal_nocturne_master.mp4")
print("Upscaling 1080p -> 4K (3840x2160)...", flush=True)
subprocess.run([
    "ffmpeg", "-y", "-i", MASTER_1080,
    "-vf", "scale=3840:2160:flags=lanczos,unsharp=5:5:0.45:5:5:0.0",
    "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
    "-c:a", "copy", MASTER_4K
], check=True)
print("✓ Final 4K OpenCV Master ready:", MASTER_4K, flush=True)
PY
