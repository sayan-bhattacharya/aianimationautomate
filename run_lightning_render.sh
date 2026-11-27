#!/usr/bin/env bash
set -euo pipefail

BASE="/teamspace/studios/this_studio"
[ -d "$BASE" ] || BASE="$PWD"
cd "$BASE"

echo "=== 1. Checking GPU & System Tools ==="
nvidia-smi
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg xz-utils libgl1 libxi6 libxrender1 libxkbcommon0 libsm6 wget
pip install -q opencv-python pillow numpy

echo "=== 2. Installing Blender 4.3.2 ==="
BLENDER_DIR="$BASE/blender-4.3.2-linux-x64"
BLENDER_BIN="$BLENDER_DIR/blender"
if [ ! -x "$BLENDER_BIN" ]; then
  rm -f "$BASE/blender.tar.xz"
  echo "Downloading Blender 4.3.2 (~352 MB)..."
  curl -fL --progress-bar "https://ftp.nluug.nl/pub/graphics/blender/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz" -o "$BASE/blender.tar.xz" \
    || curl -fL -A "Mozilla/5.0" --progress-bar "https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz" -o "$BASE/blender.tar.xz"
  ls -lh "$BASE/blender.tar.xz"
  echo "Extracting Blender 4.3.2..."
  tar -xf "$BASE/blender.tar.xz" -C "$BASE"
  rm -f "$BASE/blender.tar.xz"
fi
"$BLENDER_BIN" --version

echo "=== 3. Starting Self-Healing Blender 3D + DoF Render ==="
python3 - <<'PY'
import glob, json, os, shutil, subprocess

BASE = "/teamspace/studios/this_studio"
if not os.path.exists(BASE):
    BASE = os.getcwd()
REPO = os.path.join(BASE, "aianimationautomate")
BLENDER_BIN = os.path.join(BASE, "blender-4.3.2-linux-x64", "blender")
CLEAN_DIR = os.path.join(REPO, "assets", "blender_clean")
WORK_DIR = os.path.join(BASE, "SHE_Film_4K_Blender")
CHUNKS_DIR = os.path.join(WORK_DIR, "chunks")
os.makedirs(CHUNKS_DIR, exist_ok=True)

nb = json.load(open(os.path.join(REPO, "SHE_4K_Blender_DoF_LightningAI.ipynb")))
exec("".join(nb["cells"][4]["source"]), {"WORK_DIR": WORK_DIR, "os": os})
BLENDER_SCRIPT = os.path.join(WORK_DIR, "render_she_blender.py")

MODE = os.environ.get("SHE_MODE", "720p_preview")
RES_W, RES_H, SAMPLES = (1280, 720, 8) if MODE == "720p_preview" else (1920, 1080, 12)
TOTAL_FRAMES, CHUNK_SIZE = 10680, 1068
n_chunks = (TOTAL_FRAMES + CHUNK_SIZE - 1) // CHUNK_SIZE
chunk_mp4s = []

for ci in range(n_chunks):
    f0, f1 = ci * CHUNK_SIZE, min(TOTAL_FRAMES, (ci + 1) * CHUNK_SIZE)
    c_mp4 = os.path.join(CHUNKS_DIR, f"blender_chunk_{ci+1:02d}_{MODE}.mp4")
    chunk_mp4s.append(c_mp4)
    if os.path.exists(c_mp4) and os.path.getsize(c_mp4) > 100_000:
        print(f"Skipping completed chunk {ci+1:02d}", flush=True)
        continue
    tmp_frames = os.path.join(WORK_DIR, f"frames_{MODE}", f"chunk_{ci+1:02d}")
    subprocess.run([BLENDER_BIN, "-b", "--python", BLENDER_SCRIPT, "--", tmp_frames, str(f0), str(f1), str(RES_W), str(RES_H), str(SAMPLES), CLEAN_DIR], check=True)
    pngs = sorted(glob.glob(f"{tmp_frames}/frame_*.png"))
    print(f"Found {len(pngs)}/{f1-f0} PNGs in {tmp_frames}", flush=True)
    if not pngs:
        raise RuntimeError(f"No PNGs rendered in {tmp_frames}")
    subprocess.run(["ffmpeg", "-y", "-framerate", "24", "-pattern_type", "glob", "-i", f"{tmp_frames}/frame_*.png", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", c_mp4], check=True)
    shutil.rmtree(tmp_frames, ignore_errors=True)
    print(f"✓ Saved chunk {ci+1:02d}/{n_chunks}: {c_mp4}", flush=True)

concat_txt = os.path.join(WORK_DIR, "blender_concat.txt")
with open(concat_txt, "w") as f:
    for cp in chunk_mp4s:
        f.write(f"file '{cp}'\n")

SAFE_MASTER = os.path.join(WORK_DIR, f"SHE_Blender_DoF_{MODE}_master.mp4")
if not (os.path.exists(SAFE_MASTER) and os.path.getsize(SAFE_MASTER) > 1_000_000):
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-i", os.path.join(REPO, "She.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", SAFE_MASTER], check=True)

MASTER_4K = os.path.join(WORK_DIR, f"SHE_Blender_DoF_4K_{MODE}.mp4")
subprocess.run(["ffmpeg", "-y", "-i", SAFE_MASTER, "-vf", "scale=3840:2160:flags=lanczos,unsharp=5:5:0.45:5:5:0.0", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy", MASTER_4K], check=True)
print("✓ Final 4K Blender Master ready:", MASTER_4K, flush=True)
PY
