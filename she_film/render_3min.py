"""Render the 7:25 song film in independently restartable 45s chunks."""
import os
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from she_film.clock import DURATION, FPS, HEIGHT, WIDTH, audit
from she_film.frame import render
from she_film.plates import load

ROOT = os.path.dirname(os.path.dirname(__file__))
AUDIO = os.path.join(ROOT, "She.wav")
OUTPUT = os.path.join(ROOT, "she_full_song_master.mp4")
STILLS = os.path.join(ROOT, "stills_full_song")

MARKS = [
    (25.0, "00a_early_growth_25s"),
    (18.0, "01_title_dawn"),
    (48.0, "02_spores"),
    (88.0, "03_mushroom_bloom"),
    (126.0, "04_first_twilight"),
    (174.0, "05_saucer_landing"),
    (194.0, "06_night_launch"),
    (230.0, "07_vegetation_to_forest"),
    (236.0, "07a_whorl_fir_growth_midpoint"),
    (260.0, "07b_whorl_fir_caravan"),
    (260.0, "07a_dinosaur_reflection_260s"),
    (270.0, "08_dinosaur_grazing_marine_life"),
    (307.0, "08a_crimson_transition_307s"),
    (310.0, "09_second_moonrise"),
    (390.0, "10a_late_caravan_390s"),
    (400.0, "10b_whorl_fir_decay"),
    (406.0, "10c_bare_rest"),
    (414.0, "10d_mushroom_return"),
    (380.0, "10_cosmic_demigod_vr_waveform"),
    (420.0, "11a_studio_screen_start_420s"),
    (426.0, "11_matrix_glitch_tearing"),
    (430.0, "11b_studio_screen_motion_430s"),
    (438.0, "12_retro_telemetry_studio_animated_screens"),
]


def save_stills(plates):
    os.makedirs(STILLS, exist_ok=True)
    images = []
    for t, name in MARKS:
        im = Image.fromarray(render(plates, t))
        path = os.path.join(STILLS, f"{name}.jpg")
        im.save(path, quality=90)
        images.append((name, im))
        print(name, flush=True)
    sheet = _sheet(images)
    sheet.save(os.path.join(STILLS, "00_contact_sheet.jpg"), quality=90)
    print(os.path.join(STILLS, "00_contact_sheet.jpg"), flush=True)


def _sheet(images):
    cols = 5
    tw, th, label, pad = 384, 216, 28, 10
    rows = int(np.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + label + pad) + pad), (16, 14, 12))
    draw = ImageDraw.Draw(sheet)
    font = None
    for p in [
        r"C:\Windows\Fonts\bahnschrift.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, 16)
                break
            except Exception:
                continue
    if font is None:
        try:
            font = ImageFont.load_default(size=16)
        except Exception:
            font = ImageFont.load_default()
    for i, (name, im) in enumerate(images):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (th + label + pad)
        sheet.paste(im.resize((tw, th), Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + th + 4), name.replace("_", " "), fill=(236, 226, 210), font=font)
    return sheet


def render_chunk(bounds):
    start, end, path = bounds
    expected_frames = int((end - start) * FPS)
    if os.path.exists(path) and os.path.getsize(path) > 500_000:
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if actual_frames >= expected_frames - 2:
                print(f"[CACHE HIT] Chunk already verified ({actual_frames}/{expected_frames} frames): {path}", flush=True)
                return
            else:
                print(f"[CACHE CORRUPT] Incomplete chunk {path} ({actual_frames}/{expected_frames} frames), re-rendering...", flush=True)
        except Exception:
            pass
    plates = load()
    n0, n1 = int(start * FPS), int(end * FPS)
    total_frames = n1 - n0
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "rgb24", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        path,
    ]
    log = open(path + ".log", "w")
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
    try:
        for idx, i in enumerate(range(n0, n1)):
            pipe.stdin.write(render(plates, i / FPS).tobytes())
            if (idx + 1) % 150 == 0 or idx == total_frames - 1:
                pct = ((idx + 1) / total_frames) * 100
                print(f"[{os.path.basename(path)}] {idx + 1}/{total_frames} frames ({pct:.1f}%)", flush=True)
    finally:
        pipe.stdin.close()
        pipe.wait()
        log.close()
    if pipe.returncode != 0:
        raise SystemExit(f"ffmpeg failed {pipe.returncode} {path}")
    print(f"[COMPLETE] {path} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)", flush=True)


def _has_nvenc():
    try:
        res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        return "h264_nvenc" in res.stdout
    except Exception:
        return False


def render_movie():
    cache = os.path.join(ROOT, "she_film", "cache")
    os.makedirs(cache, exist_ok=True)
    chunk_seconds = 45
    chunks = []
    for i, start in enumerate(range(0, math.ceil(DURATION / chunk_seconds) * chunk_seconds, chunk_seconds)):
        end = min(DURATION, start + chunk_seconds)
        path = os.path.join(cache, f"song_{i:02d}.mp4")
        chunks.append((start, end, path))

    if "--clean" in sys.argv or "--force" in sys.argv:
        print("[CLEAN] Wiping chunk cache for fresh multi-worker render...", flush=True)
        for _, _, p in chunks:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def launch(bounds):
        start, end, path = bounds
        return subprocess.run(
            [sys.executable, "-m", "she_film.render_3min", "--chunk", str(start), str(end), path],
            cwd=ROOT,
        ).returncode

    # Each worker owns its own cached plates and writes a separate file.
    # This bounds memory while keeping acts independently rerenderable.
    worker_count = max(1, int(os.environ.get("SHE_RENDER_WORKERS", str(min(os.cpu_count() or 2, 4)))))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(launch, bounds): bounds for bounds in chunks}
        for future in as_completed(futures):
            bounds = futures[future]
            code = future.result()
            if code:
                raise SystemExit(f"chunk failed ({bounds[0]}–{bounds[1]}s), exit {code}")
            print(f"completed {bounds[0]:.0f}–{bounds[1]:.0f}s", flush=True)
    stitch_movie(enable_4k=("--4k" in sys.argv))


def upscale_master(resolution):
    """Create an upscaled delivery from the 1080p master without rerendering scenes."""
    dimensions = {"4k": (3840, 2160), "8k": (7680, 4320)}
    if resolution not in dimensions:
        raise ValueError(f"Unsupported upscale target: {resolution}")
    width, height = dimensions[resolution]
    output = os.path.join(ROOT, f"she_full_song_master_{resolution}.mp4")
    print(f"Upscaling 1080p master to {resolution.upper()} ({width}x{height}) with Lanczos + restrained sharpening on CPU...", flush=True)
    cmd = [
        "ffmpeg", "-y", "-i", OUTPUT,
        "-vf", f"scale={width}:{height}:flags=lanczos,unsharp=5:5:0.25:5:5:0.0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", output,
    ]
    subprocess.check_call(cmd)
    print(f"[SUCCESS] {resolution.upper()} upscale: {output}  {os.path.getsize(output) / 1024 / 1024:.1f} MB", flush=True)
    return output


def stitch_movie(enable_4k=False):
    cache = os.path.join(ROOT, "she_film", "cache")
    chunk_seconds = 45
    chunks = []
    for i, start in enumerate(range(0, math.ceil(DURATION / chunk_seconds) * chunk_seconds, chunk_seconds)):
        end = min(DURATION, start + chunk_seconds)
        path = os.path.join(cache, f"song_{i:02d}.mp4")
        chunks.append((start, end, path))

    missing = [p for _, _, p in chunks if not os.path.exists(p) or os.path.getsize(p) < 100_000]
    if missing:
        print(f"[ERROR] Cannot stitch, {len(missing)} chunk(s) missing/incomplete:")
        for m in missing:
            print("  -", m)
        return False

    listing = os.path.join(cache, "acts.txt")
    with open(listing, "w", encoding="utf-8") as handle:
        for _, _, path in chunks:
            handle.write(f"file '{path.replace(chr(92), '/')}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listing,
        "-i", AUDIO, "-t", str(DURATION),
        # Refined blue-blocker finish: 12% blue suppression with restrained
        # red/green lift for amber warmth, without a contrast curve or vignette.
        "-vf", "colorchannelmixer=rr=1.018:gg=1.006:bb=0.88",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "320k",
        "-af", f"afade=t=in:ss=0:d=1.2,afade=t=out:st={DURATION-12:.3f}:d=12",
        OUTPUT,
    ]
    subprocess.check_call(cmd)
    print(f"[SUCCESS] 1080p Master: {OUTPUT}  {DURATION:.3f}s  {os.path.getsize(OUTPUT) / 1024 / 1024:.1f} MB", flush=True)

    if enable_4k or "--4k" in sys.argv:
        upscale_master("4k")
    if "--8k" in sys.argv:
        upscale_master("8k")
    return True


def main():
    if "--chunk" in sys.argv:
        index = sys.argv.index("--chunk")
        render_chunk((float(sys.argv[index + 1]), float(sys.argv[index + 2]), sys.argv[index + 3]))
        return
    if "--stitch" in sys.argv:
        stitch_movie(enable_4k=("--4k" in sys.argv))
        return
    if "--upscale-only" in sys.argv:
        targets = [target for target in ("4k", "8k") if f"--{target}" in sys.argv]
        if not targets:
            raise SystemExit("Choose --4k or --8k with --upscale-only")
        for target in targets:
            upscale_master(target)
        return
    audit()
    if "--stills" in sys.argv:
        save_stills(load())
        return
    render_movie()


if __name__ == "__main__":
    main()
