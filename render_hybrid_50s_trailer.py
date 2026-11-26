"""Modular 50-second 9-Beat Hybrid Cinematic Trailer for SHE
(~60% original she_film + ~40% reference-driven Google Flow aerials).

Supports per-beat textless caching in `output_teaser/beats/` so any future
text, font, crossfade, or audio edit runs in ~20s with `--stitch-only`."""
import argparse
import os
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from she_film import frame, grade, plates
from she_film.clock import FPS, HEIGHT, WIDTH, smooth
from she_film.outro import _font

_WIN_FLOW_DIR = r"C:\Users\sayan\flow-mcp-out\SHE_Hybrid_Trailer_Aerials"
FLOW_DIR = _WIN_FLOW_DIR if os.path.isdir(_WIN_FLOW_DIR) else os.path.join(os.path.dirname(__file__), "assets", "flow_aerials")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output_teaser")
BEATS_DIR = os.path.join(OUT_DIR, "beats")
os.makedirs(BEATS_DIR, exist_ok=True)
TEMP_MP4 = os.path.join(OUT_DIR, "hybrid_50s_video_only.mp4")
FINAL_MP4 = os.path.join(OUT_DIR, "SHE_Teaser_50s_Hybrid_1080p.mp4")
AUDIO_WAV = os.path.join(os.path.dirname(__file__), "She.wav")


_BG96_PATH = os.path.join(os.path.dirname(__file__), "vendor", "gemini-watermark-remover", "assets", "bg_96.png")


def _load_flow_clip(filename, target_frames, grade_t=30.0, skip_start=0):
    path = os.path.join(FLOW_DIR, filename)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open Flow clip: {path}")

    alpha_48 = None
    wm_pos = None
    if os.path.exists(_BG96_PATH):
        bg = cv2.imread(_BG96_PATH)
        if bg is not None:
            alpha_48 = cv2.resize(np.max(bg, axis=2).astype(np.float32) / 255.0, (48, 48), interpolation=cv2.INTER_AREA)

    raw = []
    idx_f = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx_f < skip_start:
            idx_f += 1
            continue
        idx_f += 1
        h, w = bgr.shape[:2]
        # Reverse alpha-blending (ishara-madu/gemini-watermark-remover)
        if alpha_48 is not None:
            if wm_pos is None:
                g0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                rx0, ry0 = int(w * 0.78), int(h * 0.72)
                res = cv2.matchTemplate(g0[ry0:h, rx0:w], (alpha_48 * 255.0).astype(np.float32), cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(res)
                wm_pos = (rx0 + ml[0], ry0 + ml[1]) if mv > 0.6 else (1136, 576)
            wx, wy = wm_pos
            if wx + 48 <= w and wy + 48 <= h:
                patch = bgr[wy : wy + 48, wx : wx + 48].astype(np.float32)
                a = np.clip(alpha_48 * 0.52, 0.0, 0.99)[..., None]
                rec = np.where(a >= 0.002, (patch - a * 255.0) / np.maximum(1e-4, 1.0 - a), patch)
                bgr[wy : wy + 48, wx : wx + 48] = np.clip(np.round(rec), 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)
        graded = grade.crystal_nocturne(rgb.astype(np.float32), grade_t, intensity_override=0.35)
        raw.append(np.clip(graded, 0, 255).astype(np.uint8))
    cap.release()
    idxs = np.linspace(0, len(raw) - 1, target_frames)
    out = []
    for idx in idxs:
        i0 = int(np.floor(idx))
        i1 = min(len(raw) - 1, i0 + 1)
        frac = idx - i0
        if frac < 1e-3 or i0 == i1:
            out.append(raw[i0])
        else:
            blended = raw[i0].astype(np.float32) * (1.0 - frac) + raw[i1].astype(np.float32) * frac
            out.append(np.clip(blended, 0, 255).astype(np.uint8))
    return out


def _make_text_overlay(text, y_frac, style="caps"):
    """Unified warm champagne-gold typography matching the 'WORLDS RISE' motif."""
    plate = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    gold = (244, 206, 130, 255)
    shadow = (6, 8, 12, 230)
    cy = int(HEIGHT * y_frac)

    if style == "dramatic":
        font, fill, tracking = _font(44, bold=True), gold, 10.0
    else:
        font, fill, tracking = _font(38, bold=True), gold, 8.5

    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    scrim = np.exp(-(((xx - WIDTH * 0.5) / 520.0) ** 2 + ((yy - cy) / 92.0) ** 2))
    scrim_arr = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    scrim_arr[..., :3] = (8, 10, 14)
    scrim_arr[..., 3] = np.clip(scrim * (155.0 if style == "dramatic" else 120.0), 0, 255).astype(np.uint8)
    plate = Image.alpha_composite(plate, Image.fromarray(scrim_arr))
    draw = ImageDraw.Draw(plate, "RGBA")

    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + tracking * max(0, len(text) - 1)
    x = (WIDTH - total) / 2.0
    y = cy - 22
    for ch, w in zip(text, widths):
        draw.text((round(x) + 3, y + 3), ch, font=font, fill=shadow)
        draw.text((round(x), y), ch, font=font, fill=fill)
        x += w + tracking

    if style == "dramatic":
        ry = cy + 44
        draw.line((WIDTH // 2 - 230, ry, WIDTH // 2 + 230, ry), fill=(224, 190, 127, 220), width=2)
        draw.polygon([(WIDTH // 2, ry - 6), (WIDTH // 2 + 6, ry), (WIDTH // 2, ry + 6), (WIDTH // 2 - 6, ry)],
                     fill=(245, 222, 175, 255))

    arr = np.asarray(plate, dtype=np.float32)
    arr[..., 3] /= 255.0
    return arr


def _digital_camera_move(rgb, progress, zoom_end=1.0, tilt_y=0.0):
    if abs(zoom_end - 1.0) < 1e-4 and abs(tilt_y) < 0.5:
        return rgb
    s = 1.0 + (zoom_end - 1.0) * smooth(progress)
    dy = tilt_y * smooth(progress)
    cx, cy = WIDTH * 0.5, HEIGHT * 0.5 + dy
    nw, nh = WIDTH / s, HEIGHT / s
    x0 = np.clip(cx - nw * 0.5, 0, WIDTH - nw)
    y0 = np.clip(cy - nh * 0.5, 0, HEIGHT - nh)
    crop = rgb[int(y0) : int(y0 + nh), int(x0) : int(x0 + nw)]
    return cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)


def _apply_timed_text(rgb_f, overlay, local_t, t_in0, t_in1, t_out0, t_out1):
    if overlay is None or local_t < t_in0 or local_t > t_out1:
        return rgb_f
    if local_t < t_in1:
        a = smooth((local_t - t_in0) / max(0.05, t_in1 - t_in0))
    elif local_t <= t_out0:
        a = 1.0
    else:
        a = 1.0 - smooth((local_t - t_out0) / max(0.05, t_out1 - t_out0))
    if a <= 0.01:
        return rgb_f
    alpha = overlay[..., 3:4] * a
    return rgb_f * (1.0 - alpha) + overlay[..., :3] * alpha


def _save_raw_beat(beat_path, frames_rgb):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(beat_path, fourcc, FPS, (WIDTH, HEIGHT))
    for rgb in frames_rgb:
        vw.write(cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()


def _load_raw_beat(beat_path):
    cap = cv2.VideoCapture(beat_path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32))
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch-only", action="store_true", help="Reuse cached raw beat MP4s and only apply text/crossfades/audio.")
    ap.add_argument("--beats", type=str, default="", help="Comma-separated 1-based beat indices to force re-render (e.g. 3,5).")
    args = ap.parse_args()

    force_beats = {int(x.strip()) for x in args.beats.split(",") if x.strip()}

    beat_specs = [
        ("01_flow_dawn_approach", 7 * FPS, ("EVERY DAWN REMEMBERS", 0.20, "caps", 2.5, 4.8, 5.8, 7.0)),
        ("02_flow_mushroom_reveal", 5 * FPS, None),
        ("03_og_night_descent", 6 * FPS, ("WHAT THE TIDES FORGOT", 0.20, "caps", 2.0, 3.2, 4.4, 5.4)),
        ("04_flow_aurora_flyover", 5 * FPS, None),
        ("05_og_worlds_rise", 6 * FPS, ("WORLDS RISE", 0.22, "caps", 0.0, 1.2, 3.4, 4.6)),
        ("06_og_impermanence", 5 * FPS, ("AND RETURN TO ASH", 0.22, "caps", 2.0, 3.0, 4.0, 5.0)),
        ("07_og_buddha_stillness", 6 * FPS, ("ONLY TO BEGIN AGAIN", 0.18, "caps", 1.0, 2.2, 4.0, 5.2)),
        ("08_og_cyclical_return", 4 * FPS, ("TIME IS A CIRCLE", 0.46, "dramatic", 1.0, 2.0, 3.0, 4.0)),
        ("09_og_outro_she", 6 * FPS, None),
    ]

    p = None
    def get_plates():
        nonlocal p
        if p is None:
            p = plates.load()
        return p

    raw_generators = {
        1: lambda n: _load_flow_clip("scene-01.mp4", n, grade_t=34.0, skip_start=18),
        2: lambda n: _load_flow_clip("scene-02.mp4", n, grade_t=34.0),
        3: lambda n: [_digital_camera_move(frame.render(get_plates(), 167.0 + k / FPS), k / (n - 1), zoom_end=1.06) for k in range(n)],
        4: lambda n: _load_flow_clip("scene-03.mp4", n, grade_t=170.0),
        5: lambda n: [_digital_camera_move(frame.render(get_plates(), 280.0 + k / FPS), k / (n - 1), zoom_end=1.04) for k in range(n)],
        6: lambda n: [_digital_camera_move(frame.render(get_plates(), 348.0 + k / FPS), k / (n - 1), zoom_end=1.04, tilt_y=-18.0) for k in range(n)],
        7: lambda n: [frame.render(get_plates(), 409.0 + k / FPS) for k in range(n)],
        8: lambda n: [frame.render(get_plates(), 33.0 + k / FPS) for k in range(n)],
        9: lambda n: [frame.render(get_plates(), 438.5 + k / FPS) for k in range(n)],
    }

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(TEMP_MP4, fourcc, FPS, (WIDTH, HEIGHT))
    prev_tail = None
    total_written = 0

    for idx, (b_name, n_frames, txt_spec) in enumerate(beat_specs, 1):
        beat_mp4 = os.path.join(BEATS_DIR, f"raw_beat_{idx:02d}_{b_name}.mp4")
        if os.path.exists(beat_mp4) and idx not in force_beats and (args.stitch_only or True):
            raw_frames = _load_raw_beat(beat_mp4)
            print(f"Loaded cached {b_name} ({len(raw_frames)} frames)", flush=True)
        else:
            raw_frames = [np.asarray(f, dtype=np.float32) for f in raw_generators[idx](n_frames)]
            _save_raw_beat(beat_mp4, raw_frames)
            print(f"Rendered & cached {b_name} ({len(raw_frames)} frames)", flush=True)

        ov = _make_text_overlay(txt_spec[0], txt_spec[1], txt_spec[2]) if txt_spec else None
        beat_frames = []
        for k, cur in enumerate(raw_frames):
            if txt_spec:
                cur = _apply_timed_text(cur, ov, k / FPS, txt_spec[3], txt_spec[4], txt_spec[5], txt_spec[6])
            beat_frames.append(cur)

        xfade = 48 if idx == 3 else (12 if idx > 1 else 0)
        for k, cur in enumerate(beat_frames):
            if idx == 1 and k < 12:
                cur = cur * ((k + 1) / 12.0)
            if xfade > 0 and prev_tail is not None and k < xfade:
                a = smooth((k + 1) / xfade)
                cur = prev_tail * (1.0 - a) + cur * a
            vw.write(cv2.cvtColor(np.clip(cur, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            total_written += 1
        prev_tail = beat_frames[-1]

    vw.release()
    dur = total_written / FPS
    cmd = [
        "ffmpeg", "-y",
        "-i", TEMP_MP4,
        "-ss", "168", "-t", f"{dur:.2f}", "-i", AUDIO_WAV,
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
        "-af", f"afade=t=in:st=0:d=1.5,afade=t=out:st={dur-2.5:.2f}:d=2.5",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", FINAL_MP4,
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(TEMP_MP4):
        os.remove(TEMP_MP4)
    print(f"Saved modular 50s Hybrid Trailer: {FINAL_MP4}", flush=True)


if __name__ == "__main__":
    main()
