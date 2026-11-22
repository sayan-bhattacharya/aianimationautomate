"""Render a short, soundtrack-matched, blue-blocker review clip.

These are motion studies, not master-render substitutes. Run from the project
root with: python -m she_film.render_transition_previews
"""
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from she_film.clock import HEIGHT, WIDTH
from she_film.frame import render
from she_film.plates import load


ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, "preview_720p", "transition_clips_mirage")
SOURCE_AUDIO = os.path.join(ROOT, "she_full_song_master_blueblocker.mp4")
PREVIEWS = (
    # Focused code-rendered sample of the transition-specific mirage pass.
    (166.0, 188.0, "mirage_mushroom_to_whorl_fir_sample"),
)
TARGET_W, TARGET_H = 960, 540
PREVIEW_FPS = 12  # look-development cadence; final master remains 24 fps


def _grade(rgb):
    """Match the master mux's restrained blue-blocker channel mix."""
    graded = rgb.astype(np.float32)
    graded[:, :, 0] *= 1.018
    graded[:, :, 1] *= 1.006
    graded[:, :, 2] *= 0.88
    return np.clip(graded, 0, 255).astype(np.uint8)


def _font(size):
    for path in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\bahnschrift.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Building static plates...", flush=True)
    plates = load()
    contact_tiles = []

    for start, end, slug in PREVIEWS:
        path = os.path.join(OUT, f"{slug}.mp4")
        silent_path = os.path.join(OUT, f"{slug}_silent.mp4")
        total = int(round((end - start) * PREVIEW_FPS))
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}",
            "-pix_fmt", "rgb24", "-r", str(PREVIEW_FPS), "-i", "-",
            "-vf", f"scale={TARGET_W}:{TARGET_H}:flags=lanczos",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", silent_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        selected = {0, total // 4, total // 2, (3 * total) // 4, total - 1}
        frames = {}
        try:
            for index in range(total):
                t = start + index / PREVIEW_FPS
                frame = _grade(render(plates, t))
                if index in selected:
                    frames[t] = Image.fromarray(
                        cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
                    )
                proc.stdin.write(frame.tobytes())
                if (index + 1) % 120 == 0 or index + 1 == total:
                    print(f"{slug}: {index + 1}/{total} frames", flush=True)
        finally:
            proc.stdin.close()
            rc = proc.wait()
        if rc:
            raise RuntimeError(f"ffmpeg failed with exit code {rc} while writing {silent_path}")
        if not os.path.exists(SOURCE_AUDIO):
            raise FileNotFoundError(f"Master soundtrack source is missing: {SOURCE_AUDIO}")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", silent_path,
            "-ss", str(start), "-i", SOURCE_AUDIO, "-t", f"{end-start:.3f}",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
            "-shortest", "-movflags", "+faststart", path,
        ], check=True)
        os.remove(silent_path)
        print(f"Wrote {path} ({end-start:.1f}s, 960x540, {PREVIEW_FPS}fps, with soundtrack)", flush=True)

        for t in sorted(frames):
            tile = Image.new("RGB", (TARGET_W, TARGET_H + 30), (13, 17, 22))
            tile.paste(frames[t], (0, 0))
            draw = ImageDraw.Draw(tile)
            draw.text((10, TARGET_H + 5), f"{slug.replace('_', ' ')}  |  {t:.2f}s",
                      fill=(232, 220, 196), font=_font(15))
            contact_tiles.append(tile)

    cols = 3
    rows = (len(contact_tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TARGET_W, rows * (TARGET_H + 30)), (7, 10, 15))
    for i, tile in enumerate(contact_tiles):
        sheet.paste(tile, ((i % cols) * TARGET_W, (i // cols) * (TARGET_H + 30)))
    sheet.save(os.path.join(OUT, "transition_clips_contact_sheet.jpg"), quality=91)
    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as handle:
        handle.write(
            "SHE transition motion studies\n"
            "One 960x540/12fps mirage study rendered from the revised compositor with\n"
            "the matching segment of the existing master soundtrack.\n"
            "It uses the current blue-blocker grade; refraction affects only the ridge\n"
            "background before actor compositing, leaving actor contours intact.\n"
            "Three other short clips retain the soundtrack and are references cut from\n"
            "the existing master; they do not contain the new mirage code. None of\n"
            "these previews replaces the 1080p master.\n"
        )
    print(os.path.join(OUT, "transition_clips_contact_sheet.jpg"), flush=True)


if __name__ == "__main__":
    main()
