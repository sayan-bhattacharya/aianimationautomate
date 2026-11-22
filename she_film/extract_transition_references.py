"""Extract short art-review references, keeping the film soundtrack.

These clips document the current rendered master only. They do not contain the
new in-compositor horizon mirage; see mirage_mushroom_to_whorl_fir_sample.mp4.
"""
import os
import subprocess


ROOT = os.path.dirname(os.path.dirname(__file__))
SOURCE = os.path.join(ROOT, "she_full_song_master_blueblocker.mp4")
OUT = os.path.join(ROOT, "preview_720p", "transition_clips_mirage")
CLIPS = (
    (178.0, 12.0, "reference_saucer_landing_and_departure"),
    (240.0, 12.0, "reference_dinosaur_passage"),
    (398.0, 16.0, "reference_forest_decay_and_mushroom_return"),
)


def main():
    if not os.path.exists(SOURCE):
        raise FileNotFoundError(f"Rendered reference master not found: {SOURCE}")
    os.makedirs(OUT, exist_ok=True)
    for start, duration, slug in CLIPS:
        path = os.path.join(OUT, f"{slug}.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", SOURCE,
            "-t", str(duration), "-vf", "scale=960:540:flags=lanczos",
            "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart", path,
        ], check=True)
        print(f"Wrote {path} ({duration:.0f}s at source time {start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
