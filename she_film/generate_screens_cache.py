"""Build looping Buddha-focused B-roll for the ending telemetry CRTs.

Each monitor gets a distinct, slow Ken Burns loop made from the project's
existing Buddha plates. The cache is deliberately self-contained so Colab and
local renders use the same exact screen footage.
"""
import os
import cv2
import numpy as np

FPS = 24
LOOP_SECONDS = 22
FRAME_COUNT = FPS * LOOP_SECONDS
# The source studio monitors are portrait-oriented. Build portrait feeds so
# the compositor never has to stretch landscape Buddha footage into them.
FRAME_SIZE = (240, 320)


def _read_plate(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Required Buddha screen plate not found: {path}")
    # The compositor stores and blends plates in RGB order; OpenCV loads BGR.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _detail_crop(image, center_x=0.52, center_y=0.43, zoom=1.8):
    """Create a high-resolution alternate shot from the same Buddha plate."""
    h, w = image.shape[:2]
    crop_w = max(16, int(w / zoom))
    crop_h = max(16, int(h / zoom))
    cx, cy = center_x * w, center_y * h
    x0 = int(np.clip(cx - crop_w / 2, 0, w - crop_w))
    y0 = int(np.clip(cy - crop_h / 2, 0, h - crop_h))
    return image[y0:y0 + crop_h, x0:x0 + crop_w].copy()


def _ken_burns(image, frame_index, variant=0):
    """Aspect-preserving crop with a slow, deterministic zoom and pan."""
    out_w, out_h = FRAME_SIZE
    src_h, src_w = image.shape[:2]
    aspect = out_w / out_h
    base_w = min(src_w, int(src_h * aspect))
    base_h = min(src_h, int(base_w / aspect))
    phase = frame_index / float(FRAME_COUNT)
    # Ease in/out so the loop has no abrupt pan reversal.
    travel = 0.5 - 0.5 * np.cos(phase * 2.0 * np.pi)
    zoom = 1.02 + 0.15 * (0.5 - 0.5 * np.cos(phase * 2.0 * np.pi))
    crop_w = max(8, int(base_w / zoom))
    crop_h = max(8, int(base_h / zoom))
    center_x = src_w * (0.5 + (travel - 0.5) * (0.08 if variant == 0 else -0.06))
    center_y = src_h * (0.5 + (travel - 0.5) * 0.035)
    x0 = int(np.clip(center_x - crop_w / 2, 0, src_w - crop_w))
    y0 = int(np.clip(center_y - crop_h / 2, 0, src_h - crop_h))
    crop = image[y0:y0 + crop_h, x0:x0 + crop_w]
    frame = cv2.resize(crop, FRAME_SIZE, interpolation=cv2.INTER_LANCZOS4)

    # A restrained film transfer: slightly warm phosphor, not noisy or glitchy.
    frame = frame.astype(np.float32)
    frame[:, :, 2] *= 1.035
    frame[:, :, 0] *= 0.94
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    nx = (xx - (out_w - 1) * 0.5) / ((out_w - 1) * 0.5)
    ny = (yy - (out_h - 1) * 0.5) / ((out_h - 1) * 0.5)
    vignette = np.clip(1.0 - 0.11 * (nx * nx + ny * ny), 0.80, 1.0)
    frame *= vignette[:, :, None]
    frame[::2] *= 0.92  # soft 1970s CRT scanlines
    return np.clip(frame, 0, 255).astype(np.uint8)


def _make_loop(plates, variant=0):
    frames = []
    for i in range(FRAME_COUNT):
        half = FRAME_COUNT // 2
        source_a, source_b = plates
        if i < half:
            local = i
            source = source_a
        else:
            local = i - half
            source = source_b
        # Keep each plate on screen long enough to read, with a clean 1-second dissolve.
        cross = FPS
        if local < cross:
            previous = source_b if i < half else source_a
            a = local / float(cross)
            current = _ken_burns(source, i, variant)
            prior = _ken_burns(previous, (i - cross) % FRAME_COUNT, variant)
            current = cv2.addWeighted(prior, 1.0 - a, current, a, 0.0)
        elif local >= half - cross:
            next_source = source_b if i < half else source_a
            a = (local - (half - cross)) / float(cross)
            current = _ken_burns(source, i, variant)
            nxt = _ken_burns(next_source, (i + cross) % FRAME_COUNT, variant)
            current = cv2.addWeighted(current, 1.0 - a, nxt, a, 0.0)
        else:
            current = _ken_burns(source, i, variant)
        frames.append(current)
    return np.asarray(frames, dtype=np.uint8)


def build_screens_cache(output_dir=None):
    film_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(film_dir, "cache")
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(film_dir, "cache")

    # Both programs remain unmistakably Buddha footage: a wide cosmic statue
    # reveal on the left, and close portrait / mudra details on the right.
    cosmic_buddha = _read_plate(os.path.join(cache_dir, "buddha_cosmic_demigod.jpg"))
    left_plates = [cosmic_buddha, _detail_crop(cosmic_buddha, 0.52, 0.55, 1.45)]
    right_plates = [
        _detail_crop(cosmic_buddha, 0.52, 0.37, 2.15),
        _detail_crop(cosmic_buddha, 0.52, 0.69, 2.45),
    ]
    arr_left = _make_loop(left_plates, variant=0)
    arr_right = _make_loop(right_plates, variant=1)
    out_npz = os.path.join(output_dir, "screens_90s.npz")
    np.savez_compressed(out_npz, scr_left=arr_left, scr_right=arr_right)
    print(f"[CACHE] Built Buddha B-roll CRT loops: {out_npz} ({os.path.getsize(out_npz) / 1024 / 1024:.2f} MB)")
    return out_npz


if __name__ == "__main__":
    build_screens_cache()
