"""SHE Crystal Nocturne: restrained, scene-specific teal-and-orange finishing grade."""
import cv2
import numpy as np

from she_film.clock import lerp, smooth


def scene_intensity(t):
    """Return act-specific grade intensity (0..1) across the 7:25 timeline."""
    t = float(t)
    if t < 20.0:
        return 0.40
    if t < 113.0:
        return lerp(0.40, 0.50, smooth((t - 20.0) / 16.0))
    if t < 125.0:
        return lerp(0.50, 0.60, smooth((t - 113.0) / 8.0))
    if t < 192.0:
        return lerp(0.60, 0.78, smooth((t - 125.0) / 10.0))
    if t < 296.0:
        return lerp(0.78, 0.62, smooth((t - 192.0) / 10.0))
    if t < 360.0:
        return lerp(0.62, 0.74, smooth((t - 296.0) / 12.0))
    if t < 428.0:
        return lerp(0.74, 0.68, smooth((t - 360.0) / 12.0))
    return lerp(0.68, 0.36, smooth((t - 428.0) / 8.0))


def crystal_nocturne(frame, t, intensity_override=None):
    """Apply restrained teal-and-orange Crystal Nocturne grade to float32 BGR frame."""
    k = float(intensity_override) if intensity_override is not None else scene_intensity(t)
    if k <= 0.01:
        return frame

    img = frame.astype(np.float32, copy=True)

    # 1. Subtle black lift (+2..+4) + clean S-curve contrast (+8..+14%)
    black_lift = 3.2 * k
    contrast = 1.0 + 0.11 * k
    img = (img - 128.0) * contrast + 128.0 + black_lift

    # 2. Split-tone: teal/cyan in shadows, warm amber in highlights + gentle warm temp (+6..+10)
    lum = np.clip((0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]) / 255.0, 0.0, 1.0)
    shadow_w = ((1.0 - lum) ** 1.6)[:, :, None]
    hi_w = (lum ** 1.5)[:, :, None]

    # BGR offsets
    teal_shadows = np.array([10.5, 6.0, -8.0], dtype=np.float32) * k
    amber_highlights = np.array([-7.5, 3.0, 11.0], dtype=np.float32) * k
    warm_temp = np.array([-2.5, 1.0, 4.5], dtype=np.float32) * k

    img += shadow_w * teal_shadows + hi_w * amber_highlights + warm_temp

    # 3. Selective teal/orange saturation (+10..+18%) with restrained global saturation (+3%)
    gray = (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2])[:, :, None]
    cool_mask = np.clip((img[:, :, 0:1] - img[:, :, 2:3]) / 45.0, 0.0, 1.0)
    warm_mask = np.clip((img[:, :, 2:3] - img[:, :, 0:1]) / 45.0, 0.0, 1.0)
    sat_boost = 1.0 + (0.03 + 0.14 * cool_mask + 0.11 * warm_mask) * k
    img = gray + (img - gray) * sat_boost

    # 4. Crisp micro-contrast / clarity (+5..+10) on water, foliage, mushroom detail
    blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
    img += (img - blurred) * (0.09 * k)

    # 5. Restrained light-source bloom (only bright sun corona, stars, aurora, Buddha heart)
    bright = np.maximum(0.0, img - 205.0)
    if float(bright.max()) > 1.0:
        bloom = cv2.GaussianBlur(bright, (0, 0), 9.0)
        img += bloom * (0.14 * k)

    return np.clip(img, 0.0, 255.0)
