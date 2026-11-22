"""Subtle transition refraction for the land and air behind scene actors.

The displacement is intentionally applied before mushrooms, trees, dinosaurs,
and craft are composited. It gives the horizon a mirage-like shimmer while
leaving the designed silhouettes crisp and recognizable.
"""
import cv2
import numpy as np

from she_film.clock import HEIGHT, HORIZON, WIDTH, smooth


def _window(t, start, peak, end):
    if t <= start or t >= end:
        return 0.0
    if t < peak:
        return smooth((t - start) / max(1e-6, peak - start))
    return 1.0 - smooth((t - peak) / max(1e-6, end - peak))


def amount(t):
    """Return a restrained amount for vegetation changes, 0..1."""
    return max(
        _window(t, 166.0, 177.0, 190.0),  # peaks during the brief clearing between forms
        _window(t, 388.0, 402.0, 416.0),  # forest dissolves; mushrooms return
    )


def music_energy(audio_feats, t):
    """Return a smoothed, normalized midrange/onset envelope for slow visual breathing."""
    if audio_feats is None or len(audio_feats) == 0:
        return 0.0
    center = min(len(audio_feats) - 1, max(0, int(float(t) * 24.0)))
    lo, hi = max(0, center - 9), min(len(audio_feats), center + 10)
    # The half-second window rejects frame-to-frame flicker but still follows musical phrases.
    raw = float(np.mean(audio_feats[lo:hi, 9] * 0.65 + audio_feats[lo:hi, 5] * 0.35))
    return float(np.clip((raw - 0.24) / 0.58, 0.0, 1.0))


def focus_window(t):
    """Only apply the soft-focus illusion during authored transformation passages."""
    return max(
        _window(float(t), 166.0, 179.0, 194.0),
        0.62 * _window(float(t), 228.0, 248.0, 276.0),
        0.72 * _window(float(t), 388.0, 402.0, 416.0),
    )


def soften_background(frame, t, audio_feats):
    """Add a very restrained, music-breathing focus haze behind the actors only."""
    focus = focus_window(t)
    energy = music_energy(audio_feats, t)
    if focus < 0.002 or energy < 0.01:
        return frame

    y0, y1 = max(0, HORIZON - 240), HORIZON
    patch = frame[y0:y1]
    sigma = 0.55 + 0.95 * energy
    softened = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma, sigmaY=sigma)
    world_y = np.arange(y0, y1, dtype=np.float32)[:, None]
    near_horizon = np.clip((world_y - (HORIZON - 240.0)) / 240.0, 0.0, 1.0) ** 0.8
    mix = focus * (0.025 + 0.115 * energy) * near_horizon[:, :, None]
    frame[y0:y1] = patch * (1.0 - mix) + softened * mix
    return frame


def refract_background(frame, t):
    """Subtly refract the horizon band only; actor silhouettes stay untouched.

    `frame` is the float32 RGB background before foreground actors are added.
    Horizontal displacement peaks at 3.0 px and vertical at 0.9 px.
    The field fades toward the sky and lower waterline, avoiding a full-frame
    liquify effect or unstable edges.
    """
    strength = amount(float(t))
    if strength < 0.002:
        return frame

    y0, y1 = max(0, HORIZON - 230), min(HEIGHT, HORIZON + 105)
    band = frame[y0:y1]
    h, w = band.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    world_y = yy + y0

    # Keep the strongest lensing close to the ridge where the new forms emerge.
    vertical = np.exp(-0.5 * ((world_y - (HORIZON - 16.0)) / 86.0) ** 2)
    anchor_field = np.zeros((1, w), dtype=np.float32)
    for anchor in (340, 640, 980, 1260, 1540):
        anchor_field = np.maximum(
            anchor_field,
            np.exp(-0.5 * ((np.arange(w, dtype=np.float32)[None, :] - anchor) / 210.0) ** 2),
        )
    envelope = strength * vertical * anchor_field

    phase = xx * 0.018 + world_y * 0.041 - float(t) * 1.35
    dx = 3.0 * np.sin(phase) * envelope
    dy = 0.9 * np.sin(xx * 0.010 - float(t) * 0.72 + world_y * 0.018) * envelope
    map_x = np.clip(xx + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy + dy, 0, h - 1).astype(np.float32)
    warped = cv2.remap(band, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Softly blend the refracted background, with zero effect outside its field.
    mix = np.clip(envelope * 0.90, 0.0, 0.78)[:, :, None]
    frame[y0:y1] = band * (1.0 - mix) + warped * mix
    return frame
