"""Glitter paths. A lamp below the horizon leaves no path."""
import numpy as np

from she_film.clock import HEIGHT, HORIZON, WIDTH

STEP = 4
QH = (HEIGHT - HORIZON) // STEP
QW = WIDTH // STEP
QY = np.arange(QH, dtype=np.float32)[:, None]
QX = np.arange(QW, dtype=np.float32)[None, :]
QF = QY / float(QH)


def paint(base_row, t, lamps, stillness=0.0, audio_feats=None):
    """base_row is (water_h, 1, 3). lamps is a list of (x, weight, rgb, width0, width1)."""
    wq = np.broadcast_to(base_row[::STEP], (QH, QW, 3)).copy()
    wave_factor = max(0.0, 1.0 - float(stillness))

    # Note energy from 24fps audio feature matrix (Col 9: note onsets, Col 5-6: harmonic mids)
    note_energy = 0.0
    if audio_feats is not None:
        frame_idx = min(len(audio_feats) - 1, max(0, int(t * 24)))
        note_energy = float(audio_feats[frame_idx, 9] * 0.70 + audio_feats[frame_idx, 5] * 0.30)
    
    for lamp in lamps:
        origin_x, weight, rgb, width0, width1 = lamp[:5]
        ripple_strength = float(lamp[5]) if len(lamp) > 5 else 0.0
        if weight < 0.03 or origin_x is None:
            continue
        wobble = wave_factor * (
            12.0 * np.sin(QY * 0.22 + t * 1.4) + 5.0 * np.sin(QY * 0.47 - t * 2.0)
        )
        path_x = (origin_x / STEP) + (QW * 0.5 - origin_x / STEP) * QF * 0.08 + wobble / STEP
        half = (width0 + width1 * QF) / STEP
        dist = np.abs(QX - path_x)
        mask = np.exp(-0.5 * (dist / (half * 0.42 + 0.4)) ** 2) * (1.0 - QF * 0.22) * weight
        tint = np.array(rgb, dtype=np.float32)
        wq = wq * (1.0 - mask[:, :, None] * 0.72) + tint * (mask[:, :, None] * 0.72)
        
        # Crest highlights flatten out completely when still
        if wave_factor > 0.01:
            crest = np.clip(np.sin(QY * 0.55 + np.sin(QX * 0.05 + t * 0.7) + t * 0.4) - 0.72, 0, 1) * wave_factor
            near = np.exp(-0.5 * ((QX - origin_x / STEP) / (half * 0.9 + 1)) ** 2)
            wq += (crest * near * weight)[:, :, None] * tint * 0.35

        # A low craft adds broad concentric disturbances around its marine coordinate.
        if ripple_strength > 0.001:
            cx = origin_x / STEP
            cy = 8.0
            dx = QX - cx
            dy = (QY - cy) * 2.4
            radius = np.hypot(dx, dy)
            rings = np.sin(radius * 0.20 - t * 4.2) * np.exp(-radius / 54.0)
            ring_mask = np.clip(rings, 0.0, 1.0) * ripple_strength * wave_factor
            wq += ring_mask[:, :, None] * tint[None, None, :] * 0.34

    # Audio-reactive acoustic ocean ripples: radiating concentric wave perturbations highlighting musical notes
    # Sparser spatial frequency (wavelength ~60px) for clean, non-crowded organic ripples
    if wave_factor > 0.01 and (note_energy > 0.03 or (55.0 <= t <= 95.0)):
        ripple_centers = [(int(QW * 0.35), 14), (int(QW * 0.65), 18)]
        for rc_x, rc_y in ripple_centers:
            d_ellipse = np.hypot(QX - rc_x, (QY - rc_y) * 2.2)
            # Sparser spatial wave phase: 0.11 rad/px (~57px wavelength, wide elegant spacing)
            wave_phase = np.sin(d_ellipse * 0.11 - t * 3.2)
            decay = np.exp(-0.5 * (d_ellipse / 68.0) ** 2) * (1.0 - QF * 0.22)
            # Smooth acoustic amplitude
            amp = (0.05 + 0.42 * note_energy) * wave_factor
            sig = wave_phase * decay * amp
            pos_crest = np.clip(sig, 0, 0.40)
            neg_trough = np.clip(-sig, 0, 0.30)
            cyan_tint = np.array([175, 235, 250], dtype=np.float32)
            wq += pos_crest[:, :, None] * cyan_tint * 0.40
            wq -= neg_trough[:, :, None] * 14.0

    # Sparser wave line ripples: 8 elegant, widely spaced wave crest lines (reduced from 16)
    if wave_factor > 0.01:
        ink = np.array([150, 205, 214], dtype=np.float32) * wave_factor
        xs = np.arange(QW)
        for k in range(8):
            prog = ((k / 8.0) + (t * 0.025) % 1.0) % 1.0
            yb = int(QH * (prog ** 1.35))
            amp = (1.4 + 2.8 * prog) * wave_factor
            # Modulate wave crest perturbation slightly with note resonance
            amp_mod = amp * (1.0 + 0.35 * note_energy)
            yy = np.clip((yb + amp_mod * np.sin(xs * 0.08 + t * 0.7 + k * 1.2)).astype(np.int32), 0, QH - 1)
            wq[yy, xs] = wq[yy, xs] * (1.0 - 0.35 * wave_factor) + ink * 0.35
            
    return np.repeat(np.repeat(wq, STEP, 0), STEP, 1)[: HEIGHT - HORIZON, :WIDTH]
