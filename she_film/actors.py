"""Stars, snow, the craft, the mushroom. Each one is a paste, not a redraw."""
import math
import cv2
import numpy as np

from she_film import mirage
from she_film.clock import (
    FIR_EMERGENCE_END,
    FIR_EMERGENCE_START,
    HEIGHT,
    HORIZON,
    WIDTH,
    crimson,
    herd,
    lerp,
    moon,
    smooth,
    sky_weights,
    sun,
)

PTERO_WING_CYCLE_START = 126
PTERO_WING_CYCLE_END = 158  # exclusive: 32-frame, 1.33s full flap with matching wing pose

# Visually matched walking spans, selected away from the source clips' abrupt
# first/last poses. End indices are exclusive.
DINO_WALK_CYCLES = {0: (122, 191), 1: (85, 158), 2: (0, 48), 3: (37, 153)}


def _dino_walk_frame(anim, dino_id, t):
    if anim is None or anim.n_frames == 0:
        return None
    start, end = DINO_WALK_CYCLES.get(dino_id, (0, anim.n_frames))
    end = min(end, anim.n_frames)
    frames = anim.frames[start:end]
    if len(frames) < 2:
        return frames[0] if frames else anim.get_frame(t)
    # Start motion when the off-screen procession begins entering at 232s.
    # The frame-edge fade therefore reveals a walk already in progress.
    pos = (max(0.0, t - 232.0) * 24.0) % len(frames)
    i = int(pos)
    frac = pos - i
    nxt = (i + 1) % len(frames)
    current = frames[i] * (1.0 - frac) + frames[nxt] * frac
    return current.astype(np.float32, copy=False)


def _edge_presence(x, sprite_width):
    """Ease sprites in/out over one sprite-width as they cross the frame edge."""
    half = sprite_width * 0.5
    if x < half:
        return smooth((x + half) / max(1.0, sprite_width))
    if x > WIDTH - half:
        return smooth((WIDTH + half - x) / max(1.0, sprite_width))
    return 1.0


def pterodactyl_loop_frames(anim_ptero):
    """Return the visually matched, single wingbeat segment from the Flow clip."""
    if anim_ptero is None or anim_ptero.n_frames < PTERO_WING_CYCLE_END:
        return None
    return anim_ptero.frames[PTERO_WING_CYCLE_START:PTERO_WING_CYCLE_END]


def _pterodactyl_frame(anim_ptero, t):
    cycle = pterodactyl_loop_frames(anim_ptero)
    if cycle is None:
        return anim_ptero.get_frame(t) if anim_ptero is not None else None
    # Phase from flight entrance, not song time, so every pass starts consistently.
    frame_index = int(round(max(0.0, t - 242.0) * anim_ptero.fps)) % len(cycle)
    return cycle[frame_index]


def _celestial_shadow(x, t, scale=1.0):
    """Approximate ground-shadow direction from the active sun or moon."""
    day, warm, night = sky_weights(t)
    candidates = []
    sun_body = sun(t)
    if sun_body is not None:
        energy = day * smooth((sun_body[1] + 12.0) / 30.0)
        if energy > 0.03:
            candidates.append((energy, sun_body, (27, 24, 19)))
    moon_body = moon(t)
    if moon_body is not None:
        energy = night * 0.58 * smooth((moon_body[1] + 12.0) / 30.0)
        if energy > 0.03:
            candidates.append((energy, moon_body, (15, 21, 32)))
    if candidates:
        energy, body, color = max(candidates, key=lambda item: item[0])
        side = 1 if x >= body[0] else -1
        low_sun_stretch = 1.0 + 0.35 * (1.0 - min(1.0, max(0.0, body[1] / 420.0)))
        dx = int(side * (15.0 + 28.0 * energy) * low_sun_stretch * scale)
        return dx, int((4.0 + 8.0 * energy) * scale), color
    return 0, int(3.0 * scale), (20, 22, 28)


def over(frame, sprite, x, y, alpha=1.0):
    """sprite is float32 RGBA. Color is 0-255. Alpha is 0-1."""
    if sprite is None or alpha < 0.004:
        return
    sh, sw = sprite.shape[:2]
    x0, y0 = int(x), int(y)
    sx0 = 0 if x0 >= 0 else -x0
    sy0 = 0 if y0 >= 0 else -y0
    dx0, dy0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(WIDTH, x0 + sw), min(HEIGHT, y0 + sh)
    if x1 <= dx0 or y1 <= dy0:
        return
    crop = sprite[sy0: sy0 + (y1 - dy0), sx0: sx0 + (x1 - dx0)]
    a = crop[:, :, 3:4] * alpha
    dst = frame[dy0:y1, dx0:x1]
    dst += (crop[:, :, :3] - dst) * a


def _water_wave_refraction(reflection, t, anchor_x, wave_factor, audio_feats, phase=0.0, parallax=1.0):
    """Break a mirrored actor silhouette along moving, music-responsive wavelets following wave crests and troughs with depth parallax."""
    if reflection is None or wave_factor < 0.01:
        return reflection
    h, w = reflection.shape[:2]
    energy = mirage.music_energy(audio_feats, t)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = np.clip(yy / max(1.0, h - 1.0), 0.0, 1.0)
    # Dynamic wave velocity: higher initially, slowing smoothly to stillness as Tanpura drone swells
    speed_factor = max(0.12, wave_factor)
    t_wave_x = float(t) * (3.1 * speed_factor + 0.3)
    t_wave_y = float(t) * (2.4 * speed_factor + 0.3)
    phase_x = yy * 0.16 + xx * 0.024 + t_wave_x + anchor_x * 0.007 + phase
    phase_y = xx * 0.032 - t_wave_y + yy * 0.08 + phase
    # Parallax displacement: closer objects displace more vigorously than distant ones
    dx = wave_factor * parallax * (4.8 + 4.2 * energy) * (0.35 + 0.65 * depth) * np.sin(phase_x)
    dy = wave_factor * parallax * (3.4 + 2.8 * energy) * (0.25 + 0.75 * depth) * np.cos(phase_y)
    map_x = np.clip(xx - dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy + dy, 0, h - 1).astype(np.float32)
    out = cv2.remap(reflection, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # Crest break lines: crests shimmer and break naturally along wave ridges
    crest = np.abs(np.sin(phase_x * 0.75))
    out[:, :, 3] *= 1.0 - wave_factor * (0.08 + 0.12 * energy) * crest
    return out


def stars(frame, plates, t, night, moon_body):
    """Twinkle. Moonlight quiets them, most of all the ones beside the moon."""
    if night < 0.08:
        return
    xs, ys, mag, phase = plates["stars"]
    twinkle = 0.62 + 0.38 * np.sin(t * 1.7 + phase)
    wash = night
    if moon_body is not None:
        mx, alt = moon_body
        my = HORIZON - alt
        dist = np.hypot(xs - mx, ys - my)
        local = np.exp(-0.5 * (dist / 220.0) ** 2)
        height = min(1.0, max(0.0, alt / 280.0))
        wash = night * (1.0 - 0.42 * height) * (1.0 - 0.72 * local)
        twinkle = 0.78 + (0.22 + 0.2 * (1.0 - height)) * np.sin(t * 1.7 + phase)
    alpha = np.clip(mag * twinkle * wash, 0, 1)
    keep = alpha > 0.08
    if not np.any(keep):
        return
    x = xs[keep].astype(np.int32)
    y = ys[keep].astype(np.int32)
    col = (210 + 35 * alpha[keep])[:, None] * np.array([0.90, 0.94, 1.0], dtype=np.float32)
    frame[y, x] = frame[y, x] * (1.0 - alpha[keep][:, None]) + col * alpha[keep][:, None]


def snow(frame, plates, t, amount, moon_body, ship_body):
    if amount < 0.02:
        return
    xs, phase, speed, size = plates["flakes"]
    n = max(1, int(len(xs) * amount))
    y = ((phase[:n] + (t - 90.0) * speed[:n] * 0.045) % 1.0) * (HEIGHT + 40) - 20
    x = (xs[:n] + np.sin(t * 0.32 + phase[:n]) * 36.0) % WIDTH
    cool = np.array([226, 232, 240], dtype=np.float32)
    if moon_body is not None:
        cool = cool * 0.85 + np.array([180, 196, 220], dtype=np.float32) * 0.15
    warm = np.array([255, 214, 170], dtype=np.float32)
    xi = x.astype(np.int32)
    yi = np.clip(y.astype(np.int32), 0, HEIGHT - 1)
    color = np.broadcast_to(cool, (n, 3)).copy()
    if ship_body is not None:
        near = np.exp(-0.5 * ((x - ship_body[0]) / 180.0) ** 2) * ship_body[2]
        color = color * (1.0 - near[:, None]) + warm * near[:, None]
    a = (0.55 + 0.4 * amount) * np.clip(size[:n] / 1.4, 0.55, 1.0)
    paint = color * a[:, None]
    keep = (1.0 - a)[:, None]
    for dy, fade in ((0, 1.0), (2, 0.65), (4, 0.35)):
        yy = np.clip(yi + dy, 0, HEIGHT - 1)
        xx = xi
        frame[yy, xx] = frame[yy, xx] * (1.0 - a[:, None] * fade) + color * (a[:, None] * fade)


MUSHROOM_STANDS = [
    (340, 24.0, 0.75),   # Stand 0: Leftmost Stand (starts wave, x=340)
    (640, 20.0, 1.15),   # Stand 1: Hero Center-Left Stand (x=640)
    (980, 28.0, 0.70),   # Stand 2: Center-Right Stand (x=980)
    (1260, 50.0, 1.15),  # Stand 3: 4th Mushroom (Hero East Stand, x=1260) - becomes larger and holds
    (1540, 54.0, 0.65),  # Stand 4: Far East Stand (x=1540)
]
MUSHROOM_HOLD_INDICES = (2, 3)  # Both Stand 2 and Stand 3 (4th mushroom) reach mature size and hold permanently
MUSHROOM_GROWTH_SECONDS = 8.0
MUSHROOM2_DECAY_HOLD_END = 12.0
MUSHROOM2_DECAY_END = 20.0
MUSHROOM2_FINAL_DECAY_SCALE = 0.68


def mushroom_target_frame(growth_sprite, age):
    """Use the Scene 1 growth sequence, then hold its exact terminal mature frame."""
    if growth_sprite is None or growth_sprite.n_frames == 0:
        return None
    if age < MUSHROOM_GROWTH_SECONDS:
        return growth_sprite.get_frame(age, loop=False)
    return growth_sprite.frames[-1]


def mushroom2_lifecycle_frame(plates, age):
    """Hold Mushroom 2 at maturity, then visibly shrink and wither in place."""
    growth = plates.get("anim_mushroom_growth")
    if growth is None or not growth.n_frames:
        return None, 1.0
    if age < MUSHROOM_GROWTH_SECONDS:
        return growth.get_frame(age, loop=False), 1.0
    mature = growth.frames[-1]
    if age <= MUSHROOM2_DECAY_HOLD_END:
        return mature, 1.0

    p = smooth((age - MUSHROOM2_DECAY_HOLD_END) /
               (MUSHROOM2_DECAY_END - MUSHROOM2_DECAY_HOLD_END))
    decayed = mature.astype(np.float32).copy()
    rgb = decayed[:, :, :3]
    luminance = np.mean(rgb, axis=2, keepdims=True)
    rgb = (rgb * (1.0 - 0.38 * p) + luminance * (0.38 * p)) * (1.0 - 0.12 * p)
    rgb += p * np.array([4.0, 1.5, -2.0], dtype=np.float32)
    decayed[:, :, :3] = np.clip(rgb, 0.0, 255.0)
    # Crown-first loss of opacity suggests withering; roots remain present and grounded.
    crown_to_root = np.linspace(0.90, 0.10, decayed.shape[0], dtype=np.float32)[:, None]
    decayed[:, :, 3] *= 1.0 - p * crown_to_root
    decay_scale = 1.0 - (1.0 - MUSHROOM2_FINAL_DECAY_SCALE) * p
    return decayed, decay_scale


# Backward compatibility alias
mushroom3_target_frame = mushroom_target_frame
MUSHROOM3_TARGET_INDEX = 2


def _parasympathetic_swell_mod(t):
    """Headspace-style parasympathetic exhale during the 1m53s (113.0s–121.5s) musical swell."""
    if t < 113.0 or t > 125.0:
        return 1.0, 1.0
    if t < 121.5:
        p = smooth((t - 113.0) / 8.5)
        return 1.0 - 0.32 * p, 1.0 - 0.10 * p
    r = smooth((t - 121.5) / 3.5)
    return 0.68 + 0.32 * r, 0.90 + 0.10 * r


def _get_active_mushroom_stands(plates, t):
    """
    Returns list of (scaled_frame, x, y, grow_scale, alpha_fade) for all active stands at time t.
    Uses Flow AI-generated green-screen timelapse for gradual natural growth,
    and the Flow AI-generated subtle living breathing/swaying loop for mature stands.
    """
    vs_growth = plates.get("anim_mushroom_growth")
    vs_idle = plates.get("anim_mushroom_idle")
    ground_profile = plates.get("ridge_surface_y")

    stands = []
    if vs_growth is None or vs_idle is None:
        stages = plates.get("life", [])
        if stages:
            spr = stages[min(len(stages) - 1, int(t * 0.1) % len(stages))]
            stands.append((spr, 640 - spr.shape[1] // 2, (HORIZON - 16) - spr.shape[0], 1.0, 1.0))
        return stands

    for stand_index, (x, t0, b_scale) in enumerate(MUSHROOM_STANDS):
        if t < t0:
            continue
        age = t - t0
        # Smooth scaling across the film to show growth throughout the video
        grow_scale = b_scale * (0.85 + 0.38 * min(1.0, max(0.0, (t - 25.0) / 75.0)))

        # Slower, gentle frequency with left-to-right passing wave effect across neighboring stands
        # Delay propagates from leftmost stand (x=340) to rightmost stand (x=1540)
        wave_delay = (x - 340.0) / 380.0
        
        if stand_index == 1:
            raw, decay_scale = mushroom2_lifecycle_frame(plates, age)
            grow_scale *= decay_scale
            # User feedback: Depth in Z direction should NOT just be translucency/opacity reduction.
            # Keep opacity solid (0.94) and apply depth-of-field Gaussian blur with subtle dithering.
            a_fade = min(0.96, max(0.85, age / 0.8))
        elif age < MUSHROOM_GROWTH_SECONDS:
            raw = (mushroom_target_frame(vs_growth, age)
                   if stand_index in MUSHROOM_HOLD_INDICES
                   else vs_growth.get_frame(age, loop=False))
            a_fade = min(1.0, age / 0.8)
        elif stand_index in MUSHROOM_HOLD_INDICES:
            # Stand 2 and Stand 3 (4th mushroom) resolve to the exact terminal mature frame
            # from Mushroom Growth Scene 1 and remain held permanently at full size with
            # continuous reflection integration, never shrinking back into a sprout.
            raw = mushroom_target_frame(vs_growth, age)
            a_fade = 1.0
        else:
            # Minimized frequency (0.45x) and neighbor wave phase delay
            # Ping-pong the Flow life-cycle plate: mature -> gradual decay ->
            # small -> regrowth. It never wraps mature directly to a sprout.
            phase = (8.0 - wave_delay + max(0.0, age - 8.0) * 0.45) % 16.0
            idle_t = phase if phase <= 8.0 else 16.0 - phase
            raw = vs_idle.get_frame(idle_t, loop=False)
            a_fade = 1.0

        if raw is None:
            continue

        # Firm stalk grounding: shift the frame content down so the stalk base touches the bottom edge
        a_raw = raw[..., 3]
        rows = np.where(a_raw.max(axis=1) > 0.15)[0]
        if len(rows):
            stalk_bot = rows[-1]
            shift_down = (raw.shape[0] - 1) - stalk_bot
            if shift_down > 0:
                M = np.float32([[1, 0, 0], [0, 1, shift_down]])
                raw = cv2.warpAffine(raw, M, (raw.shape[1], raw.shape[0]),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        swell_h_mod, swell_w_mod = _parasympathetic_swell_mod(t)
        tw = max(16, int(raw.shape[1] * grow_scale * swell_w_mod))
        th = max(16, int(raw.shape[0] * grow_scale * swell_h_mod))
        scaled = cv2.resize(raw, (tw, th), interpolation=cv2.INTER_LINEAR)

        # Optical Depth of Field & Atmospheric Perspective (Reverse-engineered from Flow reference):
        # Stand 3 is the primary in-focus hero focal point (pin-sharp); distant and near stands have progressive optical defocus and atmospheric haze.
        day, warm, night = sky_weights(t)
        sky_ambient = np.array([
            210.0 * day + 130.0 * warm + 38.0 * night,
            205.0 * day + 95.0 * warm + 26.0 * night,
            198.0 * day + 70.0 * warm + 20.0 * night
        ], dtype=np.float32)

        depth_profiles = {
            3: (0.0, 0.0),      # Hero stand (focal plane): pin-sharp focus, full punchy contrast
            0: (1.25, 0.04),    # Left foreground stand: subtle portrait optical softness
            1: (1.8, 0.12),     # Mid-ground stand: mild optical defocus & atmospheric softening
            2: (1.6, 0.16),     # Mid-distance stand: subtle depth defocus & atmospheric veil
            4: (2.8, 0.26),     # Far background stand: authentic aerial perspective & defocus
        }
        sigma, haze = depth_profiles.get(stand_index, (0.0, 0.0))

        if sigma > 0.1:
            ksize = int(round(sigma * 3)) * 2 + 1
            scaled[:, :, :3] = cv2.GaussianBlur(scaled[:, :, :3], (ksize, ksize), sigma)
            scaled[:, :, 3] = cv2.GaussianBlur(scaled[:, :, 3], (ksize, ksize), sigma)

        if haze > 0.01:
            scaled[:, :, :3] = scaled[:, :, :3] * (1.0 - haze) + sky_ambient * haze

        # Root each sprite on the visible, topmost land contour at its x.
        ground_y = HORIZON - 16
        if ground_profile is not None and 0 <= int(x) < len(ground_profile):
            ground_y = min(HORIZON - 1, max(0, int(ground_profile[int(x)])))

        top = ground_y - th

        # Seamless Root Integration into Beach Soil:
        # Match local time-of-day illumination dynamically to current sun/twilight/night state
        day, warm, night = sky_weights(t)
        dune_sand_color = np.array([
            186.0 * warm + 150.0 * day + 42.0 * night,
            96.0 * warm + 142.0 * day + 40.0 * night,
            64.0 * warm + 160.0 * day + 68.0 * night
        ], dtype=np.float32)

        # Seamless root embedding without empty pixels:
        # Softly blend root base tone into local dune sand where roots clasp the ground
        sand_encase_h = max(10, int(th * 0.16))
        sand_up = np.linspace(0.0, 0.72, sand_encase_h, dtype=np.float32)[:, None]
        stalk_mask = (scaled[th - sand_encase_h:th, :, 3:4] > 0.10).astype(np.float32)
        scaled[th - sand_encase_h:th, :, :3] += (dune_sand_color - scaled[th - sand_encase_h:th, :, :3]) * sand_up[:, :, None] * stalk_mask

        if t >= 400.0:
            # Bioluminescent celestial blue glow on returning mushrooms, modulated by musical phrasing
            return_progress = min(1.0, max(0.0, (t - 400.0) / 12.0))
            music_e = mirage.music_energy(plates.get("audio_feats"), t)
            cap_rows = int(th * 0.65)
            pulse = (0.68 + 0.24 * math.sin(t * 3.2 + wave_delay * 2.0)) * (0.85 + 0.35 * music_e)
            cap_mask = scaled[:cap_rows, :, 3] * np.linspace(1.0, 0.0, cap_rows, dtype=np.float32)[:, None]
            blue_lum = np.array([235.0, 175.0, 45.0], dtype=np.float32)  # cyan-blue radiance in BGR
            scaled[:cap_rows, :, :3] += blue_lum * (cap_mask[:, :, None] * 0.45 * pulse * return_progress)
            scaled[:, :, :3] = np.clip(scaled[:, :, :3], 0, 255)

        stands.append((scaled, x - tw // 2, top, grow_scale, a_fade, wave_delay, ground_y))

    return stands


def life(frame, plates, t, alpha=1.0):
    if alpha < 0.02:
        return
    stands = _get_active_mushroom_stands(plates, t)
    for stand_index, stand in enumerate(stands):
        spr, x, y, scale, a_fade, wave_delay = stand[:6]
        ground_y = stand[6] if len(stand) > 6 else HORIZON - 1
        
        # Grounding Ambient Occlusion (AO) contact shadow directly joining stalk to sand
        # Constrained vertically so it naturally tapers to zero without hard flat clipping
        max_depth_to_water = max(2, HORIZON - 1 - ground_y)
        ao_w = max(18, int(spr.shape[1] * 0.28))
        ao_h = max(2, min(5, int(max_depth_to_water * 0.6)))
        ao_mask = np.zeros((ao_h * 2 + 1, ao_w * 2 + 1), dtype=np.float32)
        cv2.ellipse(ao_mask, (ao_w, ao_h), (ao_w, ao_h), 0, 0, 360, 1.0, -1)
        ao_mask = cv2.GaussianBlur(ao_mask, (0, 0), max(1.0, ao_h * 0.6))
        sy0 = max(0, ground_y - ao_h)
        sy1 = min(HORIZON, ground_y + ao_h + 1)
        sx0 = max(0, int(x + spr.shape[1] // 2) - ao_w)
        sx1 = min(WIDTH, int(x + spr.shape[1] // 2) + ao_w + 1)
        if sy1 > sy0 and sx1 > sx0:
            patch = ao_mask[sy0 - (ground_y - ao_h):sy1 - (ground_y - ao_h),
                            sx0 - (int(x + spr.shape[1] // 2) - ao_w):sx1 - (int(x + spr.shape[1] // 2) - ao_w)]
            ao_factor = 1.0 - patch * (0.35 * alpha * a_fade)
            frame[sy0:sy1, sx0:sx1] = np.clip(
                frame[sy0:sy1, sx0:sx1].astype(np.float32) * ao_factor[:, :, None],
                0, 255
            ).astype(np.uint8)

        # Draw firmly grounded mushroom sprite
        over(frame, spr, x, y, alpha * a_fade)

        # Foreground Sand Dune Drift Lip / Sand Cradle:
        # A soft organic sand drift that wraps in front of root tips with zero rectangular edges
        lip_w = max(18, int(spr.shape[1] * 0.30))
        lip_h = max(2, min(4, int(max_depth_to_water * 0.5)))
        lip_mask = np.zeros((lip_h * 2 + 1, lip_w * 2 + 1), dtype=np.float32)
        cv2.ellipse(lip_mask, (lip_w, lip_h), (lip_w, lip_h), 0, 0, 360, 1.0, -1)
        lip_mask = cv2.GaussianBlur(lip_mask, (0, 0), max(1.0, lip_h * 0.5))
        
        lx0 = max(0, int(x + spr.shape[1] // 2) - lip_w)
        lx1 = min(WIDTH, int(x + spr.shape[1] // 2) + lip_w + 1)
        ly0 = max(0, ground_y - lip_h)
        ly1 = min(HORIZON, ground_y + lip_h + 1)
        if ly1 > ly0 and lx1 > lx0:
            patch = lip_mask[(ly0 - (ground_y - lip_h)):(ly1 - (ground_y - lip_h)),
                             (lx0 - (int(x + spr.shape[1] // 2) - lip_w)):(lx1 - (int(x + spr.shape[1] // 2) - lip_w))]
            day, warm, night = sky_weights(t)
            dune_sand_color = np.array([
                186.0 * warm + 150.0 * day + 42.0 * night,
                96.0 * warm + 142.0 * day + 40.0 * night,
                64.0 * warm + 160.0 * day + 68.0 * night
            ], dtype=np.float32)
            sand_crest = np.array([24.0 * warm + 14.0 * day + 6.0 * night, 14.0 * warm + 10.0 * day + 6.0 * night, 8.0 * warm + 6.0 * day + 10.0 * night], dtype=np.float32)
            sand_lip_col = dune_sand_color + sand_crest
            frame_patch = frame[ly0:ly1, lx0:lx1].astype(np.float32)
            blend_a = patch[:, :, None] * (0.82 * alpha * a_fade)
            frame[ly0:ly1, lx0:lx1] = np.clip(frame_patch * (1.0 - blend_a) + sand_lip_col * blend_a, 0, 255).astype(np.uint8)

        # Ascending bioluminescent blue spores on returning mushrooms (t >= 400s)
        if t >= 400.0:
            return_progress = min(1.0, max(0.0, (t - 400.0) / 12.0))
            music_e = mirage.music_energy(plates.get("audio_feats"), t)
            for sp_idx in range(5):
                sp_phase = sp_idx * 1.3 + stand_index * 0.8
                sp_y = ground_y - int(spr.shape[0] * 0.65) - int(((t * 24.0 + sp_phase * 35.0) % 95.0))
                sp_x = int(x + spr.shape[1] // 2 + math.sin(t * 1.8 + sp_phase) * (spr.shape[1] * 0.30))
                sp_sz = 2 + (sp_idx % 2)
                sp_a = alpha * a_fade * return_progress * (0.40 + 0.35 * math.sin(t * 4.0 + sp_phase)) * (0.82 + 0.38 * music_e)
                if 0 <= sp_x < WIDTH and 0 <= sp_y < HORIZON and sp_a > 0.05:
                    cv2.circle(frame, (sp_x, sp_y), sp_sz + 1, (240, 180, 50), -1, cv2.LINE_AA)
                    cv2.circle(frame, (sp_x, sp_y), sp_sz, (255, 240, 190), -1, cv2.LINE_AA)


def life_reflect(frame, plates, t, stillness=0.0):
    wave_factor = max(0.0, 1.0 - float(stillness))
    stands = _get_active_mushroom_stands(plates, t)

    for stand_index, stand in enumerate(stands):
        spr, x, y, scale, a_fade, wave_delay = stand[:6]
        sh, sw = spr.shape[:2]
        ref_h = int(sh * 0.82)
        if ref_h < 4:
            continue
        flipped = cv2.resize(spr[::-1], (sw, ref_h), interpolation=cv2.INTER_LINEAR)
        # Rich, radiant water-surface gradient reflecting the mushroom bloom on the ocean
        fade = np.linspace(0.68, 0.0, ref_h, dtype=np.float32)[:, None, None]
        flipped_rgba = flipped.copy()
        flipped_rgba[:, :, 3:4] *= fade
        
        # Depth parallax: closer stands refract more on the waves
        parallax = 1.40 if stand_index in (0, 1) else (1.0 if stand_index == 2 else 0.75)
        flipped_rgba = _water_wave_refraction(
            flipped_rgba, t, x, wave_factor, plates.get("audio_feats"), phase=x * 0.003, parallax=parallax
        )

        # Ocean wave reflection wobble matching passing wave dynamics with Tanpura stillness easing
        speed_factor = max(0.12, wave_factor)
        wobble = int(math.sin(t * (1.8 * speed_factor + 0.2) - (x - 340.0) * 0.006) * (6.5 * wave_factor * parallax))
        
        # Bioluminescent blue reflection for returning mushrooms
        if t >= 400.0:
            blue_ref_tint = np.array([230.0, 170.0, 50.0, 0.0], dtype=np.float32)
            flipped_rgba[:, :, :3] += blue_ref_tint[:3] * (flipped_rgba[:, :, 3:4] * 0.30)
            flipped_rgba = np.clip(flipped_rgba, 0, 255)

        over(frame, flipped_rgba, x + wobble, HORIZON - 2, a_fade * 0.78)


FOREST_ANCHORS = (340, 640, 980, 1260, 1540)
# Ridge 0 is farthest inland and ridge 3 is the nearest shelf. Keep the
# center-right trees inland and all roots above the waterline.
FOREST_GROUND_LAYERS = (2, 2, 0, 0, 1)
FOREST_DEPTH_VISIBILITY = (0.96, 1.0, 0.96, 0.94, 0.90)
FOREST_REFLECTION_VISIBILITY = (0.12, 0.16, 0.035, 0.03, 0.05)
FOREST_REFLECTION_HEIGHT = (0.38, 0.45, 0.25, 0.23, 0.26)
FOREST_GROUND_MARGIN = 18
FOREST_EMERGENCE_DELAY = (0.44, 0.72, 0.0, 0.10, 0.26)
FOREST_EMERGENCE_DURATION = 0.15


def forest_ground_y(plates, index):
    """Return the selected ridge shelf row for this tree anchor."""
    layers = plates.get("ridge_surface_layers")
    if layers is None:
        row = int(plates["ridge_surface_y"][FOREST_ANCHORS[index]])
    else:
        row = int(layers[FOREST_GROUND_LAYERS[index]][FOREST_ANCHORS[index]])
    return min(row, HORIZON - FOREST_GROUND_MARGIN)


def _ridge_visibility_mask(plates, depth, left, top, width, height):
    """Visibility mask behind all land shelves that are closer to camera."""
    visible = np.ones((height, width), dtype=np.float32)
    ridges = plates.get("ridges", ())
    if depth >= len(ridges) - 1:
        return visible

    x0, y0 = int(left), int(top)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(WIDTH, x0 + width), min(HEIGHT, y0 + height)
    if dx1 <= dx0 or dy1 <= dy0:
        return visible
    sx0, sy0 = dx0 - x0, dy0 - y0
    sx1, sy1 = sx0 + dx1 - dx0, sy0 + dy1 - dy0
    blocked = np.zeros((dy1 - dy0, dx1 - dx0), dtype=np.float32)
    for ridge in ridges[depth + 1:]:
        np.maximum(blocked, ridge[dy0:dy1, dx0:dx1], out=blocked)
    visible[sy0:sy1, sx0:sx1] *= 1.0 - blocked
    return visible


def forest_emergence_alpha(t, index):
    """Stagger whole-tree reveals from the rear ridge toward the near hero tree."""
    p = np.clip((float(t) - FIR_EMERGENCE_START) /
                (FIR_EMERGENCE_END - FIR_EMERGENCE_START), 0.0, 1.0)
    local = np.clip((p - FOREST_EMERGENCE_DELAY[index]) / FOREST_EMERGENCE_DURATION, 0.0, 1.0)
    return float(smooth(float(local)))


def _tree_wind_and_contact(sprite, t, tree_x, tree_index, tree_depth):
    """Apply gradual 5m52s (352s) wind-chill leaf flutter and dinosaur-brush physics.

    Roots at the bottom remain pinned (displacement = 0) while upper leaves
    flutter in the breeze and mid/lower foliage bends elastically when passing
    dinosaurs brush against the plant (especially the leftmost stand at x=340).
    """
    wind_strength = smooth((t - 344.0) / 18.0) if t >= 344.0 else 0.0

    # Check proximity to any dinosaur in the caravan for physical brush interaction
    dino_push = 0.0
    if tree_depth >= 1 and 232.0 <= t < 405.0:
        for d_id, dx_dino in herd(t):
            dist = dx_dino - tree_x
            if abs(dist) < 115.0:
                prox = smooth(1.0 - abs(dist) / 115.0)
                # Dinosaurs march East->West (-x), pushing foliage leftward with elastic bounce
                bounce = 1.0 + 0.25 * math.sin(t * 7.5 + d_id)
                dino_push -= 11.0 * prox * bounce

    if wind_strength < 0.005 and abs(dino_push) < 0.05:
        return sprite

    h, w = sprite.shape[:2]
    v = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]  # 1 at crown, 0 at root
    # Lock the bottom 6% of rows so root grounding never shifts
    pin = np.clip((v - 0.04) / 0.12, 0.0, 1.0)

    dx_row = np.zeros((h, 1), dtype=np.float32)
    if wind_strength >= 0.005:
        gust = math.sin(t * 1.9 + tree_x * 0.005) + 0.45 * math.sin(t * 3.7 - tree_x * 0.008)
        rows = np.arange(h, dtype=np.float32)[:, None]
        leaf_flutter = np.sin(rows * 0.085 - t * 6.8 + tree_index * 1.4) * 0.42
        dx_row += wind_strength * (6.0 * gust * (v ** 1.5) + 2.6 * leaf_flutter * (v ** 1.05)) * pin

    if abs(dino_push) >= 0.05:
        contact_profile = np.exp(-0.5 * ((v - 0.36) / 0.28) ** 2) * pin
        dx_row += dino_push * contact_profile

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xx - dx_row).astype(np.float32)
    return cv2.remap(sprite, map_x, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def forest(frame, plates, t, presence, decay=0.0, depth_layers=None, respect_ridges=True):
    """Composite the selected whorl-firs with a soft celestial key and late-life fade."""
    if presence < 0.01:
        return
    day, warm, night = sky_weights(t)
    sun_body, moon_body = sun(t), moon(t)
    growing = decay < 0.001 and FIR_EMERGENCE_START <= t < FIR_EMERGENCE_END
    for index, (spr, x) in enumerate(zip(plates.get("forest", ()), FOREST_ANCHORS)):
        depth = FOREST_GROUND_LAYERS[index]
        if depth_layers is not None and depth not in depth_layers:
            continue
        # Reveal one complete silhouette at a time, rear-to-front, instead of
        # exposing every tree as a transparent full-height ghost.
        opacity = forest_emergence_alpha(t, index) if growing else presence
        tree_presence = opacity * FOREST_DEPTH_VISIBILITY[index]
        h, w = spr.shape[:2]
        ground_y = forest_ground_y(plates, index)
        # A dim, directionally offset ground shadow ties each cutout to the active lamp.
        dx, dy, shadow_rgb = _celestial_shadow(x, t, scale=min(1.3, w / 150.0))
        shadow_w = max(30, int(w * 0.62))
        shadow_h = max(8, int(12 + day * 5 + night * 3))
        shadow = np.zeros((shadow_h * 3, shadow_w * 3), np.uint8)
        cv2.ellipse(shadow, (shadow_w, shadow_h), (shadow_w // 2, shadow_h // 2), 0, 0, 360, 92, -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), max(2.0, shadow_h * 0.8)).astype(np.float32) / 255.0
        sx, sy = int(x + dx - shadow_w), ground_y - shadow_h // 2 + dy
        if respect_ridges:
            shadow *= _ridge_visibility_mask(plates, depth, sx, sy, shadow.shape[1], shadow.shape[0])
        y0, x0 = max(0, sy), max(0, sx)
        y1, x1 = min(HORIZON, sy + shadow.shape[0]), min(WIDTH, sx + shadow.shape[1])
        if y1 > y0 and x1 > x0:
            src = shadow[y0-sy:y1-sy, x0-sx:x1-sx, None] * tree_presence * 0.34
            tint = np.asarray(shadow_rgb, np.float32)
            frame[y0:y1, x0:x1] += (tint - frame[y0:y1, x0:x1]) * src

        sprite = spr.copy()
        sprite = _tree_wind_and_contact(sprite, t, x, index, depth)
        if index == 0:
            # Subtle portrait depth-of-field blur on nearest foreground tree with large angular diameter
            sprite[:, :, :3] = cv2.GaussianBlur(sprite[:, :, :3], (5, 5), 1.25)
        # Daylight keeps the firs natural; moonlight cools the shaded side subtly.
        warmth = min(0.18, warm * 0.14 + (0.07 if sun_body is not None else 0.0))
        cool = night * 0.09
        sprite[:, :, 0] *= 1.0 + warmth - cool * 0.35
        sprite[:, :, 1] *= 1.0 + warmth * 0.22
        sprite[:, :, 2] *= 1.0 - warmth * 0.30 + cool
        if decay > 0.001:
            # Slow crown-first desaturation and dissolution, rather than a uniform pop-out.
            rows = np.linspace(0.35, 1.0, h, dtype=np.float32)[:, None]
            sprite[:, :, 3] *= 1.0 - decay * rows * 0.78
            leaf = np.array([108.0, 94.0, 75.0], np.float32)
            sprite[:, :, :3] += (leaf - sprite[:, :, :3]) * decay * 0.62
        # Align the source concept's five-tree rhythm to the established mushroom anchors.
        root_rows = plates.get("forest_root_rows", ())
        root_row = root_rows[index] if index < len(root_rows) else h
        # Seat root base 5px into the soil so root spurs firmly penetrate the beach sand
        left, top = int(x - w / 2), ground_y - root_row + 5

        # Seamless Root Integration into Beach Soil:
        # Mask each column of the root flare to conform to the local ridge dune contour
        layers = plates.get("ridge_surface_layers")
        if layers is not None and depth < len(layers):
            surf = layers[depth]
            for c in range(w):
                xw = np.clip(left + c, 0, WIDTH - 1)
                y_soil = surf[xw]
                r_soil = int(y_soil - top)
                if 0 <= r_soil < h:
                    depth_in_sand = np.arange(h - r_soil, dtype=np.float32)
                    sprite[r_soil:, c, 3] *= np.clip(1.0 - depth_in_sand / 6.0, 0.0, 1.0)
                elif r_soil < 0:
                    sprite[:, c, 3] = 0.0

        if respect_ridges:
            sprite[:, :, 3] *= _ridge_visibility_mask(plates, depth, left, top, w, h)

        # Natural ambient occlusion (AO) contact shadow where roots touch the beach sand
        ao_w = max(16, int(w * 0.30))
        ao_h = max(3, int(ao_w * 0.16))
        ao_mask = np.zeros((ao_h * 2 + 1, ao_w * 2 + 1), dtype=np.float32)
        cv2.ellipse(ao_mask, (ao_w, ao_h), (ao_w, ao_h), 0, 0, 360, 1.0, -1)
        ao_mask = cv2.GaussianBlur(ao_mask, (0, 0), max(1.5, ao_h * 0.7))
        sy0 = max(0, ground_y - ao_h)
        sy1 = min(HORIZON, ground_y + ao_h + 1)
        sx0 = max(0, int(x) - ao_w)
        sx1 = min(WIDTH, int(x) + ao_w + 1)
        if sy1 > sy0 and sx1 > sx0:
            patch = ao_mask[sy0 - (ground_y - ao_h):sy1 - (ground_y - ao_h),
                            sx0 - (int(x) - ao_w):sx1 - (int(x) - ao_w)]
            ao_factor = 1.0 - patch * (0.32 * tree_presence)
            frame[sy0:sy1, sx0:sx1] = np.clip(
                frame[sy0:sy1, sx0:sx1].astype(np.float32) * ao_factor[:, :, None],
                0, 255
            ).astype(np.uint8)

        over(frame, sprite, left, top, tree_presence)

        # Foreground Sand Dune Drift Collar around tree root flare
        tree_lip_w = max(26, int(w * 0.38))
        tree_lip_h = max(4, int(h * 0.030))
        tree_lip_mask = np.zeros((tree_lip_h * 2 + 1, tree_lip_w * 2 + 1), dtype=np.float32)
        cv2.ellipse(tree_lip_mask, (tree_lip_w, tree_lip_h), (tree_lip_w, tree_lip_h), 0, 0, 180, 1.0, -1)
        tree_lip_mask = cv2.GaussianBlur(tree_lip_mask, (0, 0), max(1.2, tree_lip_h * 0.6))
        tlx0 = max(0, int(x) - tree_lip_w)
        tlx1 = min(WIDTH, int(x) + tree_lip_w + 1)
        tly0 = max(0, ground_y - 2)
        tly1 = min(HORIZON, ground_y + tree_lip_h + 1)
        if tly1 > tly0 and tlx1 > tlx0:
            patch = tree_lip_mask[(tly0 - (ground_y - 2)):(tly1 - (ground_y - 2)),
                                  (tlx0 - (int(x) - tree_lip_w)):(tlx1 - (int(x) - tree_lip_w))]
            day, warm, night = sky_weights(t)
            dune_sand_color = np.array([
                186.0 * warm + 150.0 * day + 42.0 * night,
                96.0 * warm + 142.0 * day + 40.0 * night,
                64.0 * warm + 160.0 * day + 68.0 * night
            ], dtype=np.float32)
            sand_col = dune_sand_color + np.array([20.0 * warm + 12.0 * day + 6.0 * night, 12.0 * warm + 8.0 * day + 6.0 * night, 6.0 * warm + 4.0 * day + 8.0 * night], dtype=np.float32)
            f_patch = frame[tly0:tly1, tlx0:tlx1].astype(np.float32)
            blend_a = patch[:, :, None] * (0.80 * tree_presence)
            frame[tly0:tly1, tlx0:tlx1] = np.clip(f_patch * (1.0 - blend_a) + sand_col * blend_a, 0, 255).astype(np.uint8)


def forest_reflect(frame, plates, t, presence, decay=0.0, stillness=0.0):
    if presence < 0.01:
        return
    wave_factor = max(0.0, 1.0 - float(stillness))
    day, warm, night = sky_weights(t)
    for index, (spr, x) in enumerate(zip(plates.get("forest", ()), FOREST_ANCHORS)):
        growing = decay < 0.001 and FIR_EMERGENCE_START <= t < FIR_EMERGENCE_END
        tree_alpha = forest_emergence_alpha(t, index) if growing else presence
        tree_presence = (tree_alpha * FOREST_DEPTH_VISIBILITY[index]
                         * FOREST_REFLECTION_VISIBILITY[index])
        h, w = spr.shape[:2]
        ref_src = _tree_wind_and_contact(spr.copy(), t, x, index, FOREST_GROUND_LAYERS[index])
        ref_h = max(6, int(h * FOREST_REFLECTION_HEIGHT[index]))
        reflected = cv2.resize(ref_src[::-1], (w, ref_h), interpolation=cv2.INTER_LINEAR).copy()
        fade = np.linspace(0.36, 0.0, ref_h, dtype=np.float32)[:, None]
        reflected[:, :, 3] *= fade * (1.0 - decay * 0.55)
        reflected[:, :, 0] *= 1.0 + warm * 0.08
        reflected[:, :, 2] *= 1.0 + night * 0.06
        parallax = 1.35 if index in (0, 1) else (1.0 if index == 2 else 0.70)
        reflected = _water_wave_refraction(
            reflected, t, x, wave_factor, plates.get("audio_feats"), phase=index * 0.7, parallax=parallax
        )
        speed_factor = max(0.12, wave_factor)
        wobble = int(math.sin(t * (1.8 * speed_factor + 0.2) - (x - 340.0) * 0.006) * 5.0 * wave_factor * parallax)
        over(frame, reflected, x - w / 2 + wobble, HORIZON + 2, tree_presence)


def solid(frame, t, lamp, alpha):
    """Removed: tetrahedron/polyhedron removed per creative master plan."""
    return


def birds(frame, plates, t, amount):
    if amount < 0.04:
        return
    ink = np.array([28, 22, 26], np.float32)
    water_ink = np.array([18, 48, 62], np.float32)
    n = 8 + int(amount * 8)
    
    # 1. Flocks flying in sky + direct reflections in ocean waves
    for i in range(n):
        x = int((t * 68.0 + i * 175.0) % (WIDTH + 80) - 40)
        y = int(96 + (i % 5) * 24 + math.sin(t * 1.7 + i) * 12)
        a = min(0.92, 0.38 + 0.52 * amount)
        wing = int(math.sin(t * 7.5 + i * 1.4) * 3)
        
        # Draw bird in sky
        for s in range(6):
            for px, py in ((x + s, y + s - wing), (x + 12 - s, y + s - wing)):
                if 0 <= px < WIDTH and 0 <= py < HORIZON - 35:
                    frame[py, px] = frame[py, px] * (1.0 - a) + ink * a
                    
        # Direct water wave reflections
        ref_y = HORIZON + int((HORIZON - y) * 0.42 + math.sin(t * 2.6 + x * 0.04) * 5)
        ref_x = x + int(math.sin(t * 3.4 + ref_y * 0.07) * 7)
        if HORIZON <= ref_y < HEIGHT - 2 and 0 <= ref_x < WIDTH - 14:
            ref_a = a * 0.38
            frame[ref_y:ref_y+2, ref_x:ref_x+14] = frame[ref_y:ref_y+2, ref_x:ref_x+14] * (1.0 - ref_a) + water_ink * ref_a

    # 2. Soaring Pterodactyl flying alongside bird flocks in the sky
    anim_ptero = plates.get("anim_pterodactyl")
    ptero = _pterodactyl_frame(anim_ptero, t) if anim_ptero is not None else plates.get("pokemon_pterodactyl")
    if ptero is not None and 242.0 <= t <= 342.0:
        # Flip horizontally first so head faces forward towards direction of travel (flying West / left)
        ptero = cv2.flip(ptero, 1)
        p_progress = (t - 242.0) / 100.0
        # Continuous majestic panoramic glide across the dawn sky
        p_x = lerp(WIDTH + 180.0, -220.0, smooth(p_progress))
        p_y = 130.0 + 16.0 * math.sin(p_progress * math.pi * 3.0)
        # Aerodynamic banking angle derived from flight trajectory derivative
        p_tilt = -math.cos(p_progress * math.pi * 3.0) * 7.5
        pw, ph = ptero.shape[1], ptero.shape[0]
        M_ptero = cv2.getRotationMatrix2D((pw / 2, ph / 2), p_tilt, 1.0)
        rot_ptero = cv2.warpAffine(ptero, M_ptero, (pw, ph), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        over(frame, rot_ptero, p_x - pw / 2, p_y - ph / 2, min(0.95, 0.45 + 0.5 * amount))

        pterodactyl_reflect(frame, rot_ptero, p_x, p_y, t, min(0.95, 0.45 + 0.5 * amount))

    # Note: Marine animals (flying fish, plesiosaur, leviathan) temporarily removed per creative plan
    # for dedicated multi-frame Flow animation enhancement.


def pterodactyl_reflect(frame, sprite, center_x, sky_y, t, alpha=0.65, stillness=0.0):
    """Paint a compressed, altitude-shifted flying-reptile reflection on the sea."""
    if sprite is None or alpha < 0.02:
        return
    sh, sw = sprite.shape[:2]
    altitude = max(0.0, HORIZON - float(sky_y))
    # Higher flight produces a shorter, more displaced reflection with a wider ripple.
    scale_y = float(np.clip(0.48 - altitude / 1800.0, 0.24, 0.48))
    ref_h = max(8, int(sh * scale_y))
    ref_y = HORIZON + int(altitude * 0.46)
    wave = max(0.0, 1.0 - float(stillness))
    wobble = math.sin(t * 3.1 + center_x * 0.024) * (11.0 + altitude * 0.012) * wave
    ref_w = max(8, int(sw * (0.92 + 0.08 * scale_y)))
    reflection = cv2.resize(sprite[::-1], (ref_w, ref_h), interpolation=cv2.INTER_AREA).copy()

    day, warm, night = sky_weights(t)
    tint = (np.array([56, 76, 92], np.float32) * day
            + np.array([110, 70, 42], np.float32) * warm
            + np.array([38, 88, 154], np.float32) * night)
    reflection[:, :, :3] = np.clip(reflection[:, :, :3] * 0.30 + tint, 0, 255)
    fresnel = float(np.clip(1.0 - altitude / 1100.0, 0.25, 0.85))
    fade = np.linspace(0.42, 0.015, ref_h, dtype=np.float32)[:, None, None]
    ripple = 0.72 + 0.28 * np.sin(np.arange(ref_h, dtype=np.float32)[:, None, None] * 0.36 + t * 4.0)
    reflection[:, :, 3:4] *= fade * ripple * fresnel
    if ref_y < HEIGHT - 1:
        over(frame, reflection, center_x - ref_w / 2 + wobble, ref_y, alpha * fresnel * 0.62)


def _draw_grazing_sauropod(frame, x, y, neck_bend, t):
    """Draws hero sauropod with neck bending down into plant fronds to graze."""
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (500, 280), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    hide = (120, 108, 80, 255)
    shade = (70, 58, 48, 255)
    belly = (165, 148, 115, 255)
    light = (215, 185, 130, 255)
    rim = (255, 225, 150, 255)

    # Torso and lower flank countershading
    d.polygon([(110, 125), (170, 95), (290, 95), (350, 120), (370, 155), (320, 185), (180, 185), (120, 155)], fill=hide)
    d.ellipse((140, 85, 330, 185), fill=hide)
    d.ellipse((150, 120, 310, 180), fill=belly)

    # Tail held high
    d.polygon([(120, 135), (70, 115), (20, 85), (5, 75), (25, 95), (80, 135), (130, 155)], fill=hide)
    d.line([(5, 75), (20, 85), (70, 115), (120, 135)], fill=rim, width=2)

    # Animated neck IK bending down to canopy foliage
    base_n = (325, 120)
    hx = 360 + int(neck_bend * 45)
    hy = 28 + int(neck_bend * 125)
    mx = 345 + int(neck_bend * 26)
    my = 72 + int(neck_bend * 62)

    pts = [base_n, (mx, my), (hx, hy)]
    d.line(pts, fill=hide, width=32, joint="curve")
    d.line([(pts[0][0], pts[0][1] - 8), (pts[1][0], pts[1][1] - 8), (pts[2][0], pts[2][1] - 8)], fill=rim, width=3, joint="curve")

    # Head with chewing motion
    head_chew = math.sin(t * 18.0) * 3.5 if neck_bend > 0.4 else 0.0
    d.ellipse((hx - 20, hy - 14, hx + 20, hy + 14), fill=hide)
    d.ellipse((hx + 5, hy - 7, hx + 10, hy - 2), fill=(30, 24, 20, 255))
    d.ellipse((hx + 7, hy - 5, hx + 9, hy - 3), fill=(255, 235, 160, 255))
    
    if head_chew != 0.0:
        d.line([(hx - 10, hy + 10), (hx + 14, hy + 10 + int(head_chew))], fill=(22, 18, 16, 255), width=3)
        # Drifting green leaf nibbles
        for k in range(3):
            ly = hy + 14 + int((t * 24 + k * 7) % 18)
            lx = hx + 8 + int(math.sin(t * 8 + k) * 6)
            d.ellipse((lx - 2, ly - 2, lx + 2, ly + 2), fill=(75, 140, 55, 230))

    # Columnar limbs with walking cycle
    leg_phase = t * 3.5
    for idx, lx in enumerate([185, 295]):
        swing = math.sin(leg_phase + idx * math.pi) * 12.0
        d.polygon([(lx - 12 + swing*0.5, 150), (lx - 10 - swing*0.5, 245), (lx + 16 - swing*0.5, 245), (lx + 18 + swing*0.5, 155)], fill=shade)
        d.line([(lx - 8 - swing*0.5, 160), (lx - 8 - swing*0.5, 240)], fill=light, width=4)
        d.polygon([(lx - 14 - swing*0.5, 240), (lx + 20 - swing*0.5, 240), (lx + 18 - swing*0.5, 248), (lx - 16 - swing*0.5, 248)], fill=(190, 165, 120, 255))

    arr = np.asarray(im, dtype=np.float32)
    arr[:, :, 3] *= 1.0 / 255.0
    over(frame, arr, x, y, 0.96)


def ridge_elevation_y(x):
    """Calculates terrain elevation: dinosaurs walk gracefully along the lower coastal shelf close to the water body."""
    ridge = 52.0 * math.exp(-x / 800.0) + 30.0 * math.exp(-(WIDTH - x) / 700.0)
    wave = 10.0 * math.sin(x * 0.0035 + 0.6) + 5.0 * math.cos(x * 0.007)
    return HORIZON - 16.0 - (ridge + wave)


def beasts(frame, plates, herd, t=0.0):
    if not herd:
        return
    forms = plates["beasts"]
    
    # Audio accents affect light/shadow only; gait timing stays temporally smooth.
    audio_feats = plates.get("audio_feats")
    f_idx = min(len(audio_feats) - 1, max(0, int(t * 24.0))) if audio_feats is not None else 0
    piano_onset = float(audio_feats[f_idx, 9]) if audio_feats is not None else 0.0
    piano_energy = float(audio_feats[f_idx, 5]) if audio_feats is not None else 0.0
    
    t_piano = t

    for d_id, x in herd:
        ground = ridge_elevation_y(x)
        # 1. Hero Sauropod (d_id == 0) - Character-driven Grass-Sauropod
        if d_id == 0:
            anim_sauropod = plates.get("anim_sauropod")
            hero_spr = _dino_walk_frame(anim_sauropod, d_id, t_piano) if anim_sauropod is not None else plates.get("pokemon_sauropod")
            neck_bend = max(0.0, math.sin((t - 258.0) / 30.0 * math.pi)) if 258.0 <= t <= 288.0 else 0.0
            
            if hero_spr is not None:
                hw, hh = hero_spr.shape[1], hero_spr.shape[0]
                # Ground contact shadow under feet deepens subtly on rhythmic piano accents
                shad_rx = int(115 * (1.0 + 0.12 * piano_onset))
                shad_ry = int(12 * (1.0 + 0.14 * piano_onset))
                _, _, shadow_tint = _celestial_shadow(x, t)
                cv2.ellipse(frame, (int(x), int(ground)), (shad_rx, shad_ry), 0, 0, 360, shadow_tint, -1, cv2.LINE_AA)
                # Dynamic long slanting shadow downward towards ocean water edge
                shadow_dx, shadow_dy, _ = _celestial_shadow(x, t, scale=1.0)
                cv2.ellipse(frame, (int(x + shadow_dx), int(ground + shadow_dy)), (int(shad_rx * 1.25), int(shad_ry * 1.6)), -12, 0, 360, shadow_tint, -1, cv2.LINE_AA)
                sy = ground - hh + 6
                over(frame, hero_spr, x - hw / 2, sy, 0.98 * _edge_presence(x, hw))
                
                # Falling foliage nibbles during canopy grazing sync with piano roll
                if 258.0 <= t <= 288.0 and neck_bend > 0.3:
                    for k in range(3):
                        ly = int(sy + 40 + ((t * 22 + k * 7 + piano_onset * 5) % 25))
                        lx = int(x + hw * 0.28 + math.sin(t * 7 + k) * 6)
                        if 0 <= lx < WIDTH and 0 <= ly < HEIGHT:
                            cv2.circle(frame, (lx, ly), 2, (75, 138, 55), -1)
            else:
                sprite_y = ground - 280 + 4
                _, _, shadow_tint = _celestial_shadow(x, t)
                cv2.ellipse(frame, (int(x + 10), int(ground)), (140, 14), 0, 0, 360, shadow_tint, -1, cv2.LINE_AA)
                _draw_grazing_sauropod(frame, x - 250, sprite_y, neck_bend, t)
        else:
            # Trailing species: Hadrosaur (1), Ankylosaur (2), Raptor (3)
            anim_hadro = plates.get("anim_hadrosaur")
            anim_anky = plates.get("anim_ankylosaur")
            anim_raptor = plates.get("anim_raptor")
            if d_id == 1 and anim_hadro is not None:
                sprite = _dino_walk_frame(anim_hadro, d_id, t_piano)
            elif d_id == 2 and anim_anky is not None:
                sprite = _dino_walk_frame(anim_anky, d_id, t_piano)
            elif d_id == 3 and anim_raptor is not None:
                sprite = _dino_walk_frame(anim_raptor, d_id, t_piano)
            else:
                sprite = forms[d_id % len(forms)]
            sw, sh = sprite.shape[1], sprite.shape[0]
            sy = ground - sh + 5
            # Foot contact shadow permanently sized to this exact species
            shad_r = int(sw * 0.32 * (1.0 + 0.1 * piano_onset))
            shadow_dx, shadow_dy, shadow_tint = _celestial_shadow(x, t, scale=min(1.0, sw / 160.0))
            cv2.ellipse(frame, (int(x), int(ground)), (shad_r, 8), 0, 0, 360, shadow_tint, -1, cv2.LINE_AA)
            cv2.ellipse(frame, (int(x + shadow_dx), int(ground + shadow_dy)), (int(shad_r * 1.15), 11), -10, 0, 360, shadow_tint, -1, cv2.LINE_AA)
            over(frame, sprite, x - sw / 2, sy, 0.96 * _edge_presence(x, sw))


def beasts_reflect(frame, plates, herd, t=0.0, stillness=0.0):
    """Renders wave-rippled watery reflections of the dinosaur caravan on the ocean below."""
    if not herd:
        return
    audio_feats = plates.get("audio_feats")
    f_idx = min(len(audio_feats) - 1, max(0, int(t * 24.0))) if audio_feats is not None else 0
    piano_onset = float(audio_feats[f_idx, 9]) if audio_feats is not None else 0.0
    t_piano = t
    forms = plates["beasts"]
    
    # Stillness factor: when Tanpura swells (stillness -> 1.0), wobble attenuates to 0.0
    wave_factor = max(0.0, 1.0 - float(stillness))

    for d_id, x in herd:
        ground = ridge_elevation_y(x)
        if d_id == 0:
            anim_sauropod = plates.get("anim_sauropod")
            spr = _dino_walk_frame(anim_sauropod, d_id, t_piano) if anim_sauropod is not None else plates.get("pokemon_sauropod")
        elif d_id == 1:
            anim_hadro = plates.get("anim_hadrosaur")
            spr = _dino_walk_frame(anim_hadro, d_id, t_piano) if anim_hadro is not None else forms[1 % len(forms)]
        elif d_id == 2:
            anim_anky = plates.get("anim_ankylosaur")
            spr = _dino_walk_frame(anim_anky, d_id, t_piano) if anim_anky is not None else forms[2 % len(forms)]
        elif d_id == 3:
            anim_raptor = plates.get("anim_raptor")
            spr = _dino_walk_frame(anim_raptor, d_id, t_piano) if anim_raptor is not None else forms[3 % len(forms)]
        else:
            spr = forms[d_id % len(forms)]
        
        if spr is None:
            continue
        sw, sh = spr.shape[1], spr.shape[0]
        
        # When water is still (Tanpura in full force), wave_wobble is exactly 0.0!
        wave_wobble = math.sin(t * 2.8 + x * 0.02) * 5.0 * wave_factor
        ref_y = HORIZON + 2 + int(abs(wave_wobble) * 0.4)
        ref_x = int(x - sw / 2 + wave_wobble)
        
        if HORIZON <= ref_y < HEIGHT - 10 and -sw < ref_x < WIDTH:
            max_ref_h = min(sh, HEIGHT - ref_y)
            if max_ref_h > 10:
                # Derive the water tint from the actual exposure and visible moon altitude.
                day, warm, night = sky_weights(t)
                moon_body = moon(t)
                moon_visibility = (smooth((moon_body[1] + 12.0) / 30.0)
                                   if moon_body is not None else 0.0)
                night_factor = night * (0.25 + 0.75 * moon_visibility)
                sun_body = sun(t)
                sun_visibility = (day * smooth((sun_body[1] + 12.0) / 30.0)
                                  if sun_body is not None else 0.0)
                
                # Heightened opacity at night so reflection stands out boldly against dark sea
                top_fade = lerp(0.50, 0.88, night_factor) + 0.10 * float(stillness)
                bot_fade = lerp(0.02, 0.22, night_factor)
                fade = np.linspace(top_fade, bot_fade, max_ref_h, dtype=np.float32)[:, None, None]
                
                ref_sprite = spr[:max_ref_h][::-1].copy()
                ref_sprite[:, :, 3:4] *= fade * (0.80 + 0.20 * piano_onset) * _edge_presence(x, sw)
                
                # In daylight: warm amber-teal tint; At night: luminous moonlit silvery-blue radiance
                # Guaranteeing dinosaurs' reflections are vividly visible on the ocean
                r_add = 12.0 + 12.0 * sun_visibility + 42.0 * warm + 62.0 * night_factor + 8.0 * crimson(t)
                g_add = 16.0 + 8.0 * sun_visibility + 22.0 * warm + 96.0 * night_factor + 5.0 * crimson(t)
                b_add = 24.0 + 4.0 * sun_visibility + 8.0 * warm + 138.0 * night_factor + 14.0 * crimson(t)
                
                ref_sprite[:, :, 0] = np.clip(ref_sprite[:, :, 0] * 0.70 + r_add, 0, 255)
                ref_sprite[:, :, 1] = np.clip(ref_sprite[:, :, 1] * 0.78 + g_add, 0, 255)
                ref_sprite[:, :, 2] = np.clip(ref_sprite[:, :, 2] * 0.85 + b_add, 0, 255)
                
                # Subtle water crest lines shimmering through the reflection
                wave_shimmer = 1.0 + 0.18 * np.sin(np.arange(max_ref_h)[:, None, None] * 0.16 + t * 3.5) * wave_factor
                ref_sprite[:, :, 3:4] = np.clip(ref_sprite[:, :, 3:4] * wave_shimmer, 0, 255)
                ref_sprite = _water_wave_refraction(
                    ref_sprite, t, x, wave_factor, audio_feats, phase=d_id * 0.8, parallax=0.85
                )
                
                eff_over_alpha = lerp(0.60, 0.88, night_factor)
                over(frame, ref_sprite, ref_x, ref_y, eff_over_alpha)


_VAN_GOGH_HALO_SPRITE = None


def _get_van_gogh_halo_sprite():
    global _VAN_GOGH_HALO_SPRITE
    if _VAN_GOGH_HALO_SPRITE is None:
        dim = 96
        center = (dim - 1) / 2.0
        ys, xs = np.ogrid[:dim, :dim]
        dist = np.hypot(xs - center, ys - center) / (dim * 0.48)
        
        # Outer ethereal cyan aura
        outer = np.exp(-0.5 * (dist / 0.52) ** 2) * 0.42
        # Mid golden impasto ring (Van Gogh Starry Night halo)
        ring_dist = np.abs(dist - 0.48)
        ring = np.exp(-0.5 * (ring_dist / 0.13) ** 2) * 0.62
        
        alpha = np.clip(outer + ring, 0.0, 1.0)
        gold = np.array([125.0, 205.0, 255.0], dtype=np.float32)  # BGR
        cyan = np.array([245.0, 215.0, 160.0], dtype=np.float32)  # BGR
        
        bgr = (cyan * outer[:, :, None] + gold * ring[:, :, None]) / np.maximum(1e-4, outer[:, :, None] + ring[:, :, None])
        
        sprite = np.zeros((dim, dim, 4), dtype=np.float32)
        sprite[:, :, :3] = bgr
        sprite[:, :, 3] = alpha
        _VAN_GOGH_HALO_SPRITE = sprite
    return _VAN_GOGH_HALO_SPRITE


def _van_gogh_celestial_currents(frame, t, alpha):
    """Render subtle swirling starlight currents across the upper sky inspired by Van Gogh's Starry Night."""
    if alpha < 0.06:
        return
    t_slow = t * 0.12
    strip_w, strip_h = 384, 160
    strip = np.zeros((strip_h, strip_w, 3), dtype=np.float32)
    
    x_coords = np.arange(strip_w, dtype=np.float32)
    # Primary swirling stream across the sky
    y_wave1 = 65.0 + 26.0 * np.sin(x_coords * 0.018 + t_slow) + 14.0 * np.cos(x_coords * 0.035 - t_slow * 0.8)
    # Secondary counter-stream
    y_wave2 = 105.0 + 22.0 * np.sin(x_coords * 0.022 - t_slow * 1.1) + 16.0 * np.sin(x_coords * 0.01 + t_slow * 0.5)
    
    pts1 = np.column_stack([x_coords, np.clip(y_wave1, 0, strip_h - 1)]).astype(np.int32)
    cv2.polylines(strip, [pts1], False, (220, 195, 140), 4, cv2.LINE_AA)
    
    pts2 = np.column_stack([x_coords, np.clip(y_wave2, 0, strip_h - 1)]).astype(np.int32)
    cv2.polylines(strip, [pts2], False, (245, 215, 165), 3, cv2.LINE_AA)
    
    strip = cv2.GaussianBlur(strip, (0, 0), 5.5)
    
    full_currents = cv2.resize(strip, (WIDTH, strip_h * 2), interpolation=cv2.INTER_LINEAR)
    y0, y1 = 20, min(HORIZON - 30, 20 + full_currents.shape[0])
    ch = y1 - y0
    
    current_alpha = min(0.22, 0.15 * alpha)
    frame[y0:y1, :] += full_currents[:ch, :] * current_alpha


def shimmering_stars(frame, plates, t, night_amount, stillness=0.0):
    """Renders shimmering celestial stars with subtle breathing halos and ocean mirror reflections in a pristine, clear night sky."""
    if night_amount < 0.04:
        return
    star_spr = plates.get("star_sprite")
    if star_spr is None:
        return
    
    # Sky clarity: swirling current bands disabled to keep night sky clear and serene

    # Constellation coordinates across the night sky (x, y, base_size, frequency, phase)
    constellations = [
        (220, 75, 26, 1.8, 0.2),
        (460, 135, 34, 2.3, 1.4),
        (720, 60, 46, 1.5, 2.7),   # Zenith radiant star
        (960, 115, 30, 2.1, 0.9),
        (1210, 70, 40, 1.7, 3.1),
        (1450, 125, 28, 2.5, 1.8),
        (1680, 85, 32, 1.9, 0.5),
        (340, 185, 22, 2.7, 2.2),
        (830, 190, 24, 2.4, 1.1),
        (1330, 180, 26, 2.2, 3.5),
        (1580, 210, 22, 2.6, 0.8),
    ]
    
    halo_spr = _get_van_gogh_halo_sprite()
    music_e = mirage.music_energy(plates.get("audio_feats"), t)
    for sx, sy, base_sz, freq, phase in constellations:
        # Scintillation twinkle pulse
        shimmer = 0.5 + 0.5 * math.sin(t * freq + phase)
        shimmer_slow = 0.7 + 0.3 * math.cos(t * (freq * 0.38) + phase)
        sz = max(10, int(base_sz * (0.85 + 0.30 * shimmer)))
        alpha = night_amount * min(0.96, 0.45 + 0.50 * shimmer * shimmer_slow)
        
        # Subtle starlight halo keeping the sky clear and pristine, breathing with the music
        halo_sz = max(12, int(sz * 1.25))
        halo_pulse = 0.5 + 0.5 * math.sin(t * (freq * 0.8) + phase)
        halo_alpha = alpha * 0.16 * (0.70 + 0.30 * halo_pulse) * (0.85 + 0.30 * music_e)
        scaled_halo = cv2.resize(halo_spr, (halo_sz, halo_sz), interpolation=cv2.INTER_LINEAR)
        over(frame, scaled_halo, sx - halo_sz // 2, sy - halo_sz // 2, halo_alpha)

        # Core radiant star sprite
        scaled_star = cv2.resize(star_spr, (sz, sz), interpolation=cv2.INTER_AREA)
        over(frame, scaled_star, sx - sz // 2, sy - sz // 2, alpha)

        # Still mirror reflection on the ocean when Tanpura brings mind/water to complete stillness
        if stillness > 0.05:
            ref_y = HORIZON + int((HORIZON - sy) * 0.75)
            if ref_y < HEIGHT - 20:
                ref_sz = max(6, int(sz * 0.75))
                ref_alpha = alpha * 0.45 * float(stillness)
                ref_star = cv2.resize(scaled_star, (ref_sz, ref_sz), interpolation=cv2.INTER_AREA)
                over(frame, ref_star, sx - ref_sz // 2, ref_y - ref_sz // 2, ref_alpha)
                # Halo reflection on calm water
                ref_halo_sz = max(10, int(halo_sz * 0.75))
                ref_halo = cv2.resize(scaled_halo, (ref_halo_sz, ref_halo_sz), interpolation=cv2.INTER_LINEAR)
                over(frame, ref_halo, sx - ref_halo_sz // 2, ref_y - ref_halo_sz // 2, halo_alpha * 0.40 * float(stillness))


def _ship_for_near(plates, near):
    """Scale the saucer continuously with camera depth instead of size stepping."""
    source = plates.get("ship_src")
    if source is None:
        source = plates["ships"][0]
        return source
    near = min(1.0, max(0.0, float(near)))
    # Preserve the landed scale at near=0.70 while letting the craft recede
    # convincingly toward -Z instead of stopping at 62% of its original size.
    factor = 0.18 + 1.56 * near
    w = max(8, int(round(source.shape[1] * factor)))
    h = max(8, int(round(source.shape[0] * factor)))
    return cv2.resize(source, (w, h), interpolation=cv2.INTER_LINEAR)


def craft(frame, plates, body):
    if body is None:
        return
    x, y, near = body[:3]
    visibility = float(body[3]) if len(body) > 3 else 1.0
    sprite = _ship_for_near(plates, near)
    over(
        frame,
        sprite,
        x - sprite.shape[1] / 2,
        y - sprite.shape[0] / 2,
        visibility * min(1.0, 0.25 + near * 4.0) * _edge_presence(x, sprite.shape[1]),
    )


def craft_reflect(frame, plates, body, t, stillness=0.0):
    """Render a faint, wave-softened saucer reflection; the broad cue is in water.paint."""
    if body is None:
        return
    x, y, near = body[:3]
    visibility = float(body[3]) if len(body) > 3 else 1.0
    sprite = _ship_for_near(plates, near)
    if sprite is None:
        return
    sw, sh = sprite.shape[1], sprite.shape[0]
    
    # Height of ship above horizon determines reflection vertical position and scale
    alt_above_water = max(0.0, float(HORIZON - y))
    ref_y = HORIZON + 2 + int(alt_above_water * 0.35)
    wave_wobble = math.sin(t * 1.7 + x * 0.015) * 3.5
    ref_x = int(x - sw / 2 + wave_wobble)
    
    if HORIZON <= ref_y < HEIGHT - 10 and -sw < ref_x < WIDTH:
        max_h = min(sh, HEIGHT - ref_y)
        if max_h > 8:
            ref_sprite = sprite[:max_h][::-1].copy()
            # Celestial luminescent starlight tint
            ref_sprite[:, :, 0] = np.clip(ref_sprite[:, :, 0] * 0.78 + 20, 0, 255)
            ref_sprite[:, :, 1] = np.clip(ref_sprite[:, :, 1] * 0.82 + 22, 0, 255)
            ref_sprite[:, :, 2] = np.clip(ref_sprite[:, :, 2] * 0.90 + 30, 0, 255)
            ref_sprite = _water_wave_refraction(
                ref_sprite, t, x, max(0.0, 1.0 - float(stillness)), plates.get("audio_feats"), phase=1.4
            )
            alpha = min(0.16, (0.025 + 0.13 * near) * max(0.0, 1.0 - alt_above_water / 420.0))
            alpha *= visibility * _edge_presence(x, sw)
            if alpha > 0.02:
                over(frame, ref_sprite, ref_x, ref_y, alpha)
