"""Stars, snow, the craft, the mushroom. Each one is a paste, not a redraw."""
import math
import cv2
import numpy as np

from she_film.clock import HEIGHT, HORIZON, WIDTH, lerp, smooth, sky_weights


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
    (340, 24.0, 0.75),   # Leftmost Stand (starts wave, x=340)
    (640, 20.0, 1.15),   # Hero Center-Left Stand (x=640)
    (980, 28.0, 0.70),   # Center-Right Stand (x=980)
    (1260, 50.0, 1.05),  # East Stand (x=1260)
    (1540, 54.0, 0.65),  # Far East Stand (x=1540)
]


def _get_active_mushroom_stands(plates, t):
    """
    Returns list of (scaled_frame, x, y, grow_scale, alpha_fade) for all active stands at time t.
    Uses Flow AI-generated green-screen timelapse for gradual natural growth,
    and the Flow AI-generated subtle living breathing/swaying loop for mature stands.
    """
    vs_growth = plates.get("anim_mushroom_growth")
    vs_idle = plates.get("anim_mushroom_idle")
    base_y = HORIZON - 72

    stands = []
    if vs_growth is None or vs_idle is None:
        stages = plates.get("life", [])
        if stages:
            spr = stages[min(len(stages) - 1, int(t * 0.1) % len(stages))]
            stands.append((spr, 640 - spr.shape[1] // 2, base_y - spr.shape[0], 1.0, 1.0))
        return stands

    for x, t0, b_scale in MUSHROOM_STANDS:
        if t < t0:
            continue
        age = t - t0
        # Smooth scaling across the film to show growth throughout the video
        grow_scale = b_scale * (0.85 + 0.38 * min(1.0, max(0.0, (t - 25.0) / 75.0)))

        # Slower, gentle frequency with left-to-right passing wave effect across neighboring stands
        # Delay propagates from leftmost stand (x=340) to rightmost stand (x=1540)
        wave_delay = (x - 340.0) / 380.0
        
        if age < 8.0:
            raw = vs_growth.get_frame(age, loop=False)
            a_fade = min(1.0, age / 0.8)
        else:
            # Minimized frequency (0.45x) and neighbor wave phase delay
            idle_t = ((age - 8.0) * 0.45 - wave_delay) % 8.0
            raw = vs_idle.get_frame(idle_t, loop=True)
            a_fade = 1.0

        if raw is None:
            continue

        tw = max(16, int(raw.shape[1] * grow_scale))
        th = max(16, int(raw.shape[0] * grow_scale))
        scaled = cv2.resize(raw, (tw, th), interpolation=cv2.INTER_LINEAR)
        stands.append((scaled, x - tw // 2, base_y - th, grow_scale, a_fade, wave_delay))

    return stands


def life(frame, plates, t, alpha=1.0):
    if alpha < 0.02:
        return
    stands = _get_active_mushroom_stands(plates, t)
    for spr, x, y, scale, a_fade, wave_delay in stands:
        over(frame, spr, x, y, alpha * a_fade)


def life_reflect(frame, plates, t, stillness=0.0):
    wave_factor = max(0.0, 1.0 - float(stillness))
    stands = _get_active_mushroom_stands(plates, t)

    for spr, x, y, scale, a_fade, wave_delay in stands:
        sh, sw = spr.shape[:2]
        ref_h = int(sh * 0.80)
        if ref_h < 4:
            continue
        flipped = cv2.resize(spr[::-1], (sw, ref_h), interpolation=cv2.INTER_LINEAR)
        fade = np.linspace(0.45, 0.0, ref_h, dtype=np.float32)[:, None, None]
        flipped_rgba = flipped.copy()
        flipped_rgba[:, :, 3:4] *= fade

        # Ocean wave reflection wobble matching passing wave dynamics from left to right
        wobble = int(math.sin(t * 1.1 - (x - 340.0) * 0.006) * (6.0 * wave_factor))
        over(frame, flipped_rgba, x + wobble, HORIZON - 2, a_fade * 0.65)


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
    ptero = anim_ptero.get_frame(t) if anim_ptero is not None else plates.get("pokemon_pterodactyl")
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
    
    # Audio-reactive synchronization with the track's rhythmic piano roll (120 BPM cadence)
    audio_feats = plates.get("audio_feats")
    f_idx = min(len(audio_feats) - 1, max(0, int(t * 24.0))) if audio_feats is not None else 0
    piano_onset = float(audio_feats[f_idx, 9]) if audio_feats is not None else 0.0
    piano_energy = float(audio_feats[f_idx, 5]) if audio_feats is not None else 0.0
    
    # Musical gait cadence locked to piano downbeats
    t_piano = t + 0.05 * piano_onset

    for d_id, x in herd:
        ground = ridge_elevation_y(x)
        # 1. Hero Sauropod (d_id == 0) - Character-driven Grass-Sauropod
        if d_id == 0:
            anim_sauropod = plates.get("anim_sauropod")
            hero_spr = anim_sauropod.get_frame(t_piano) if anim_sauropod is not None else plates.get("pokemon_sauropod")
            neck_bend = max(0.0, math.sin((t - 258.0) / 30.0 * math.pi)) if 258.0 <= t <= 288.0 else 0.0
            
            if hero_spr is not None:
                hw, hh = hero_spr.shape[1], hero_spr.shape[0]
                # Ground contact shadow under feet deepens subtly on rhythmic piano accents
                shad_rx = int(115 * (1.0 + 0.12 * piano_onset))
                shad_ry = int(12 * (1.0 + 0.14 * piano_onset))
                cv2.ellipse(frame, (int(x), int(ground)), (shad_rx, shad_ry), 0, 0, 360, (22, 26, 20), -1, cv2.LINE_AA)
                # Dynamic long slanting shadow downward towards ocean water edge
                cv2.ellipse(frame, (int(x - 24), int(ground + 15)), (int(shad_rx * 1.25), int(shad_ry * 1.6)), -12, 0, 360, (18, 22, 16), -1, cv2.LINE_AA)
                sy = ground - hh + 6
                over(frame, hero_spr, x - hw / 2, sy, 0.98)
                
                # Falling foliage nibbles during canopy grazing sync with piano roll
                if 258.0 <= t <= 288.0 and neck_bend > 0.3:
                    for k in range(3):
                        ly = int(sy + 40 + ((t * 22 + k * 7 + piano_onset * 5) % 25))
                        lx = int(x + hw * 0.28 + math.sin(t * 7 + k) * 6)
                        if 0 <= lx < WIDTH and 0 <= ly < HEIGHT:
                            cv2.circle(frame, (lx, ly), 2, (75, 138, 55), -1)
            else:
                sprite_y = ground - 280 + 4
                cv2.ellipse(frame, (int(x + 10), int(ground)), (140, 14), 0, 0, 360, (24, 28, 22), -1, cv2.LINE_AA)
                _draw_grazing_sauropod(frame, x - 250, sprite_y, neck_bend, t)
        else:
            # Trailing species: Hadrosaur (1), Ankylosaur (2), Raptor (3)
            anim_hadro = plates.get("anim_hadrosaur")
            anim_anky = plates.get("anim_ankylosaur")
            anim_raptor = plates.get("anim_raptor")
            if d_id == 1 and anim_hadro is not None:
                sprite = anim_hadro.get_frame(t_piano)
            elif d_id == 2 and anim_anky is not None:
                sprite = anim_anky.get_frame(t_piano)
            elif d_id == 3 and anim_raptor is not None:
                sprite = anim_raptor.get_frame(t_piano)
            else:
                sprite = forms[d_id % len(forms)]
            sw, sh = sprite.shape[1], sprite.shape[0]
            sy = ground - sh + 5
            # Foot contact shadow permanently sized to this exact species
            shad_r = int(sw * 0.32 * (1.0 + 0.1 * piano_onset))
            cv2.ellipse(frame, (int(x), int(ground)), (shad_r, 8), 0, 0, 360, (24, 28, 22), -1, cv2.LINE_AA)
            cv2.ellipse(frame, (int(x - 14), int(ground + 10)), (int(shad_r * 1.15), 11), -10, 0, 360, (18, 22, 16), -1, cv2.LINE_AA)
            over(frame, sprite, x - sw / 2, sy, 0.96)


def beasts_reflect(frame, plates, herd, t=0.0, stillness=0.0):
    """Renders wave-rippled watery reflections of the dinosaur caravan on the ocean below."""
    if not herd:
        return
    audio_feats = plates.get("audio_feats")
    f_idx = min(len(audio_feats) - 1, max(0, int(t * 24.0))) if audio_feats is not None else 0
    piano_onset = float(audio_feats[f_idx, 9]) if audio_feats is not None else 0.0
    t_piano = t + 0.05 * piano_onset
    forms = plates["beasts"]
    
    # Stillness factor: when Tanpura swells (stillness -> 1.0), wobble attenuates to 0.0
    wave_factor = max(0.0, 1.0 - float(stillness))

    for d_id, x in herd:
        ground = ridge_elevation_y(x)
        if d_id == 0:
            anim_sauropod = plates.get("anim_sauropod")
            spr = anim_sauropod.get_frame(t_piano) if anim_sauropod is not None else plates.get("pokemon_sauropod")
        elif d_id == 1:
            anim_hadro = plates.get("anim_hadrosaur")
            spr = anim_hadro.get_frame(t_piano) if anim_hadro is not None else forms[1 % len(forms)]
        elif d_id == 2:
            anim_anky = plates.get("anim_ankylosaur")
            spr = anim_anky.get_frame(t_piano) if anim_anky is not None else forms[2 % len(forms)]
        elif d_id == 3:
            anim_raptor = plates.get("anim_raptor")
            spr = anim_raptor.get_frame(t_piano) if anim_raptor is not None else forms[3 % len(forms)]
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
                # Nighttime moonlit reflection boost: clearly visible on the dark ocean
                night_factor = min(1.0, max(0.0, (t - 280.0) / 20.0))
                
                # Heightened opacity at night so reflection stands out boldly against dark sea
                top_fade = lerp(0.50, 0.88, night_factor) + 0.10 * float(stillness)
                bot_fade = lerp(0.02, 0.22, night_factor)
                fade = np.linspace(top_fade, bot_fade, max_ref_h, dtype=np.float32)[:, None, None]
                
                ref_sprite = spr[:max_ref_h][::-1].copy()
                ref_sprite[:, :, 3:4] *= fade * (0.80 + 0.20 * piano_onset)
                
                # In daylight: warm amber-teal tint; At night: luminous moonlit silvery-blue radiance
                # Guaranteeing dinosaurs' reflections are vividly visible on the ocean
                r_add = lerp(16.0, 78.0, night_factor)
                g_add = lerp(22.0, 118.0, night_factor)
                b_add = lerp(30.0, 168.0, night_factor)
                
                ref_sprite[:, :, 0] = np.clip(ref_sprite[:, :, 0] * 0.70 + r_add, 0, 255)
                ref_sprite[:, :, 1] = np.clip(ref_sprite[:, :, 1] * 0.78 + g_add, 0, 255)
                ref_sprite[:, :, 2] = np.clip(ref_sprite[:, :, 2] * 0.85 + b_add, 0, 255)
                
                # Subtle water crest lines shimmering through the reflection
                wave_shimmer = 1.0 + 0.18 * np.sin(np.arange(max_ref_h)[:, None, None] * 0.16 + t * 3.5) * wave_factor
                ref_sprite[:, :, 3:4] = np.clip(ref_sprite[:, :, 3:4] * wave_shimmer, 0, 255)
                
                eff_over_alpha = lerp(0.60, 0.88, night_factor)
                over(frame, ref_sprite, ref_x, ref_y, eff_over_alpha)


def shimmering_stars(frame, plates, t, night_amount, stillness=0.0):
    """Renders shimmering celestial stars with optical diffraction spikes at night and mirror reflections."""
    if night_amount < 0.04:
        return
    star_spr = plates.get("star_sprite")
    if star_spr is None:
        return
    
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
    
    for sx, sy, base_sz, freq, phase in constellations:
        # Scintillation twinkle pulse
        shimmer = 0.5 + 0.5 * math.sin(t * freq + phase)
        shimmer_slow = 0.7 + 0.3 * math.cos(t * (freq * 0.38) + phase)
        sz = max(10, int(base_sz * (0.85 + 0.30 * shimmer)))
        alpha = night_amount * min(0.96, 0.45 + 0.50 * shimmer * shimmer_slow)
        
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


def craft(frame, plates, body):
    if body is None:
        return
    x, y, near = body
    sizes = plates["ships"]
    idx = min(len(sizes) - 1, int(near * (len(sizes) - 1) + 0.001))
    sprite = sizes[idx]
    over(
        frame,
        sprite,
        x - sprite.shape[1] / 2,
        y - sprite.shape[0] / 2,
        min(1.0, 0.25 + near * 4.0),
    )


def craft_reflect(frame, plates, body, t):
    """Renders the luminescent saucer / starcraft reflection and light column on the ocean."""
    if body is None:
        return
    x, y, near = body
    sizes = plates["ships"]
    idx = min(len(sizes) - 1, int(near * (len(sizes) - 1) + 0.001))
    sprite = sizes[idx]
    if sprite is None:
        return
    sw, sh = sprite.shape[1], sprite.shape[0]
    
    # Height of ship above horizon determines reflection vertical position and scale
    alt_above_water = max(0.0, float(HORIZON - y))
    ref_y = HORIZON + 2 + int(alt_above_water * 0.35)
    wave_wobble = math.sin(t * 3.5 + x * 0.03) * 6.0
    ref_x = int(x - sw / 2 + wave_wobble)
    
    if HORIZON <= ref_y < HEIGHT - 10 and -sw < ref_x < WIDTH:
        max_h = min(sh, HEIGHT - ref_y)
        if max_h > 8:
            ref_sprite = sprite[:max_h][::-1].copy()
            # Celestial luminescent starlight tint
            ref_sprite[:, :, 0] = np.clip(ref_sprite[:, :, 0] * 1.15 + 35, 0, 255)
            ref_sprite[:, :, 1] = np.clip(ref_sprite[:, :, 1] * 0.95 + 20, 0, 255)
            ref_sprite[:, :, 2] = np.clip(ref_sprite[:, :, 2] * 0.70, 0, 255)
            alpha = min(0.75, (0.25 + 0.55 * near) * max(0.0, 1.0 - alt_above_water / 500.0))
            if alpha > 0.02:
                over(frame, ref_sprite, ref_x, ref_y, alpha)
                
                # Ethereal vertical beam of light connecting saucer to ocean surface
                if y < HORIZON:
                    beam_w = max(24, int(sw * 0.45))
                    bx0 = max(0, int(x - beam_w // 2))
                    bx1 = min(WIDTH, int(x + beam_w // 2))
                    by0 = HORIZON
                    by1 = min(HEIGHT, HORIZON + 90)
                    beam_alpha = float(alpha * 0.32)
                    frame[by0:by1, bx0:bx1, 0] += 130.0 * beam_alpha
                    frame[by0:by1, bx0:bx1, 1] += 145.0 * beam_alpha
                    frame[by0:by1, bx0:bx1, 2] += 95.0 * beam_alpha
