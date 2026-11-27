"""One frame from the plates and the clock. No plate is rebuilt here."""
import math
import cv2
import numpy as np
from PIL import Image

from she_film import actors, grade, mirage, outro, water
from she_film.mirage import refract_background
from she_film.clock import (
    HEIGHT,
    HORIZON,
    DURATION,
    FPS,
    OUTRO_FADE_START,
    OUTRO_TRANSITION_START,
    WIDTH,
    birds,
    arrival_cue,
    buddha,
    colony,
    forest_cycle,
    mushroom_cycle_alpha,
    crimson,
    dream,
    herd,
    ice,
    moon,
    scene_light,
    sky_weights,
    smooth,
    snow_amount,
    sun,
    star_visibility,
    title_lines,
    visit,
    waking,
    water_stillness,
)

_SUN_CACHE = {}
_CRT_MAP_CACHE = {}

# Inner phosphor bounds measured from the exact buddha_retro_studio_sleep.jpg
# plate. Quads are ordered top-left, top-right, bottom-right, bottom-left.
STUDIO_SCREEN_QUADS = {
    "scr_left": ((72.0, 302.0), (314.0, 363.0), (314.0, 631.0), (72.0, 682.0)),
    "scr_mid_left": ((344.0, 374.0), (484.0, 414.0), (484.0, 641.0), (344.0, 663.0)),
    "scr_mid_right": ((496.0, 416.0), (570.0, 442.0), (570.0, 614.0), (496.0, 637.0)),
    "scr_right": ((1544.0, 386.0), (1786.0, 311.0), (1792.0, 666.0), (1542.0, 647.0)),
}
STUDIO_BEZEL_QUADS = {
    "scr_left": ((52.0, 278.0), (330.0, 350.0), (330.0, 652.0), (52.0, 712.0)),
    "scr_mid_left": ((332.0, 360.0), (494.0, 405.0), (494.0, 649.0), (332.0, 681.0)),
    "scr_mid_right": ((488.0, 404.0), (580.0, 434.0), (580.0, 622.0), (488.0, 649.0)),
    "scr_right": ((1526.0, 371.0), (1807.0, 291.0), (1812.0, 696.0), (1526.0, 702.0)),
}


def validate_studio_screen_bounds():
    """Check that each video quad is contained by its source-plate bezel."""
    for key, screen in STUDIO_SCREEN_QUADS.items():
        bezel = STUDIO_BEZEL_QUADS[key]
        bezel_poly = np.asarray(bezel, dtype=np.float32)
        if len(screen) != 4 or any(
            cv2.pointPolygonTest(bezel_poly, (float(x), float(y)), False) < 0
            for x, y in screen
        ):
            return False, f"{key} video quad escapes the TV bezel bounds"
        if len(set(screen)) != 4:
            return False, f"{key} video quad has duplicate corners"
    return True, "both video quads are inside their measured bezel bounds"


def studio_screen_debug_overlay(studio):
    """Return a still with red bezel edges and cyan active video edges."""
    from PIL import Image, ImageDraw
    image = Image.fromarray(np.clip(studio, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    for key, points in STUDIO_BEZEL_QUADS.items():
        draw.line(list(points) + [points[0]], fill=(255, 70, 70), width=4)
        screen = STUDIO_SCREEN_QUADS[key]
        draw.line(list(screen) + [screen[0]], fill=(40, 255, 210), width=3)
        for index, (x, y) in enumerate(screen):
            r = 7
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(30, 110, 255), outline=(255, 255, 255), width=2)
            draw.text((x + 8, y + 4), f"{key[-1]}{index+1}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(image)


def _crt_bulge(image, strength=0.055):
    """Apply a subtle convex-glass barrel warp to the monitor's picture area."""
    h, w = image.shape[:2]
    key = (w, h, float(strength))
    maps = _CRT_MAP_CACHE.get(key)
    if maps is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx - (w - 1) * 0.5) / max(1.0, (w - 1) * 0.5)
        ny = (yy - (h - 1) * 0.5) / max(1.0, (h - 1) * 0.5)
        r2 = nx * nx + ny * ny
        # Destination-to-source mapping: content bends gently around the CRT face.
        factor = 1.0 + strength * r2
        map_x = ((w - 1) * 0.5 + nx * factor * (w - 1) * 0.5).astype(np.float32)
        map_y = ((h - 1) * 0.5 + ny * factor * (h - 1) * 0.5).astype(np.float32)
        maps = (map_x, map_y)
        _CRT_MAP_CACHE[key] = maps
    return cv2.remap(image, maps[0], maps[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def _scaled_sun(sprite, radius, squash):
    key = (int(radius), int(squash * 24))
    hit = _SUN_CACHE.get(key)
    if hit is None:
        im = Image.fromarray(sprite)
        w = max(8, int(im.width * (radius / 280.0)))
        h = max(8, int(w * squash))
        hit = np.asarray(im.resize((w, h), Image.Resampling.BILINEAR)).astype(np.float32)
        hit[:, :, 3] *= 1.0 / 255.0
        # Soft atmospheric sun edge blur and optical corona glow
        hit[:, :, 3] = cv2.GaussianBlur(hit[:, :, 3], (0, 0), 5.5)
        glow = cv2.GaussianBlur(hit[:, :, 3], (0, 0), 12.0)
        hit[:, :, 3] = np.clip(hit[:, :, 3] * 0.85 + glow * 0.25, 0.0, 1.0)
        _SUN_CACHE[key] = hit
    return hit


def _separated(craft, moon_body):
    if craft is None:
        return False
    if moon_body is None:
        return True
    mx, alt = moon_body
    return math.hypot(craft[0] - mx, craft[1] - (HORIZON - alt)) > 220.0


def _logo_frame(plates, t):
    frame = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    frame[:, :] = (22, 18, 16)
    yy = np.linspace(-1, 1, HEIGHT, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, WIDTH, dtype=np.float32)[None, :]
    glow = np.clip(1.0 - np.sqrt(xx ** 2 + yy ** 2), 0, 1) ** 2
    frame += glow[:, :, None] * np.array([90, 50, 16], np.float32)
    logo = plates["logo"]
    alpha = 1.0
    if t < 2.0:
        alpha = 1.0
        frame += (1.0 - t / 2.0) * 180.0
    elif t >= 4.6:
        alpha = 1.0 - smooth((t - 4.6) / 3.4)
    actors.over(frame, logo, (WIDTH - logo.shape[1]) / 2, (HEIGHT - logo.shape[0]) / 2, alpha)
    if t >= 4.6:
        world = render(plates, 8.2)
        p = smooth((t - 4.6) / 3.4)
        frame = frame * (1.0 - p) + world.astype(np.float32) * p
    return np.clip(frame, 0, 255).astype(np.uint8)


def _render_buddha_waveform(frame, t, buddha_alpha, plates):
    """
    Audio-reactive oscilloscope waveform with prominent crests and troughs over Buddha's heart region (y=600),
    and slowly expanding, highly saturated radiant white mystical love heart glow during fadeout.
    """
    feats = plates.get("audio_feats")
    frame_idx = min(int(DURATION * FPS) - 1, max(0, int(t * FPS)))
    
    if feats is not None:
        idx0 = max(0, frame_idx - 3)
        idx1 = min(len(feats), frame_idx + 4)
        sub_bass = float(np.mean(feats[idx0:idx1, 2]))
        onset = float(feats[frame_idx, 9]) if frame_idx < len(feats) else 0.5
    else:
        sub_bass = 0.95
        onset = 0.5
    
    # Buddha's heart center is located at (heart_x=960, center_y=600)
    center_y = 600
    heart_x = 960
    xs = np.arange(0, WIDTH, 3, dtype=np.int32)
    
    # Gaussian chest envelope centered at x=960 across the torso
    chest_env = np.exp(-((xs - heart_x) / 480.0) ** 2).astype(np.float32)
    
    # Distinct, visible waveform crests and troughs (musical harmonics, peaks and valleys)
    # Primary musical wave + harmonic overtone + fast transient wave detail
    w_sub = np.sin(xs * 0.048 + t * 4.2) * (42.0 * sub_bass)
    w_harm = np.sin(xs * 0.115 - t * 7.8) * (22.0 * (0.8 + 0.4 * onset))
    w_fine = np.sin(xs * 0.220 + t * 12.5) * (12.0 * sub_bass)
    total_wave = (w_sub + w_harm + w_fine) * chest_env
    
    ys_primary = (center_y + total_wave).astype(np.int32)
    ys_secondary = (center_y - total_wave * 0.65 + np.sin(xs * 0.08 + t * 5.5) * 10.0 * chest_env).astype(np.int32)
    
    pts_prim = np.column_stack((xs, ys_primary)).reshape((-1, 1, 2))
    pts_sec = np.column_stack((xs, ys_secondary)).reshape((-1, 1, 2))
    
    glow = np.zeros_like(frame)
    # Multi-layered glowing oscilloscope traces
    cv2.polylines(glow, [pts_prim], False, (50, 240, 220), 5, cv2.LINE_AA)
    cv2.polylines(glow, [pts_prim], False, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(glow, [pts_sec], False, (255, 120, 210), 3, cv2.LINE_AA)
    cv2.polylines(glow, [pts_sec], False, (255, 225, 140), 1, cv2.LINE_AA)
    
    alpha = float(buddha_alpha * 0.85)
    frame += glow * alpha

    # Mystical Love Saturated White Heart Glow: slowly animated expansion (410s <= t <= 428s)
    if t >= 410.0:
        glow_fade = min(1.0, max(0.0, (t - 410.0) / 12.0))
        # Animated slow expansion curve from t=410s to 428s
        expand_p = smooth((t - 410.0) / 17.0)
        # Gentle 43 BPM meditative heartbeat pulse
        pulse = 1.0 + 0.14 * math.sin((t - 410.0) * 4.3)
        
        # Expanding radius: core blooms from 45px to 125px; outer aura blooms from 140px to 340px
        core_r = (45.0 + 80.0 * expand_p) * pulse
        aura_r = (140.0 + 200.0 * expand_p) * pulse
        
        yy, xx = np.ogrid[max(0, center_y - 300):min(HEIGHT, center_y + 300),
                          max(0, heart_x - 380):min(WIDTH, heart_x + 380)]
        dist_sq = ((xx - heart_x) / 1.3) ** 2 + (yy - center_y) ** 2
        
        core_val = np.exp(-dist_sq / (2.0 * core_r ** 2))
        aura_val = np.exp(-dist_sq / (2.0 * aura_r ** 2))
        
        # Subtle divine rays rotating slowly
        angle = np.arctan2(yy - center_y, xx - heart_x)
        rays = 0.5 + 0.5 * np.cos(8.0 * angle + t * 0.35)
        
        # Highly saturated pure luminous white emission at heart core
        white_intensity = glow_fade * (core_val * 255.0 + aura_val * 110.0 + rays * aura_val * 40.0 * expand_p)
        gold_intensity = glow_fade * (aura_val * 50.0)
        
        y0, y1 = max(0, center_y - 300), min(HEIGHT, center_y + 300)
        x0, x1 = max(0, heart_x - 380), min(WIDTH, heart_x + 380)
        
        # Saturated white light: maximum equal RGB boosting with golden warmth at outer fringe
        frame[y0:y1, x0:x1, 0] = np.clip(frame[y0:y1, x0:x1, 0] + white_intensity + gold_intensity * 0.5, 0, 255)
        frame[y0:y1, x0:x1, 1] = np.clip(frame[y0:y1, x0:x1, 1] + white_intensity + gold_intensity * 0.3, 0, 255)
        frame[y0:y1, x0:x1, 2] = np.clip(frame[y0:y1, x0:x1, 2] + white_intensity, 0, 255)


def _render_studio_screens(base_studio, t, plates=None):
    """
    Renders the four-screen Buddha telemetry suite
    using true perspective planar homography (cv2.warpPerspective) onto calibrated
    perspective trapezoids that respect Z-axis depth vanishing geometry:
      - Large left: Buddha meditation B-roll
      - Two inner left screens: dinosaur and channel-surfing B-roll
      - Right: Buddha portrait and mudra detail B-roll
    """
    studio = base_studio.copy()
    screens_data = plates.get("screens_90s") if plates is not None else None
    h_bg, w_bg = studio.shape[:2]

    def apply_perspective_screen(quad, frames_key, default_rgb):
        q = np.array(quad, dtype=np.float32)
        w_top = np.linalg.norm(q[1] - q[0])
        w_bot = np.linalg.norm(q[2] - q[3])
        h_l = np.linalg.norm(q[3] - q[0])
        h_r = np.linalg.norm(q[2] - q[1])
        sw = max(16, int(max(w_top, w_bot) * 2))
        sh = max(16, int(max(h_l, h_r) * 2))

        if screens_data is not None and frames_key in screens_data:
            frames = screens_data[frames_key]
            # Screen programs begin with the studio scene, not at song time zero.
            f_idx = int(max(0.0, t - 420.0) * 24.0) % len(frames)
            # Cover the measured screen plane while preserving the portrait
            # feed aspect ratio; crop excess edges instead of stretching.
            src = frames[f_idx]
            src_h, src_w = src.shape[:2]
            target_aspect = sw / float(sh)
            src_aspect = src_w / float(src_h)
            if src_aspect > target_aspect:
                crop_w = max(1, int(src_h * target_aspect))
                x0 = (src_w - crop_w) // 2
                src = src[:, x0:x0 + crop_w]
            else:
                crop_h = max(1, int(src_w / target_aspect))
                y0 = (src_h - crop_h) // 2
                src = src[y0:y0 + crop_h, :]
            scr = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        else:
            scr = np.full((sh, sw, 3), default_rgb, dtype=np.float32)

        # Curved CRT glass is simulated in the screen's own coordinates before
        # the perspective projection maps its rectangular bezel onto the wall.
        scr = _crt_bulge(scr)

        # Authentic 1990s horizontal scanlines
        scr[::2, :] *= 0.88
        # CRT glass falloff and phosphor lift.
        yy, xx = np.ogrid[:sh, :sw]
        nx = (xx - sw * 0.5) / (sw * 0.5)
        ny = (yy - sh * 0.5) / (sh * 0.5)
        vig = np.clip(1.0 - 0.16 * (nx ** 2 + ny ** 2), 0.76, 1.0)[:, :, None]
        scr = np.clip(scr * vig * 1.05 + 4.0, 0, 255)

        # Planar Homography Warping
        src_pts = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_pts, q)
        warped = cv2.warpPerspective(scr, H, (w_bg, h_bg), flags=cv2.INTER_LANCZOS4)

        # High-precision anti-aliased feathered mask
        x0 = max(0, int(np.min(q[:, 0]) - 8))
        y0 = max(0, int(np.min(q[:, 1]) - 8))
        x1 = min(w_bg, int(np.max(q[:, 0]) + 8))
        y1 = min(h_bg, int(np.max(q[:, 1]) + 8))

        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        local_poly = (q - np.array([x0, y0])).astype(np.int32)
        cv2.fillConvexPoly(mask, local_poly, 1.0, lineType=cv2.LINE_AA)
        # fillConvexPoly already antialiases the boundary. Extra blur expands
        # coverage beyond the active area and bleeds into the bezel.
        mask = mask[:, :, None]

        roi_bg = studio[y0:y1, x0:x1]
        roi_warp = warped[y0:y1, x0:x1]
        # Keep the feed inside its active-area mask; the bezel remains visible.
        studio[y0:y1, x0:x1] = roi_bg * (1.0 - mask) + roi_warp * mask

    # Large left screen: Buddha meditation B-roll.
    apply_perspective_screen(STUDIO_SCREEN_QUADS["scr_left"], "scr_left", [50, 70, 80])

    # Inner monitors show the dinosaur plate and the playful channel-surfing reel.
    apply_perspective_screen(STUDIO_SCREEN_QUADS["scr_mid_left"], "scr_mid_left", [50, 70, 80])
    apply_perspective_screen(STUDIO_SCREEN_QUADS["scr_mid_right"], "scr_mid_right", [50, 70, 80])

    # Right screen: Buddha portrait and mudra detail B-roll.
    apply_perspective_screen(STUDIO_SCREEN_QUADS["scr_right"], "scr_right", [45, 65, 30])

    return studio


def render(plates, t):
    if t < 8.0:
        return _logo_frame(plates, t)

    day, warm, night = sky_weights(t)
    sun_body = sun(t)
    moon_body = moon(t)
    red = crimson(t)
    if sun_body and moon_body and sun_body[1] > 0 and moon_body[1] > 0:
        raise RuntimeError(f"two lamps at t={t}")

    frame = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    frame[:HORIZON] = plates["day"] * day + plates["warm"] * warm + plates["night"] * night
    buddha_alpha = buddha(t)
    if buddha_alpha > 0.01:
        actors.over(frame, plates["nebula"], 0, 0, buddha_alpha * 0.85)
    if red > 0.01:
        frame[:HORIZON, :, 0] += 36.0 * red
        frame[:HORIZON, :, 1] *= 1.0 - 0.22 * red
        frame[:HORIZON, :, 2] *= 1.0 - 0.35 * red
    star_amount = star_visibility(t) * (1.0 - 0.25 * red)
    if star_amount > 0.005:
        actors.stars(frame, plates, t, star_amount, moon_body)
        actors.shimmering_stars(frame, plates, t, star_amount, stillness=water_stillness(t))

    if moon_body is not None:
        sprite = plates["moon_red"] if red > 0.4 else plates["moon"]
        fade = smooth((moon_body[1] + 12.0) / 24.0)
        actors.over(
            frame,
            sprite,
            moon_body[0] - sprite.shape[1] / 2,
            HORIZON - moon_body[1] - sprite.shape[0] / 2,
            fade,
        )

    if sun_body is not None:
        alt = sun_body[1]
        radius = 300.0 - 180.0 * smooth(alt / 340.0)
        squash = 1.0 - 0.4 * smooth((52.0 - alt) / 52.0)
        sprite = _scaled_sun(plates["sun_src"], radius, squash)
        fade = smooth((alt + 24.0) / 30.0)
        actors.over(
            frame,
            sprite,
            sun_body[0] - sprite.shape[1] / 2,
            HORIZON - alt - sprite.shape[0] / 2,
            fade,
        )

    ridge_colors = (
        np.array([186 * warm + 150 * day + 42 * night, 96 * warm + 142 * day + 40 * night, 64 * warm + 160 * day + 68 * night], np.float32),
        np.array([128 * warm + 102 * day + 30 * night, 72 * warm + 112 * day + 30 * night, 64 * warm + 130 * day + 52 * night], np.float32),
        np.array([64 * warm + 54 * day + 20 * night, 58 * warm + 78 * day + 22 * night, 62 * warm + 90 * day + 36 * night], np.float32),
        np.array([34 * warm + 30 * day + 12 * night, 44 * warm + 52 * day + 16 * night, 50 * warm + 62 * day + 26 * night], np.float32),
    )
    y0 = 280
    band = frame[y0:HORIZON]
    yy_grid = np.arange(y0, HORIZON, dtype=np.float32)[:, None]
    xx_grid = np.arange(WIDTH, dtype=np.float32)[None, :]
    ridge_layers = plates.get("ridge_surface_layers")

    # Render sculpted 3D beach sand dunes with sunlit crest highlights and wind-rippled slopes
    for k, (mask, base_color) in enumerate(zip(plates["ridges"], ridge_colors)):
        a = mask[y0:HORIZON, :, None]
        if a.max() < 0.005:
            continue

        if ridge_layers is not None and k < len(ridge_layers):
            surf_y = ridge_layers[k][None, :].astype(np.float32)
        else:
            surf_y = (HORIZON - 40) * np.ones((1, WIDTH), dtype=np.float32)

        depth_in_dune = np.maximum(0.0, yy_grid - surf_y)
        
        # 1. Warm sunlit sand crest highlight along the dune rim
        crest_highlight = np.exp(-0.5 * (depth_in_dune / 9.0) ** 2)
        crest_light = np.array([
            55.0 * warm + 32.0 * day + 10.0 * night,
            34.0 * warm + 22.0 * day + 14.0 * night,
            14.0 * warm + 12.0 * day + 24.0 * night
        ], dtype=np.float32)

        # 2. Smooth dune slope shading: graceful gradient from sunlit crest into deeper hollows
        slope_depth = np.clip(depth_in_dune / 80.0, 0.0, 1.0)
        hollow_shade = np.array([
            -22.0 * warm - 16.0 * day - 8.0 * night,
            -16.0 * warm - 12.0 * day - 6.0 * night,
            -10.0 * warm - 6.0 * day + 4.0 * night
        ], dtype=np.float32)

        # 3. Sweeping coastal wind-sculpted sand ripples along the beach dunes
        ripple_angle = xx_grid * 0.032 + yy_grid * 0.065 + k * 1.8
        ripple_mod = np.sin(xx_grid * 0.008 + 1.2) * 1.4
        sand_ripples = np.sin(ripple_angle + ripple_mod) * (0.055 * (1.0 - night * 0.45))

        # Composite the 3D sculpted sand dune layer
        dune_rgb = base_color + crest_light * crest_highlight[:, :, None] + hollow_shade * slope_depth[:, :, None]
        dune_rgb += base_color * sand_ripples[:, :, None]

        band += (dune_rgb - band) * a

    cover = ice(t)
    if cover > 0.01:
        land = plates["land"][y0:HORIZON, :, None]
        white = np.array([228, 234, 240], np.float32)
        band += (white - band) * (cover * 0.8) * land

    # A small, timed horizon refraction bridges vegetation states. It is
    # deliberately applied before sprites so the mushrooms/firs retain their
    # authored contours and readable material detail.
    refract_background(frame, t)
    mirage.soften_background(frame, t, plates.get("audio_feats"))

    alpha, stage, copies = colony(t)
    mushroom_alpha = mushroom_cycle_alpha(t)
    forest_alpha, forest_decay = forest_cycle(t)
    if mushroom_alpha > 0.02:
        actors.life(frame, plates, t, mushroom_alpha)
    if forest_alpha > 0.02:
        # Inland stands live behind the caravan and the nearer ridge shelves.
        actors.forest(frame, plates, t, forest_alpha, forest_decay, depth_layers=(0, 1))
    actors.beasts(frame, plates, herd(t), t)
    arrived = visit(t)

    # Composite the nearest shore in front of distant actors. The tree masks
    # additionally account for every nearer ridge, rather than one hard cutoff.
    y1 = HORIZON - 70
    front = plates["ridges"][-1][y1:HORIZON, :, None]
    if front.max() > 0.005:
        surf_front = ridge_layers[-1][None, :].astype(np.float32) if ridge_layers is not None else (HORIZON - 30)
        depth_front = np.maximum(0.0, np.arange(y1, HORIZON, dtype=np.float32)[:, None] - surf_front)
        crest_f = np.exp(-0.5 * (depth_front / 8.0) ** 2)
        crest_light_f = np.array([
            50.0 * warm + 30.0 * day + 8.0 * night,
            32.0 * warm + 20.0 * day + 12.0 * night,
            12.0 * warm + 10.0 * day + 20.0 * night
        ], dtype=np.float32)
        front_rgb = ridge_colors[-1] + crest_light_f * crest_f[:, :, None]
        frame[y1:HORIZON] += (front_rgb - frame[y1:HORIZON]) * front
    if cover > 0.01:
        land = plates["land"][y1:HORIZON, :, None]
        white = np.array([228, 234, 240], np.float32)
        frame[y1:HORIZON] += (white - frame[y1:HORIZON]) * (cover * 0.8) * land

    # A landed/near craft sits in front of the inland tree cutouts. As its
    # projected depth recedes, restore the foreground trees over it so the
    # takeoff travels naturally behind the nearer landscape.
    craft_is_near = arrived is not None and arrived[2] >= 0.42
    if forest_alpha > 0.02 and craft_is_near:
        actors.forest(frame, plates, t, forest_alpha, forest_decay, depth_layers=(2, 3))
    if arrived is not None:
        actors.craft(frame, plates, arrived)
    if forest_alpha > 0.02 and not craft_is_near:
        actors.forest(frame, plates, t, forest_alpha, forest_decay, depth_layers=(2, 3))

    base = (
        plates["water_warm"] * warm
        + plates["water_day"] * day
        + plates["water_night"] * night
    )
    lamps = []
    if sun_body is not None:
        lamps.append((sun_body[0], smooth((sun_body[1] + 16.0) / 30.0), (252, 186, 80), 64, 420))
    elif moon_body is not None:
        rgb = (210, 72, 64) if red > 0.35 else (198, 210, 224)
        lamps.append((moon_body[0], 0.42 * smooth((moon_body[1] + 10.0) / 24.0), rgb, 16, 110))
    cue = arrival_cue(t)
    if cue is not None:
        cue_x, cue_weight = cue
        lamps.append((cue_x, cue_weight, (178, 207, 230), 18, 110, 0.012))
    if arrived is not None:
        cx, cy, near = arrived[:3]
        visibility = float(arrived[3]) if len(arrived) > 3 else 1.0
        music_e = mirage.music_energy(plates.get("audio_feats"), t)
        # Flying saucer celestial glitter column across the ocean waves (matching the moon style)
        saucer_alt = max(0.0, float(HORIZON - cy))
        saucer_wt = min(0.11, (0.025 + 0.10 * near) * max(0.0, 1.0 - saucer_alt / 480.0)) * visibility * (0.90 + 0.25 * music_e)
        if saucer_wt > 0.04:
            # A restrained crystal streak follows the craft without a hard beam.
            lamps.append((cx, saucer_wt, (188, 211, 232), 16, 104,
                          0.018 * near * max(0.0, 1.0 - saucer_alt / 420.0)))
    still = max(cover, water_stillness(t))
    frame[HORIZON:] = water.paint(base, t, lamps, stillness=still, audio_feats=plates.get("audio_feats"))
    if star_amount > 0.04:
        actors.aurora_reflect(
            frame,
            t,
            star_amount,
            stillness=still,
            audio_feats=plates.get("audio_feats"),
            moon_body=moon_body,
        )
    if mushroom_alpha > 0.02:
        actors.life_reflect(frame, plates, t, stillness=still)
    if forest_alpha > 0.02:
        actors.forest_reflect(frame, plates, t, forest_alpha, forest_decay, stillness=still)
    actors.beasts_reflect(frame, plates, herd(t), t, stillness=still)
    if arrived is not None:
        actors.craft_reflect(frame, plates, arrived, t, stillness=still)

    # The cosmic demigod Buddha reveal (Maya & Lila / The Projection Mechanism)
    if buddha_alpha > 0.01:
        frame = frame * (1.0 - buddha_alpha) + plates["cosmic_demigod"] * buddha_alpha
        _render_buddha_waveform(frame, t, buddha_alpha, plates)

    for idx_t, ((rule_a, text_a), (words, rule, x, y)) in enumerate(zip(title_lines(t), plates["titles"])):
        # Subtle 5px upward drift during each layer's fade-in (meditative settle)
        drift_y = 5.0 * (1.0 - min(1.0, text_a)) if (8.0 <= t <= 14.5 and text_a > 0.02) else 0.0
        if rule is not None and rule_a > 0.02:
            if idx_t == 0:
                ry = 116.0 + drift_y
                draw_prog = smooth((t - 9.5) / 1.1)
            else:
                # Animated underline beneath SHE expanding smoothly from center
                ry = 232.0 + drift_y
                draw_prog = smooth((t - 11.2) / 1.3)
            rw = rule.shape[1]
            vis_w = max(2, int(round(rw * draw_prog)))
            c0 = (rw - vis_w) // 2
            sub_rule = rule[:, c0:c0 + vis_w]
            rx = x + (words.shape[1] - rw) / 2 + c0
            actors.over(frame, sub_rule, rx, ry, rule_a)
        if text_a > 0.02:
            actors.over(frame, words, x, y + drift_y, text_a)

    actors.snow(frame, plates, t, snow_amount(t), moon_body, None)
    actors.birds(frame, plates, t, birds(t), stillness=still)

    # The Waking / Glitch De-rezzing Transition into the 1970s Telemetry Studio & Front-Facing Buddha
    veil = dream(t)
    if veil > 0.005:
        # Front-facing Buddha in walnut telemetry suite with animated CRT screen feeds
        physical_reality = _render_studio_screens(plates["lab"], t, plates)
        
        # Keep the transition uncanny but calm; large flashes and long tears
        # interrupt the sustained listening the film is built for.
        eff_veil = veil
        if 0.02 < veil < 0.98:
            flicker = math.sin(t * 9.0) * math.cos(t * 5.2)
            micro = 0.035 * flicker
            eff_veil = np.clip(veil + micro, 0.0, 1.0)
            
        mixed = frame * (1.0 - eff_veil) + physical_reality * eff_veil
        
        # A very occasional, narrow CRT tear keeps the reveal tangible without
        # turning it into a high-energy glitch montage.
        if 0.04 < veil < 0.96 and int(t * 24) % 29 == 0:
            glitched = mixed.copy()
            rng_val = int(t * 150)
            for i in range(2):
                y_start = (rng_val * 61 + i * 197) % (HEIGHT - 18)
                band_h = 3 + (rng_val * 7 + i * 11) % 8
                y_end = min(HEIGHT, y_start + band_h)
                shift_x = ((rng_val + i * 13) % 13) - 6
                if shift_x != 0:
                    band = glitched[y_start:y_end]
                    band[:, :, 0] = np.roll(band[:, :, 0], shift_x, axis=1)
                    band[:, :, 2] = np.roll(band[:, :, 2], -shift_x, axis=1)
                    band[:, :, :] = np.clip(band * 1.035 + 3, 0, 255)
                    glitched[y_start:y_end] = band
            frame = glitched
        else:
            frame = mixed


    exposure, warmth = scene_light(t)
    if exposure > 1.01:
        frame *= exposure
        frame[:, :, 0] += 48.0 * warmth
        frame[:, :, 1] += 22.0 * warmth
    frame *= plates["vignette"]

    # Apply scene-specific "SHE Crystal Nocturne" teal-and-orange grade before end card
    frame = grade.crystal_nocturne(frame, t)

    # Give the awakened studio a short, gentle handoff to a restrained brand
    # end card. The final two seconds fade both picture and credits to black.
    if t >= OUTRO_TRANSITION_START and plates.get("outro_card") is not None:
        card_mix = smooth((t - OUTRO_TRANSITION_START) / 2.0)
        card = outro.animate_outro_card(plates["outro_card"], t)
        frame = frame * (1.0 - card_mix) + card * card_mix
    if t >= OUTRO_FADE_START:
        picture_fade = smooth((t - OUTRO_FADE_START) / (DURATION - OUTRO_FADE_START))
        frame *= 1.0 - picture_fade
    return np.clip(frame, 0, 255).astype(np.uint8)
