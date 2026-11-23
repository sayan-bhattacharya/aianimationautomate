"""Build distinct B-roll loops for the four ending telemetry CRTs."""
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


def _dinosaur_plate(asset_path):
    """Make a closed-loop forest-floor dinosaur plate from the alpha sprite."""
    sprite = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
    if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
        raise FileNotFoundError(f"Required dinosaur screen asset not found: {asset_path}")
    sprite = cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA)
    out_w, out_h = FRAME_SIZE
    frames = []
    for i in range(FRAME_COUNT):
        # Include the identical start pose at the end of the cache. This makes
        # the actual frame N-1 -> frame 0 transition exact, not just close.
        phase = 2.0 * np.pi * i / max(1, FRAME_COUNT - 1)
        half_phase = np.pi * i / max(1, FRAME_COUNT - 1)
        yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
        sky = np.zeros((out_h, out_w, 3), dtype=np.float32)
        # Gentle diurnal light drift across the wildlife habitat
        sun_shift = np.sin(half_phase)
        sky[:, :, 0] = 31 + 18 * (yy / out_h) + 14.0 * sun_shift
        sky[:, :, 1] = 65 + 42 * (yy / out_h) + 18.0 * sun_shift
        sky[:, :, 2] = 47 + 18 * (yy / out_h) + 8.0 * sun_shift
        # Distant forest silhouettes and a softly lit clearing.
        ridge = (190 + 8 * np.sin(xx * 0.025 + phase)
                 + 3 * np.sin(xx * 0.057 + 2.0 * phase))
        forest = yy > ridge
        sky[forest] *= np.array([0.54, 0.73, 0.55], dtype=np.float32)
        sky[yy >= 252] *= np.array([0.63, 0.67, 0.48], dtype=np.float32)
        sh, sw = sprite.shape[:2]
        scale = 0.82
        nw, nh = int(sw * scale), int(sh * scale)
        dino = cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_CUBIC)
        # The animal walks in place with a gentle slow traverse and weight shift.
        x = int(round((out_w - nw) * 0.5 + 8.0 * np.sin(half_phase)))
        y = 223 + int(round(1.5 * (1.0 - np.cos(4.0 * phase)))) - nh
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(out_w, x + nw), min(out_h, y + nh)
        if x1 > x0 and y1 > y0:
            crop = dino[y0-y:y1-y, x0-x:x1-x]
            alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
            sky[y0:y1, x0:x1] = sky[y0:y1, x0:x1] * (1.0 - alpha) + crop[:, :, :3] * alpha
        frame = np.clip(sky, 0, 255).astype(np.uint8)
        cv2.putText(frame, "WILDLIFE FIELD NOTES", (9, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.31, (229, 223, 190), 1, cv2.LINE_AA)
        frames.append(frame)
    return np.asarray(frames, dtype=np.uint8)


def _channel_hop_loop():
    """A playful channel surf: warm rom-com, analog snow, and wildlife."""
    out_w, out_h = FRAME_SIZE
    frames = []
    for i in range(FRAME_COUNT):
        section = (i * 3) // FRAME_COUNT
        if section == 0:
            yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
            glow = np.clip(1.0 - ((xx - out_w * .52) ** 2 / 15000 + (yy - out_h * .44) ** 2 / 26000), 0, 1)
            frame_f = np.zeros((out_h, out_w, 3), dtype=np.float32)
            frame_f[:] = [48, 32, 63]
            frame_f += glow[:, :, None] * np.array([158, 108, 83], dtype=np.float32)
            frame = np.clip(frame_f, 0, 255).astype(np.uint8)
            cv2.circle(frame, (out_w // 2 - 19, 208), 24, (37, 30, 50), -1, cv2.LINE_AA)
            cv2.circle(frame, (out_w // 2 + 19, 208), 24, (37, 30, 50), -1, cv2.LINE_AA)
            cv2.putText(frame, "TONIGHT ON LOVE TV", (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (255, 231, 205), 1, cv2.LINE_AA)
        elif section == 1:
            rng = np.random.default_rng(i)
            frame = rng.integers(0, 256, (out_h, out_w, 3), dtype=np.uint8)
            frame = cv2.blur(frame, (1, 3))
            for y in range(28, out_h, 41):
                if (i + y) % 3:
                    cv2.line(frame, (0, y), (out_w - 1, y), (220, 220, 220), 1)
            cv2.putText(frame, "NO SIGNAL", (64, 170), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            frame[:, :] = [28, 58, 66]
            cv2.rectangle(frame, (0, 218), (out_w, out_h), (39, 83, 62), -1)
            for j in range(4):
                x = (j * 83 + i * 2) % (out_w + 50) - 25
                cv2.ellipse(frame, (x, 165 + 7 * (j % 2)), (28, 17), 0, 0, 360,
                            (95, 131, 86), -1, cv2.LINE_AA)
            cv2.putText(frame, "NATIONAL GEOGRAPHIC", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.27, (248, 225, 166), 1, cv2.LINE_AA)
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return np.asarray(frames, dtype=np.uint8)


def build_screens_cache(output_dir=None):
    film_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(film_dir, "cache")
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(film_dir, "cache")

    # Assign distinct programs to all four monitors:
    # scr_left: Rom-com "LOVE TV" & cheetah "NATIONAL GEOGRAPHIC" channel surfing loop
    # scr_mid_left: Dinosaur closed-loop walk animation
    # scr_mid_right: Dynamic dinosaur simulation perspective loop
    # scr_right: Cosmic Buddha kaleidoscopic motion
    cosmic_buddha = _read_plate(os.path.join(cache_dir, "buddha_cosmic_demigod.jpg"))
    right_plates = [
        _detail_crop(cosmic_buddha, 0.52, 0.37, 2.15),
        _detail_crop(cosmic_buddha, 0.52, 0.69, 2.45),
    ]
    arr_right = _make_loop(right_plates, variant=1)
    arr_dino = _dinosaur_plate(os.path.join(cache_dir, "flow_pokemon_dino_scaled.png"))
    arr_channels = _channel_hop_loop()
    arr_dino_sim = _detail_crop(arr_dino, 0.5, 0.5, 1.15) if arr_dino.ndim == 3 else arr_dino[:, :, ::-1]
    
    out_npz = os.path.join(output_dir, "screens_90s.npz")
    np.savez_compressed(out_npz, scr_left=arr_channels, scr_mid_left=arr_dino,
                        scr_mid_right=arr_dino_sim, scr_right=arr_right)
    print(f"[CACHE] Built Buddha B-roll CRT loops: {out_npz} ({os.path.getsize(out_npz) / 1024 / 1024:.2f} MB)")
    return out_npz


if __name__ == "__main__":
    build_screens_cache()
