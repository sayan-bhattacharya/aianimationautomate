"""
Chroma-keys, downscales, and manages animated video sprites from Flow green-screen video generations.
"""
import os
import cv2
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "video_sprites")
os.makedirs(CACHE_DIR, exist_ok=True)


class VideoSprite:
    def __init__(self, name, frames_rgba, fps=24.0):
        self.name = name
        self.frames = frames_rgba  # float32 array of shape (N, H, W, 4), alpha in 0..1
        self.fps = fps
        self.n_frames = len(frames_rgba) if frames_rgba is not None else 0

    def get_frame(self, t, loop=True):
        if self.n_frames == 0 or self.frames is None:
            return None
        idx = int(t * self.fps)
        if loop:
            idx = idx % self.n_frames
        else:
            idx = min(idx, self.n_frames - 1)
        return self.frames[idx]


def process_green_screen_video(mp4_path, target_size, cache_name, flip_h=False):
    """
    Extracts frames from an MP4 video, keys out pure chroma green (#00FF00),
    suppresses spill, crops bounding box smoothly, downsamples to target_size,
    optionally flips horizontally (e.g. for creatures facing East to face West),
    and saves to a fast NPZ cache.
    """
    out_npz = os.path.join(CACHE_DIR, f"{cache_name}.npz")
    if not os.path.exists(mp4_path):
        print(f"[VideoSprite] Video not found: {mp4_path}")
        return None

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"[VideoSprite] Failed to open: {mp4_path}")
        return None

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    # 1. First pass: find overall bounding box across frames for consistent framing
    all_raw_frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        all_raw_frames.append(frame_bgr)
    cap.release()

    if not all_raw_frames:
        print(f"[VideoSprite] No frames in: {mp4_path}")
        return None

    h_orig, w_orig = all_raw_frames[0].shape[:2]
    min_x, min_y, max_x, max_y = 10000, 10000, 0, 0

    masks = []
    for frame_bgr in all_raw_frames:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Studio chroma green screen criteria
        is_green = (hsv[..., 0] >= 42) & (hsv[..., 0] <= 82) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 75)
        mask_bg = (is_green * 255).astype(np.uint8)

        # Flood fill from the 4 outer image borders so only external background is keyed out
        mask_flood = mask_bg.copy()
        ff_mask = np.zeros((h_orig + 2, w_orig + 2), np.uint8)
        cv2.floodFill(mask_flood, ff_mask, (0, 0), 255)
        cv2.floodFill(mask_flood, ff_mask, (w_orig - 1, 0), 255)
        cv2.floodFill(mask_flood, ff_mask, (0, h_orig - 1), 255)
        cv2.floodFill(mask_flood, ff_mask, (w_orig - 1, h_orig - 1), 255)

        alpha = np.where(mask_flood == 255, 0, 255).astype(np.uint8)

        # Fill internal holes on the creature body (preserving genuine air between limbs)
        contours, hierarchy = cv2.findContours(alpha, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        solid_alpha = alpha.copy()
        if hierarchy is not None:
            for cnt, hier in zip(contours, hierarchy[0]):
                if hier[3] != -1:  # Internal hole
                    if cv2.contourArea(cnt) < 4000:  # Fill scale/feather holes, keep wide leg gaps
                        cv2.drawContours(solid_alpha, [cnt], 0, 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        solid_alpha = cv2.morphologyEx(solid_alpha, cv2.MORPH_CLOSE, kernel)
        masks.append(solid_alpha)

        coords = cv2.findNonZero(solid_alpha)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

    pad = 8
    if min_x < max_x and min_y < max_y:
        x0 = max(0, min_x - pad)
        y0 = max(0, min_y - pad)
        x1 = min(w_orig, max_x + pad)
        y1 = min(h_orig, max_y + pad)
    else:
        x0, y0, x1, y1 = 0, 0, w_orig, h_orig

    # 2. Process each frame
    tw, th = target_size
    processed_frames = []
    for frame_bgr, alpha in zip(all_raw_frames, masks):
        cropped_bgr = frame_bgr[y0:y1, x0:x1]
        cropped_alpha = alpha[y0:y1, x0:x1]

        # Despill
        b, g, r = cv2.split(cropped_bgr)
        excess_g = g.astype(float) - np.maximum(r, b).astype(float)
        g_despill = np.where(excess_g > 0, np.maximum(r, b), g).astype(np.uint8)
        despilled_bgr = cv2.merge([b, g_despill, r])

        # Convert to RGB float32
        rgb = cv2.cvtColor(despilled_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        a = (cropped_alpha.astype(np.float32) / 255.0)[:, :, None]

        rgba = np.concatenate([rgb, a], axis=2)
        scaled = cv2.resize(rgba, (tw, th), interpolation=cv2.INTER_AREA)
        if flip_h:
            scaled = cv2.flip(scaled, 1)
        processed_frames.append(scaled)

    frames_arr = np.stack(processed_frames, axis=0)  # (N, th, tw, 4)
    np.savez_compressed(out_npz, frames=frames_arr, fps=fps)
    print(f"[VideoSprite] Saved {len(processed_frames)} frames to: {out_npz}")
    return VideoSprite(cache_name, frames_arr, fps=fps)


def load_cached_video_sprite(cache_name):
    out_npz = os.path.join(CACHE_DIR, f"{cache_name}.npz")
    if os.path.exists(out_npz):
        try:
            data = np.load(out_npz)
            frames = data["frames"]
            if frames.dtype == np.uint8:
                frames_f = frames.astype(np.float32)
                frames_f[..., 3] /= 255.0
                frames = frames_f
            return VideoSprite(cache_name, frames, fps=float(data["fps"]))
        except Exception as e:
            print(f"[VideoSprite] Error loading {out_npz}: {e}")
    return None
