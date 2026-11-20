"""
Process Flow AI-generated mushroom stills into clean, transparent RGBA PNG overlays.
Applies HSV chroma-keying, green spill suppression, edge anti-aliasing, and creates multiple scale variants.
"""
import os
import cv2
import numpy as np

FLOW_DIR = r"C:\Users\sayan\flow-mcp-out\MushroomAssets"
OUT_DIR = r"e:\betweenthenotes\drone_animation_project\she_film\cache\mushrooms"
os.makedirs(OUT_DIR, exist_ok=True)

MUSHROOM_FILES = {
    "hero_mycena": os.path.join(FLOW_DIR, "scene-01.jpeg"),
    "golden_trumpet": os.path.join(FLOW_DIR, "scene-02.jpeg"),
    "biolum_cluster": os.path.join(FLOW_DIR, "scene-03.jpeg"),
    "ancestral_pagoda": os.path.join(FLOW_DIR, "clip-01.jpeg"),
}


def extract_mushroom_sprite(image_path, target_max_dim=320):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Pure green screen detector: H ~ 35 to 88, S > 60, V > 50
    green_mask = (h >= 35) & (h <= 88) & (s >= 55) & (v >= 45)

    # Invert to get subject mask (0 for green, 255 for subject)
    alpha = np.where(green_mask, 0, 255).astype(np.uint8)

    # Morphological cleaning: close tiny pinholes in subject, open to remove green stray specks
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel_small)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_med)

    # Edge soft anti-aliasing feathering
    alpha_blurred = cv2.GaussianBlur(alpha, (5, 5), 0)

    # Green spill suppression on foreground RGB
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)

    # Excess green clamping
    max_rb = np.maximum(r, b)
    excess_green = np.maximum(0.0, g - max_rb)
    g_despill = g - excess_green * 0.85
    r_warm = np.clip(r + excess_green * 0.15, 0, 255)
    b_cool = np.clip(b + excess_green * 0.10, 0, 255)

    despilled = np.stack([b_cool, g_despill, r_warm], axis=-1).astype(np.uint8)

    # Find tight bounding box of subject
    pts = cv2.findNonZero(alpha)
    if pts is not None:
        bx, by, bw, bh = cv2.boundingRect(pts)
        pad = 6
        bx0 = max(0, bx - pad)
        by0 = max(0, by - pad)
        bx1 = min(img.shape[1], bx + bw + pad)
        by1 = min(img.shape[0], by + bh + pad)
        cropped_rgb = despilled[by0:by1, bx0:bx1]
        cropped_a = alpha_blurred[by0:by1, bx0:bx1]
    else:
        cropped_rgb = despilled
        cropped_a = alpha_blurred

    # Resize to target maximum dimension while preserving aspect ratio
    ch, cw = cropped_rgb.shape[:2]
    scale = target_max_dim / float(max(ch, cw))
    nw, nh = int(round(cw * scale)), int(round(ch * scale))
    scaled_rgb = cv2.resize(cropped_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    scaled_a = cv2.resize(cropped_a, (nw, nh), interpolation=cv2.INTER_AREA)

    rgba = np.dstack([scaled_rgb, scaled_a])
    return rgba


def process_all():
    results = {}
    for name, path in MUSHROOM_FILES.items():
        if os.path.exists(path):
            max_dim = 340 if "hero" in name or "ancestral" in name else 240
            sprite = extract_mushroom_sprite(path, target_max_dim=max_dim)
            out_path = os.path.join(OUT_DIR, f"{name}.png")
            cv2.imwrite(out_path, sprite)
            results[name] = out_path
            print(f"[PROCESSED] {name} -> {out_path} ({sprite.shape[1]}x{sprite.shape[0]})")
        else:
            print(f"[MISSING] {path}")

    # Generate a contact sheet of all 4 isolated transparent mushrooms
    canvas = np.zeros((400, 1200, 4), dtype=np.uint8)
    # Dark grey check pattern background to demonstrate transparency
    check_w, check_h = 20, 20
    for y in range(0, 400, check_h):
        for x in range(0, 1200, check_w):
            val = 32 if ((x // check_w) + (y // check_h)) % 2 == 0 else 48
            canvas[y:y+check_h, x:x+check_w, :3] = val
            canvas[y:y+check_h, x:x+check_w, 3] = 255

    col_w = 300
    for idx, (name, path) in enumerate(results.items()):
        spr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        sh, sw = spr.shape[:2]
        cx = idx * col_w + (col_w - sw) // 2
        cy = 380 - sh
        alpha = spr[:, :, 3] / 255.0
        for c in range(3):
            canvas[cy:cy+sh, cx:cx+sw, c] = (
                canvas[cy:cy+sh, cx:cx+sw, c] * (1.0 - alpha) + spr[:, :, c] * alpha
            )
    cv2.imwrite(os.path.join(OUT_DIR, "mushrooms_lineup_verified.png"), canvas)
    print(f"[SUCCESS] Mushroom lineup saved to {os.path.join(OUT_DIR, 'mushrooms_lineup_verified.png')}")


if __name__ == "__main__":
    process_all()
