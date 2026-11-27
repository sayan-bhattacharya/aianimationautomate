import os
import glob
import cv2
import numpy as np

OUT_DIR = r"e:\betweenthenotes\drone_animation_project\assets\blender_clean"
os.makedirs(OUT_DIR, exist_ok=True)


def fill_interior_alpha_holes(bgra: np.ndarray) -> np.ndarray:
    """Keep exterior background transparent via border flood-fill; force interior holes to opaque."""
    h, w = bgra.shape[:2]
    alpha = bgra[:, :, 3]
    # Binary mask of clearly transparent pixels
    trans = (alpha < 20).astype(np.uint8)
    # Pad by 1px so flood-fill from (0,0) reaches all exterior borders
    padded = np.pad(trans, ((1, 1), (1, 1)), mode="constant", constant_values=1)
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, mask, (0, 0), 2)
    exterior = padded[1:-1, 1:-1] == 2
    out = bgra.copy()
    # Any pixel NOT connected to the exterior background is inside the dinosaur silhouette -> opaque
    out[~exterior, 3] = 255
    return out


def clean_botanical_base(bgra: np.ndarray, bottom_frac: float = 0.16) -> np.ndarray:
    """Strip baked-in dark/grey rectangular floor pads near the bottom while preserving roots."""
    h, w = bgra.shape[:2]
    out = bgra.copy()
    bot_start = int(h * (1.0 - bottom_frac))
    sub = out[bot_start:, :]
    b, g, r, a = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2], sub[:, :, 3]
    # Suppress low-alpha fringe and dark neutral studio floor/shadow pads
    sat = np.maximum(np.maximum(b, g), r).astype(np.int16) - np.minimum(np.minimum(b, g), r).astype(np.int16)
    luma = (0.114 * b + 0.587 * g + 0.299 * r)
    # Neutral dark/grey pad pixels with partial alpha or very dark flat shadow
    pad_mask = ((a > 0) & (a < 185) & (sat < 28) & (luma < 95)) | (a < 50)
    sub[pad_mask, 3] = 0
    # Soft 6px feather on the very bottom edge to avoid any hard horizontal line
    feather = min(8, sub.shape[0])
    fade = np.linspace(1.0, 0.0, feather, dtype=np.float32)[:, None]
    sub[-feather:, :, 3] = (sub[-feather:, :, 3].astype(np.float32) * fade).astype(np.uint8)
    out[bot_start:, :] = sub
    return out


def upscale_4k_plate(bgra: np.ndarray, min_long_edge: int = 2048) -> np.ndarray:
    """Upscale sprite plates cleanly with Lanczos4 so 4K renders stay razor-sharp."""
    h, w = bgra.shape[:2]
    long_edge = max(h, w)
    if long_edge >= min_long_edge:
        return bgra
    scale = min_long_edge / float(long_edge)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(bgra, (nw, nh), interpolation=cv2.INTER_LANCZOS4)


def main():
    dinos = [
        r"e:\betweenthenotes\drone_animation_project\she_film\cache\pokemon_sauropod.png",
        r"e:\betweenthenotes\drone_animation_project\she_film\cache\pokemon_hadrosaur.png",
        r"e:\betweenthenotes\drone_animation_project\she_film\cache\pokemon_ankylosaur.png",
        r"e:\betweenthenotes\drone_animation_project\she_film\cache\pokemon_raptor.png",
        r"e:\betweenthenotes\drone_animation_project\she_film\cache\pokemon_pterodactyl.png",
    ]
    for p in dinos:
        if not os.path.exists(p):
            continue
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        cleaned = upscale_4k_plate(fill_interior_alpha_holes(img), min_long_edge=2048)
        out_p = os.path.join(OUT_DIR, os.path.basename(p))
        cv2.imwrite(out_p, cleaned)
        print(f"Dino cleaned (4K plate): {out_p} {cleaned.shape[:2]}")

    botanicals = [
        r"e:\betweenthenotes\drone_animation_project\she_film\cache\mushrooms\hero_pagoda_mushroom_master.png",
        r"e:\betweenthenotes\drone_animation_project\assets\whorl_firs\full_leaf_variants\pine_fuller_warm.png",
        r"e:\betweenthenotes\drone_animation_project\assets\whorl_firs\full_leaf_variants\pine_fuller_deep.png",
        r"e:\betweenthenotes\drone_animation_project\assets\whorl_firs\full_leaf_variants\pine_fuller_cool.png",
    ]
    for p in botanicals:
        if not os.path.exists(p):
            continue
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        cleaned = upscale_4k_plate(clean_botanical_base(img), min_long_edge=2048)
        out_p = os.path.join(OUT_DIR, os.path.basename(p))
        cv2.imwrite(out_p, cleaned)
        print(f"Botanical cleaned (4K plate): {out_p} {cleaned.shape[:2]}")


if __name__ == "__main__":
    main()
