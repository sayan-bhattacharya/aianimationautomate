"""
Mushroom Lab: Realistic Anatomical Generative Fungal Module.
Implements:
  - Anatomical Pileus (cap) with wavy margins, rolled edges, thickness, and asymmetry.
  - Radial Lamellae (gills) radiating from stipe apex, curving with underside.
  - Stipe (stem) with longitudinal fibrous striations, bulbous base, and curvature.
  - Age progression: button -> juvenile -> mature -> flattened -> upturned / collapsing.
  - Lighting: Directional form shading, warm rim light, teal ambient bounce, cap translucency.
  - Bioluminescence (Mycena chlorophos): Day is strictly MATTE. Night glows 525nm cool green-cyan
    with local moss bounce/spill.
  - 7 Distinct Species Presets.
  - High-res baking (2x/3x supersampled) with growth series.
"""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from opensimplex import OpenSimplex

DIR_OUTPUT = r"e:\betweenthenotes\drone_animation_project\mushroom_lab_output"
os.makedirs(DIR_OUTPUT, exist_ok=True)

# 525nm bioluminescent spectral color (Mycena chlorophos green-cyan)
COLOR_BIO_EMERALD = np.array([45, 235, 160], dtype=np.float32)
COLOR_MOSS = np.array([38, 54, 32], dtype=np.float32)

class MushroomSpec:
    def __init__(self, name, cap_color, gill_color, stem_color,
                 cap_shape="convex", cap_radius=60, stem_length=110,
                 stem_thickness=12, has_annulus=False, bioluminescent=False,
                 cap_waviness=0.15, cap_striations=True):
        self.name = name
        self.cap_color = np.array(cap_color, dtype=np.float32)
        self.gill_color = np.array(gill_color, dtype=np.float32)
        self.stem_color = np.array(stem_color, dtype=np.float32)
        self.cap_shape = cap_shape # "convex", "bell", "funnel", "flat", "conical"
        self.cap_radius = cap_radius
        self.stem_length = stem_length
        self.stem_thickness = stem_thickness
        self.has_annulus = has_annulus
        self.bioluminescent = bioluminescent
        self.cap_waviness = cap_waviness
        self.cap_striations = cap_striations

# 7 Botanical Species Presets
SPECIES = {
    "mycena_chlorophos": MushroomSpec(
        name="Mycena chlorophos",
        cap_color=[195, 205, 190],
        gill_color=[215, 225, 210],
        stem_color=[220, 230, 215],
        cap_shape="bell",
        cap_radius=52,
        stem_length=125,
        stem_thickness=8,
        bioluminescent=True,
        cap_waviness=0.18,
        cap_striations=True
    ),
    "cantharellus_cibarius": MushroomSpec(
        name="Cantharellus cibarius",
        cap_color=[235, 175, 55],
        gill_color=[245, 195, 75],
        stem_color=[225, 165, 45],
        cap_shape="funnel",
        cap_radius=68,
        stem_length=85,
        stem_thickness=16,
        bioluminescent=False,
        cap_waviness=0.28,
        cap_striations=False
    ),
    "amanita_muscaria": MushroomSpec(
        name="Amanita muscaria",
        cap_color=[210, 48, 38],
        gill_color=[240, 238, 230],
        stem_color=[245, 242, 235],
        cap_shape="convex",
        cap_radius=75,
        stem_length=130,
        stem_thickness=20,
        has_annulus=True,
        bioluminescent=False,
        cap_waviness=0.12,
        cap_striations=False
    ),
    "coprinus_comatus": MushroomSpec(
        name="Coprinus comatus",
        cap_color=[230, 226, 218],
        gill_color=[180, 175, 170],
        stem_color=[240, 238, 232],
        cap_shape="conical",
        cap_radius=42,
        stem_length=150,
        stem_thickness=14,
        bioluminescent=False,
        cap_waviness=0.14,
        cap_striations=True
    ),
    "psilocybe_semilanceata": MushroomSpec(
        name="Psilocybe semilanceata",
        cap_color=[175, 135, 85],
        gill_color=[120, 95, 70],
        stem_color=[205, 185, 150],
        cap_shape="conical",
        cap_radius=35,
        stem_length=115,
        stem_thickness=6,
        bioluminescent=False,
        cap_waviness=0.10,
        cap_striations=True
    ),
    "pleurotus_ostreatus": MushroomSpec(
        name="Pleurotus ostreatus",
        cap_color=[145, 140, 135],
        gill_color=[225, 220, 215],
        stem_color=[210, 205, 200],
        cap_shape="flat",
        cap_radius=72,
        stem_length=70,
        stem_thickness=18,
        bioluminescent=False,
        cap_waviness=0.24,
        cap_striations=False
    ),
    "panellus_stipticus": MushroomSpec(
        name="Panellus stipticus",
        cap_color=[190, 160, 120],
        gill_color=[205, 175, 135],
        stem_color=[180, 150, 110],
        cap_shape="flat",
        cap_radius=45,
        stem_length=50,
        stem_thickness=10,
        bioluminescent=True,
        cap_waviness=0.22,
        cap_striations=False
    )
}

def generate_mushroom_sprite(spec, age=1.0, seed=42, light_dir=(-0.6, -0.8), is_night=False, scale=2):
    """
    Renders an anatomically articulated mushroom at `scale`x supersampling.
    age:
      0.15 = button
      0.45 = juvenile
      0.80 = mature
      1.00 = expanded/flattened
      1.25 = upturned / collapsing
    """
    gen = OpenSimplex(seed=seed)
    W = int(320 * scale)
    H = int(360 * scale)
    cx = W // 2
    base_y = int(310 * scale)

    # Effective anatomical dimensions based on age
    eff_stem_len = spec.stem_length * min(1.0, age * 1.2) * scale
    eff_stem_thick = spec.stem_thickness * (0.6 + 0.4 * min(1.0, age)) * scale
    eff_cap_rad = spec.cap_radius * (0.35 + 0.65 * min(1.1, age)) * scale

    stem_top_y = base_y - eff_stem_len
    # Slight lean using noise
    lean_angle = gen.noise2(seed * 0.1, age) * 0.14 # radians
    stem_top_x = cx + math.sin(lean_angle) * eff_stem_len

    # RGBA float buffer
    canvas = np.zeros((H, W, 4), dtype=np.float32)

    # 1. GROUNDING: Contact Shadow & AO Base
    shadow_w = int(eff_cap_rad * 1.3)
    shadow_h = int(14 * scale)
    y_grid, x_grid = np.ogrid[:H, :W]
    dist_shadow = ((x_grid - cx) / float(shadow_w))**2 + ((y_grid - (base_y - 2*scale)) / float(shadow_h))**2
    shadow_mask = np.clip(1.0 - dist_shadow, 0.0, 1.0)**2
    # Apply ground shadow
    canvas[:, :, :3] += np.array([12, 14, 16], dtype=np.float32) * shadow_mask[:, :, None] * 0.65
    canvas[:, :, 3] = np.maximum(canvas[:, :, 3], shadow_mask * 0.65)

    # 2. STEM (STIPE) GENERATION
    # Cubic spline spine with fibrous longitudinal striations
    num_pts = 60
    stem_pts = []
    for i in range(num_pts):
        frac = i / float(num_pts - 1)
        # Tapering: bulbous at base, slightly thinner mid, flare at apex
        w_factor = 1.4 * (1.0 - frac)**2 + 0.85 * math.sin(frac * math.pi) + 1.15 * frac**3
        pt_x = cx + (stem_top_x - cx) * frac + gen.noise2(seed, frac * 3.0) * (4 * scale)
        pt_y = base_y - frac * eff_stem_len
        r = (eff_stem_thick * 0.5) * w_factor
        stem_pts.append((pt_x, pt_y, r))

    # 2. STEM (STIPE) GENERATION
    # Render stem onto a high-res RGBA image using anti-aliased convex polygons
    stem_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d_stem = ImageDraw.Draw(stem_img)

    for i in range(num_pts - 1):
        x0, y0, r0 = stem_pts[i]
        x1, y1, r1 = stem_pts[i + 1]
        
        # Quadrilateral connecting successive stem discs
        poly = [
            (x0 - r0, y0),
            (x0 + r0, y0),
            (x1 + r1, y1),
            (x1 - r1, y1)
        ]
        frac = i / float(num_pts)
        shading = 0.65 + 0.35 * math.sin(frac * math.pi)
        col = tuple(np.clip(spec.stem_color * shading, 0, 255).astype(int)) + (255,)
        d_stem.polygon(poly, fill=col)

    # Longitudinal fibrous striations on stem
    for f_idx in range(-int(eff_stem_thick * 0.4), int(eff_stem_thick * 0.4), max(1, int(2 * scale))):
        f_pts = [(pt_x + f_idx + math.sin(i * 0.2) * (1.5 * scale), pt_y) for i, (pt_x, pt_y, _) in enumerate(stem_pts)]
        f_col = tuple(np.clip(spec.stem_color * 0.82, 0, 255).astype(int)) + (180,)
        d_stem.line(f_pts, fill=f_col, width=max(1, int(1.2 * scale)))

    arr_stem = np.array(stem_img, dtype=np.float32)
    s_a = arr_stem[:, :, 3:4] / 255.0
    canvas[:, :, :3] = canvas[:, :, :3] * (1.0 - s_a) + arr_stem[:, :, :3] * s_a
    canvas[:, :, 3] = np.maximum(canvas[:, :, 3], arr_stem[:, :, 3] / 255.0)

    # 3. GILLS (LAMELLAE) UNDERSIDE
    # Gills radiate outward from stem_top to cap margin
    num_gills = 36
    cap_center_x = stem_top_x
    cap_center_y = stem_top_y

    gill_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d_gill = ImageDraw.Draw(gill_img)

    for g in range(num_gills):
        angle = (g / float(num_gills)) * 2.0 * math.pi
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
        # Underside projection (flattened oval)
        r_outer = eff_cap_rad * 0.95
        x_end = cap_center_x + cos_a * r_outer
        y_end = cap_center_y + sin_a * (r_outer * 0.32) + (10 * scale)

        # Draw curving gill plate
        x_mid = (cap_center_x + x_end) * 0.5 + float(gen.noise2(seed + g, 0.5)) * (3 * scale)
        y_mid = (cap_center_y + y_end) * 0.5 + 4 * scale

        gill_shade = 0.55 + 0.35 * max(0.0, -sin_a)
        g_col = tuple(np.clip(spec.gill_color * gill_shade, 0, 255).astype(int)) + (220,)
        d_gill.line([(cap_center_x, cap_center_y + 4*scale), (x_mid, y_mid), (x_end, y_end)],
                    fill=g_col, width=max(1, int(1.2 * scale)))

    arr_gill = np.array(gill_img, dtype=np.float32)
    ag = arr_gill[:, :, 3:4] / 255.0
    canvas[:, :, :3] = canvas[:, :, :3] * (1.0 - ag) + arr_gill[:, :, :3] * ag
    canvas[:, :, 3] = np.maximum(canvas[:, :, 3], arr_gill[:, :, 3] / 255.0)

    # 4. CAP (PILEUS) GENERATION
    # Mathematical curvature based on species and age
    # Profile: y_offset = f(x_rel)
    res_theta = 120
    cap_poly = []

    # Apex position
    apex_h = eff_cap_rad * 0.85
    if spec.cap_shape == "funnel":
        apex_h = -eff_cap_rad * 0.25 # depressed center
    elif spec.cap_shape == "flat":
        apex_h = eff_cap_rad * 0.35
    elif spec.cap_shape == "conical":
        apex_h = eff_cap_rad * 1.35

    # Upturn effect in old age
    upturn = max(0.0, age - 1.0) * 0.8

    # Generate upper cap silhouette
    for t_step in range(res_theta + 1):
        theta = math.pi * (t_step / float(res_theta)) # 0 to pi (right to left)
        cos_t = math.cos(theta) # 1 to -1
        sin_t = math.sin(theta) # 0 to 1

        # Irregular wavy margin noise
        margin_noise = gen.noise2(seed * 2.0 + cos_t * 3.0, age * 2.0) * spec.cap_waviness

        # Rolled edge thickness
        rad = eff_cap_rad * (1.0 + margin_noise)

        px = cap_center_x + cos_t * rad
        # Curvature equation
        if spec.cap_shape == "bell":
            arch = (math.cos(cos_t * math.pi * 0.5)) ** 1.6
        elif spec.cap_shape == "funnel":
            arch = - (1.0 - (cos_t ** 2)) ** 0.8
        elif spec.cap_shape == "conical":
            arch = (1.0 - abs(cos_t)) ** 0.95
        else: # convex
            arch = math.sqrt(max(0.0, 1.0 - cos_t**2))

        py = cap_center_y - arch * apex_h + (cos_t**2) * (upturn * 22 * scale)
        cap_poly.append((px, py))

    # Complete closed polygon around underside rolled margin
    for t_step in range(res_theta, -1, -1):
        theta = math.pi * (t_step / float(res_theta))
        cos_t = math.cos(theta)
        rad_inner = eff_cap_rad * (0.92 + gen.noise2(seed * 3.0 + cos_t * 2.0, age) * spec.cap_waviness)
        px = cap_center_x + cos_t * rad_inner
        py = cap_center_y + (math.sin(theta) * 0.22 * eff_cap_rad) + (4 * scale)
        cap_poly.append((px, py))

    # Rasterize cap via PIL with supersampled antialiasing
    cap_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d_cap = ImageDraw.Draw(cap_img)
    d_cap.polygon(cap_poly, fill=(255, 255, 255, 255))

    arr_cap = np.array(cap_img, dtype=np.float32)
    cap_mask = arr_cap[:, :, 3] / 255.0

    # 5. CAP LIGHTING & SHADING
    # 2D Normal Map synthesis for cap
    # Height field gradient
    gy, gx = np.gradient(cap_mask)
    norm_len = np.sqrt(gx**2 + gy**2 + 0.08)
    nx = -gx / norm_len
    ny = -gy / norm_len
    nz = 0.4 / norm_len

    # Light direction dot product
    lx, ly = light_dir
    l_norm = math.hypot(lx, ly) if math.hypot(lx, ly) > 1e-5 else 1.0
    lx /= l_norm
    ly /= l_norm

    form_light = np.clip(nx * (-lx) + ny * (-ly) + nz * 0.4, 0.12, 1.25)
    # Warm golden rim light on top-left edge
    rim = np.clip((-nx * lx + -ny * ly), 0.0, 1.0) ** 3.5
    # Subsurface translucency at thin cap margin
    translucency = np.clip(1.0 - norm_len * 4.0, 0.0, 1.0) * max(0.0, -ly)

    # Striation lines along cap surface (Mycena / Coprinus)
    if spec.cap_striations:
        rel_x = (x_grid - cap_center_x) / float(eff_cap_rad)
        striations = (np.sin(rel_x * 42.0) * 0.08)[:, :, None]
    else:
        striations = 0.0

    cap_rgb = spec.cap_color * (form_light[:, :, None] + striations)
    cap_rgb += np.array([255, 205, 120], dtype=np.float32) * (rim[:, :, None] * 0.55)
    cap_rgb += np.array([240, 160, 80], dtype=np.float32) * (translucency[:, :, None] * 0.35)

    # 6. BIOLUMINESCENCE (Mycena chlorophos) RULE:
    # Daytime is 100% MATTE. Bioluminescence is ONLY visible at night!
    if spec.bioluminescent and is_night:
        # 525nm Emerald Cyan glow concentrated at gills and cap underside
        # In Mycena chlorophos, the gills glow most intensely (~525nm peak)
        dist_from_stipe = np.sqrt((x_grid - cap_center_x)**2 + (y_grid - cap_center_y)**2)
        gill_glow_zone = np.clip(1.0 - (dist_from_stipe / (eff_cap_rad * 0.95)), 0.0, 1.0) ** 1.8
        glow_intensity = gill_glow_zone * cap_mask

        # Blend emerald luminescence
        cap_rgb = cap_rgb * 0.25 + COLOR_BIO_EMERALD * (glow_intensity[:, :, None] * 1.8)

        # Light spill on ground / moss
        moss_dist = np.sqrt((x_grid - cx)**2 + ((y_grid - base_y) * 2.5)**2)
        moss_spill = np.clip(1.0 - (moss_dist / (eff_cap_rad * 1.6)), 0.0, 1.0)**2
        canvas[:, :, :3] += COLOR_BIO_EMERALD * (moss_spill[:, :, None] * 0.42)
        canvas[:, :, 3] = np.maximum(canvas[:, :, 3], moss_spill * 0.45)

    # Blend cap onto canvas
    c_a = cap_mask[:, :, None]
    canvas[:, :, :3] = canvas[:, :, :3] * (1.0 - c_a) + cap_rgb * c_a
    canvas[:, :, 3] = np.maximum(canvas[:, :, 3], cap_mask)

    # Downsample from supersampled buffer to crisp, antialiased print-quality resolution
    final_w = W // scale
    final_h = H // scale
    img_sup = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGBA")
    img_down = img_sup.resize((final_w, final_h), Image.Resampling.LANCZOS)
    return img_down


def render_species_and_age_lineup():
    """Generates a complete lineup sheet of all 7 species across 5 distinct developmental ages."""
    print("Generating Species & Developmental Age Lineup...")
    ages = [
        (0.20, "1. Button"),
        (0.50, "2. Juvenile"),
        (0.80, "3. Mature"),
        (1.05, "4. Flattened"),
        (1.25, "5. Collapsing")
    ]
    
    # 7 species rows, 5 age columns
    thumb_w, thumb_h = 320, 360
    sheet_w = thumb_w * len(ages) + 200
    sheet_h = thumb_h * len(SPECIES) + 120

    sheet = Image.new("RGBA", (sheet_w, sheet_h), (24, 22, 20, 255))
    draw = ImageDraw.Draw(sheet)

    for r_idx, (sp_key, spec) in enumerate(SPECIES.items()):
        y_pos = 90 + r_idx * thumb_h
        # Species label
        draw.text((30, y_pos + thumb_h // 2 - 10), spec.name, fill=(225, 215, 195, 255))
        
        for c_idx, (age, age_label) in enumerate(ages):
            x_pos = 180 + c_idx * thumb_w
            if r_idx == 0:
                draw.text((x_pos + thumb_w // 2 - 35, 45), age_label, fill=(245, 195, 80, 255))

            # Ground baseline line
            draw.line([(x_pos + 20, y_pos + thumb_h - 45), (x_pos + thumb_w - 20, y_pos + thumb_h - 45)],
                      fill=(55, 50, 45, 255), width=2)

            m_sprite = generate_mushroom_sprite(spec, age=age, seed=100 + r_idx * 17, is_night=False, scale=2)
            sheet.paste(m_sprite, (x_pos, y_pos), m_sprite)

    out_path = os.path.join(DIR_OUTPUT, "01_mushroom_species_age_lineup.jpg")
    sheet.convert("RGB").save(out_path, quality=95)
    print(f"[SAVED] Lineup sheet: {out_path}")
    return out_path


def render_day_vs_night_cluster():
    """Generates side-by-side cluster comparison: Daytime Matte vs Nighttime 525nm Bioluminescence."""
    print("Generating Day vs Night Cluster Comparison...")
    W, H = 1920, 1080
    canvas = Image.new("RGBA", (W, H), (18, 16, 15, 255))
    draw = ImageDraw.Draw(canvas)

    # Half line
    draw.line([(W // 2, 0), (W // 2, H)], fill=(45, 40, 36, 255), width=3)

    # Left: Daytime Matte Cluster
    draw.text((120, 60), "DAYTIME MATTE COLONY (Solar Rim & Form Shading)", fill=(250, 205, 110, 255))
    # Right: Nighttime Mycena chlorophos
    draw.text((W // 2 + 120, 60), "NIGHTTIME BIOLUMINESCENCE (Mycena chlorophos 525nm)", fill=(45, 235, 160, 255))

    # Cluster configuration (x_offsets, scales, ages)
    cluster_cfg = [
        (-140, 0.75, 0.35, 101),
        (-80, 1.15, 0.95, 102),
        (-10, 1.35, 1.10, 103), # Hero
        (70, 0.95, 0.75, 104),
        (130, 0.65, 0.40, 105),
        (190, 0.85, 0.85, 106),
    ]

    base_ground_y = 860

    # Draw Daytime Cluster on Left
    center_left = W // 4
    for dx, s, age, sd in cluster_cfg:
        sp = SPECIES["mycena_chlorophos"]
        sprite = generate_mushroom_sprite(sp, age=age, seed=sd, light_dir=(-0.7, -0.7), is_night=False, scale=2)
        sw, sh = int(sprite.width * s), int(sprite.height * s)
        sp_resized = sprite.resize((sw, sh), Image.Resampling.LANCZOS)
        canvas.paste(sp_resized, (center_left + dx - sw // 2, base_ground_y - sh + int(45*s)), sp_resized)

    # Draw Nighttime Cluster on Right
    center_right = W * 3 // 4
    for dx, s, age, sd in cluster_cfg:
        sp = SPECIES["mycena_chlorophos"]
        sprite = generate_mushroom_sprite(sp, age=age, seed=sd, light_dir=(0.5, -0.8), is_night=True, scale=2)
        sw, sh = int(sprite.width * s), int(sprite.height * s)
        sp_resized = sprite.resize((sw, sh), Image.Resampling.LANCZOS)
        canvas.paste(sp_resized, (center_right + dx - sw // 2, base_ground_y - sh + int(45*s)), sp_resized)

    out_path = os.path.join(DIR_OUTPUT, "02_day_vs_night_cluster.jpg")
    canvas.convert("RGB").save(out_path, quality=95)
    print(f"[SAVED] Day vs Night Cluster: {out_path}")
    return out_path


def render_100pct_anatomical_crops():
    """Extracts 100% full-resolution anatomical crops showing pileus edge, lamellae gills, and stipe fibers."""
    print("Generating 100% Anatomical Crops...")
    # Render hero Mycena at scale=3 for pristine detail inspection
    hero = generate_mushroom_sprite(SPECIES["mycena_chlorophos"], age=0.9, seed=777, scale=3)
    
    # 1. Cap Margin & Lamellae crop
    crop_cap = hero.crop((hero.width // 2 - 80, 40, hero.width // 2 + 100, 170))
    crop_cap_path = os.path.join(DIR_OUTPUT, "03_crop_pileus_and_lamellae_100pct.png")
    crop_cap.save(crop_cap_path)

    # 2. Stipe (Stem) Longitudinal Fibers & Base crop
    crop_stipe = hero.crop((hero.width // 2 - 50, 150, hero.width // 2 + 60, 310))
    crop_stipe_path = os.path.join(DIR_OUTPUT, "04_crop_stipe_fibers_100pct.png")
    crop_stipe.save(crop_stipe_path)

    print(f"[SAVED] 100% Crops:\n  - {crop_cap_path}\n  - {crop_stipe_path}")
    return crop_cap_path, crop_stipe_path


def render_growth_clip_3s():
    """Renders a 3-second (72-frame @ 24fps) anatomical growth MP4."""
    import av
    print("Rendering 3-Second Growth Clip (72 frames @ 24fps)...")
    clip_path = os.path.join(DIR_OUTPUT, "05_mushroom_growth_3s.mp4")

    W, H = 1080, 1080
    container = av.open(clip_path, mode="w")
    stream = container.add_stream("h264", rate=24)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}

    spec = SPECIES["mycena_chlorophos"]
    num_frames = 72

    for f in range(num_frames):
        p = f / float(num_frames - 1)
        # Non-linear biological sigmoid growth curve
        age = 0.15 + 0.95 / (1.0 + math.exp(-8.0 * (p - 0.45)))
        # Slight breathing & micro-sway
        sway = math.sin(f * 0.18) * 0.05

        frame_canvas = Image.new("RGBA", (W, H), (22, 20, 18, 255))
        d_frame = ImageDraw.Draw(frame_canvas)
        # Ground
        d_frame.line([(100, 880), (W - 100, 880)], fill=(45, 42, 38, 255), width=3)
        d_frame.text((50, 50), f"MYCENA CHLOROPHOS GROWTH | Frame {f+1}/72 (t={f/24.0:.2f}s) | Age: {age:.2f}",
                     fill=(245, 195, 80, 255))

        m_sprite = generate_mushroom_sprite(spec, age=age, seed=42, light_dir=(-0.7 + sway, -0.7), scale=2)
        # Position anchored at base
        bx = W // 2 - m_sprite.width // 2
        by = 880 - m_sprite.height + 45
        frame_canvas.paste(m_sprite, (bx, by), m_sprite)

        # PyAV encode
        rgb_arr = np.array(frame_canvas.convert("RGB"))
        av_frame = av.VideoFrame.from_ndarray(rgb_arr, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"[SAVED] 3-second growth video: {clip_path}")
    return clip_path


if __name__ == "__main__":
    render_species_and_age_lineup()
    render_day_vs_night_cluster()
    render_100pct_anatomical_crops()
    render_growth_clip_3s()
