"""Static plates. Built once, stored as numpy arrays, reused every frame."""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from she_film.clock import HEIGHT, HORIZON, WIDTH
from she_film import video_sprites

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(__file__), "cache")
LOGO_PATH = os.path.join(ROOT_DIR, "assets", "logo-circle.png")


def _sky(color_fn):
    img = np.zeros((HORIZON, WIDTH, 3), dtype=np.float32)
    for y in range(HORIZON):
        img[y, :, :] = color_fn(y / float(HORIZON))
    return img


def _ridge_mask(amp1, dec1, amp2, dec2, base_y, freq1, p1, freq2):
    pts = [(0, HORIZON)]
    for x in range(0, WIDTH + 20, 12):
        ridge = amp1 * math.exp(-x / dec1) + amp2 * math.exp(-(WIDTH - x) / dec2)
        my = HORIZON - base_y - int(ridge + 22 * math.sin(x * freq1 + p1) + 12 * math.cos(x * freq2))
        pts.append((x, my))
    pts.append((WIDTH, HORIZON))
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    return np.asarray(mask).astype(np.float32) / 255.0


def _sun_sprite():
    size, core = 860, 280
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = size // 2
    for rad in range(420, core, -6):
        alpha = int(50 * ((420 - rad) / 140.0))
        d.ellipse([c - rad, c - rad, c + rad, c + rad], fill=(255, 170, 60, alpha))
    im = im.filter(ImageFilter.GaussianBlur(18))
    d = ImageDraw.Draw(im)
    for rad in range(core, 0, -3):
        s = rad / float(core)
        d.ellipse(
            [c - rad, c - rad, c + rad, c + rad],
            fill=(255, int(230 * (1 - s * 0.45)), int(150 * (1 - s * 0.75)), 255),
        )
    return np.asarray(im)


def _moon_sprite():
    """Mostly lit. The dark cap is on the east, the left edge."""
    size, cover = 156, 0.13
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    pad = 8
    md.ellipse([pad, pad, size - pad, size - pad], fill=255)
    shift = size * 0.2
    md.ellipse([pad - shift, pad, size - pad - shift, size - pad], fill=0)
    mask = mask.filter(ImageFilter.GaussianBlur(1.1))
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    earth = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(earth).ellipse([pad, pad, size - pad, size - pad], fill=(206, 214, 224, 18))
    im.paste(earth, (0, 0))
    lit = Image.new("RGBA", (size, size), (244, 242, 234, 255))
    lit.putalpha(mask)
    im.paste(lit, (0, 0), lit)
    return np.asarray(im)


def _font(size, light=True):
    # Cross-platform font resolution: Windows, Linux (Ubuntu/Colab), with scalable Pillow fallback
    candidates = [
        r"C:\Windows\Fonts\segoeuisl.ttf" if light else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _title_cards():
    """Each line is its own plate. The rule is a separate plate so it can lead."""
    specs = [
        ("BETWEEN  THE  NOTES  PRESENTS", 26, (20, 72, 84, 255), (186, 72, 52, 255), 188, True),
        ("S    H    E", 96, (255, 252, 246, 255), None, 268, False),
        ("A FILM SET TO THE COMPOSITION BY DRONE WATCH", 24, (28, 24, 20, 255), (20, 72, 84, 210), 352, True),
        ("ORIGINAL SOUND    ATANDRA CHAKRABORTY (BURO)", 22, (28, 24, 20, 255), None, 408, False),
        ("CINEMA  AND  DIRECTION    SAYAN BHATTACHARYA", 22, (28, 24, 20, 255), None, 442, False),
        ("VIDEO PRODUCED BY SAYAN BHATTACHARYA", 22, (28, 24, 20, 255), None, 476, False),
    ]
    cards = []
    for text, size, fill, rule, y, _ruled in specs:
        font = _font(size)
        probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        box = probe.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        pad = 8
        words = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
        ImageDraw.Draw(words).text((pad - box[0], pad - box[1]), text, font=font, fill=fill)
        line = None
        if rule is not None:
            span = tw if _ruled else 240
            line = Image.new("RGBA", (span + pad * 2, 8), (0, 0, 0, 0))
            ImageDraw.Draw(line).rectangle([pad, 2, pad + span, 5], fill=rule)
            line = np.asarray(line)
        cards.append((np.asarray(words), line, 960 - (tw + pad * 2) / 2, y - th / 2))
    return cards


def _shroom(d, x, base, scale, opened, woody, hero=False):
    """A bell on a thick stalk. The foot flares so the ridge can swallow it."""
    cap_h = 0.0
    rise = 0.3 + 0.7 * min(1.0, opened + 0.25)
    stem_h = (48 + 150 * scale) * rise
    if hero:
        stem_h *= 1.15
    if woody > 0:
        stem_h += 22 * woody * scale
    top = base - stem_h
    foot = max(10, 14 + 16 * scale)
    neck = foot * 0.42
    d.polygon(
        [
            (x - foot * 1.7, base + 6),
            (x + foot * 1.7, base + 6),
            (x + neck, top + 6),
            (x - neck * 0.8, top + 6),
        ],
        fill=(78, 56, 42, 245),
    )
    d.polygon(
        [(x - foot * 0.2, base - stem_h * 0.35), (x - foot * 1.8, base + 4), (x - foot * 0.3, base + 4)],
        fill=(64, 48, 38, 230),
    )
    d.polygon(
        [(x + foot * 0.15, base - stem_h * 0.4), (x + foot * 1.6, base + 4), (x + foot * 0.2, base + 4)],
        fill=(64, 48, 38, 220),
    )
    d.line([(x - neck * 0.2, base - 8), (x - neck * 0.05, top + 8)], fill=(124, 92, 68, 160), width=2)
    if hero:
        band = top + stem_h * 0.34
        d.rounded_rectangle([x - 8, band, x + 18, band + 26], radius=2, fill=(58, 60, 64, 220))
        d.rectangle([x - 1, band + 8, x + 12, band + 13], fill=(176, 198, 190, 170))
    if opened > 0.12:
        cap_w = (34 + 72 * scale) * min(1.0, opened) * (1.2 if hero else 1.0)
        cap_h = cap_w * 0.62
        d.chord(
            [x - cap_w, top - cap_h * 0.55, x + cap_w, top + cap_h * 0.85],
            200, 340, fill=(122, 70, 50, 245),
        )
        d.pieslice(
            [x - cap_w * 0.96, top - cap_h * 0.35, x + cap_w * 0.96, top + cap_h * 0.7],
            205, 335, fill=(168, 118, 82, 210),
        )
        d.ellipse(
            [x - cap_w * 0.72, top + cap_h * 0.02, x + cap_w * 0.72, top + cap_h * 0.72],
            fill=(92, 54, 42, 235),
        )
        ribs = 16 if hero else 9
        for g in range(-ribs, ribs + 1):
            d.line(
                [(x, top + cap_h * 0.12), (x + g * cap_w * 0.042, top + cap_h * 0.62)],
                fill=(186, 150, 116, 165),
                width=1,
            )
        d.ellipse([x - cap_w * 0.15, top - cap_h * 0.02, x + cap_w * 0.42, top + cap_h * 0.22], fill=(74, 90, 64, 165))
        d.ellipse([x - cap_w * 0.62, top - cap_h * 0.18, x - cap_w * 0.28, top + cap_h * 0.05], fill=(88, 102, 70, 140))
        d.line([(x + cap_w * 0.28, top + 4), (x + cap_w * 0.15, base - stem_h * 0.2)], fill=(48, 50, 54, 170), width=2)
        d.line([(x - cap_w * 0.18, top + 10), (x - foot * 0.9, base - stem_h * 0.15)], fill=(48, 50, 54, 140), width=1)
        if opened > 0.6 and hero:
            for s in range(16):
                sx = x - cap_w * 0.35 + (s % 5) * cap_w * 0.14
                sy = top + cap_h * 0.35 + (s // 5) * 14
                d.ellipse([sx, sy, sx + 3, sy + 3], fill=(236, 216, 168, 200))
    if woody > 0.45:
        canopy = (22 + 36 * scale) * woody
        d.ellipse(
            [x - canopy, top - cap_h * 0.45 - canopy * 0.35, x + canopy * 0.7, top + canopy * 0.05],
            fill=(58, 74, 56, int(160 * woody)),
        )


def _life():
    """One stand rooted in the near land. Flow AI-generated high-quality mushrooms with botanical canopy."""
    mush_dir = os.path.join(CACHE, "mushrooms")
    ancestral_path = os.path.join(mush_dir, "ancestral_pagoda.png")
    ancestral_spr = Image.open(ancestral_path).convert("RGBA") if os.path.exists(ancestral_path) else None

    # Dignified Japanese Zen Bonsai layout: Hero centerpiece and two gracefully spaced companions (340px negative space)
    colony = [
        (520, 1.10, True),   # Center Hero Ancestral Pagoda Bonsai
        (180, 0.65, False),  # West Companion Pagoda (generous negative space)
        (860, 0.60, False),  # East Companion Pagoda (generous negative space)
    ]
    stages = []
    width, height = 1040, 560
    for stage in range(8):
        im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        base = height - 10
        opened = min(1.0, max(0.0, (stage - 0.3) / 2.4))
        growth = min(1.0, 0.25 + 0.75 * opened)
        
        if stage == 0:
            # Soft ethereal ascending spores
            for x, _scale, _hero in colony:
                for s in range(4):
                    sy = base - 18 - s * 16
                    d.ellipse([x - 2, sy - 2, x + 2, sy + 2], fill=(225, 240, 210, 180))
                    d.ellipse([x - 1, sy - 1, x + 1, sy + 1], fill=(255, 255, 245, 230))
        else:
            # Render solely the Ancestral Pagoda Bonsai Mushroom with clean negative space
            for x, scale, hero in colony:
                if ancestral_spr is not None:
                    sw, sh = ancestral_spr.size
                    target_w = max(24, int(sw * scale * (0.50 + 0.50 * growth)))
                    target_h = max(30, int(sh * scale * growth))
                    scaled = ancestral_spr.resize((target_w, target_h), Image.Resampling.LANCZOS).copy()
                    pos = (x - target_w // 2, base - target_h)
                    im.paste(scaled, pos, scaled)
                else:
                    _shroom(d, x, base, scale, opened, 0.0, hero)

        stages.append(np.asarray(im))
    return stages


def _seat(awake):
    """Facing us, seated in the one chair. The visor is the dream."""
    im = Image.new("RGBA", (160, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(80, 78), (18, 190), (142, 190)], fill=(176, 98, 52, 250))
    d.ellipse([28, 86, 132, 150], fill=(150, 78, 42, 240))
    d.ellipse([38, 6, 122, 96], fill=(214, 170, 140, 250))
    d.ellipse([22, 40, 48, 78], fill=(196, 150, 124, 240))
    d.ellipse([112, 40, 138, 78], fill=(196, 150, 124, 240))
    if awake:
        d.arc([52, 40, 108, 72], 200, 340, fill=(48, 36, 30, 230), width=2)
    else:
        d.rounded_rectangle([40, 44, 120, 66], radius=8, fill=(176, 198, 206, 250))
        d.rectangle([58, 50, 102, 60], fill=(232, 246, 250, 235))
    return np.asarray(im)


def _sauropod_sprite():
    """Hero Giant Sauropod: massive columnar limbs, sweeping neck, counterbalancing high tail."""
    im = Image.new("RGBA", (500, 280), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    hide = (120, 108, 80, 255)
    shade = (70, 58, 48, 255)
    belly = (165, 148, 115, 255)
    light = (215, 185, 130, 255)
    rim = (255, 225, 150, 255)
    
    # Torso & lower flank countershading
    d.polygon([(110, 125), (170, 95), (290, 95), (350, 120), (370, 155), (320, 185), (180, 185), (120, 155)], fill=hide)
    d.ellipse((140, 85, 330, 185), fill=hide)
    d.ellipse((150, 120, 310, 180), fill=belly)
    
    # Tail held high
    d.polygon([(120, 135), (70, 115), (20, 85), (5, 75), (25, 95), (80, 135), (130, 155)], fill=hide)
    d.line([(5, 75), (20, 85), (70, 115), (120, 135)], fill=rim, width=2)
    
    # Arched neck & head
    base_n = (325, 120)
    hx, hy = 360, 28
    mx, my = 345, 72
    d.line([base_n, (mx, my), (hx, hy)], fill=hide, width=32, joint="curve")
    d.line([(base_n[0]+2, base_n[1]-8), (mx+2, my-8), (hx+2, hy-8)], fill=rim, width=3, joint="curve")
    d.ellipse((hx - 20, hy - 14, hx + 20, hy + 14), fill=hide)
    d.ellipse((hx + 5, hy - 7, hx + 10, hy - 2), fill=(30, 24, 20, 255))
    d.ellipse((hx + 7, hy - 5, hx + 9, hy - 3), fill=(255, 235, 160, 255))
    
    # Columnar limbs
    for lx in (185, 295):
        d.polygon([(lx - 12, 150), (lx - 10, 245), (lx + 16, 245), (lx + 18, 155)], fill=shade)
        d.line([(lx - 8, 160), (lx - 8, 240)], fill=light, width=4)
        d.polygon([(lx - 14, 240), (lx + 20, 240), (lx + 18, 248), (lx - 16, 248)], fill=(190, 165, 120, 255))
    return np.asarray(im)


def _hadrosaur_sprite():
    """Hadrosaur: backward sweeping tubular crest, duck bill, bipedal stance."""
    im = Image.new("RGBA", (380, 220), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    hide = (145, 95, 62, 255)
    belly = (185, 135, 90, 255)
    shade = (85, 55, 38, 255)
    rim = (255, 215, 140, 255)
    
    d.polygon([(90, 95), (160, 65), (250, 75), (280, 105), (260, 140), (160, 140), (100, 120)], fill=hide)
    d.ellipse((120, 65, 260, 140), fill=belly)
    d.polygon([(100, 105), (40, 85), (5, 65), (35, 80), (70, 105), (100, 120)], fill=hide)
    d.line([(5, 65), (40, 85), (100, 105)], fill=rim, width=2)
    
    d.polygon([(245, 85), (285, 45), (320, 40), (300, 75), (265, 110)], fill=hide)
    d.polygon([(300, 42), (345, 52), (348, 65), (312, 68), (295, 60)], fill=hide)
    d.polygon([(305, 45), (265, 22), (275, 18), (318, 38)], fill=(175, 75, 48, 255))
    d.line([(275, 18), (318, 38), (345, 52)], fill=rim, width=2)
    d.ellipse((314, 48, 318, 52), fill=(25, 20, 18, 255))
    d.ellipse((316, 49, 317, 50), fill=(255, 240, 180, 255))
    
    d.polygon([(185, 120), (195, 195), (218, 195), (210, 120)], fill=shade)
    d.polygon([(220, 120), (230, 195), (252, 195), (240, 120)], fill=shade)
    d.polygon([(260, 115), (272, 148), (280, 145), (268, 112)], fill=hide)
    return np.asarray(im)


def _ankylosaur_sprite():
    """Armoured Ankylosaur: dorsal conical scutes, lateral spikes, tail club."""
    im = Image.new("RGBA", (340, 180), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    hide = (95, 98, 72, 255)
    armor = (135, 138, 105, 255)
    shade = (55, 58, 42, 255)
    rim = (245, 215, 135, 255)
    
    d.ellipse((60, 50, 260, 120), fill=hide)
    for sx in range(80, 240, 24):
        d.polygon([(sx, 52), (sx + 8, 38), (sx + 16, 52)], fill=armor)
        d.line([(sx, 52), (sx + 8, 38)], fill=rim, width=2)
        d.polygon([(sx - 4, 88), (sx + 6, 98), (sx + 16, 88)], fill=armor)
        
    d.polygon([(245, 72), (295, 78), (285, 102), (238, 95)], fill=hide)
    d.polygon([(262, 70), (272, 58), (278, 72)], fill=armor)
    d.ellipse((275, 82, 279, 86), fill=(25, 20, 15, 255))
    
    d.polygon([(75, 78), (25, 72), (28, 88), (78, 92)], fill=hide)
    d.ellipse((5, 66, 32, 94), fill=armor)
    d.line([(10, 68), (26, 74)], fill=rim, width=2)
    
    for lx in (105, 145, 195, 235):
        d.polygon([(lx - 8, 105), (lx - 6, 152), (lx + 12, 152), (lx + 10, 105)], fill=shade)
    return np.asarray(im)


def _raptor_sprite():
    """Feathered Raptor: horizontal spine, quill crest, balanced tail fan, sickle claw."""
    im = Image.new("RGBA", (240, 140), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    hide = (155, 82, 64, 255)
    feather = (215, 125, 75, 255)
    shade = (85, 45, 35, 255)
    rim = (255, 225, 160, 255)
    
    d.ellipse((75, 42, 165, 82), fill=hide)
    d.polygon([(85, 58), (15, 48), (2, 42), (18, 55), (85, 68)], fill=hide)
    for tx in range(5, 55, 10):
        d.line([(tx, 45), (tx - 6, 36)], fill=feather, width=2)
    d.line([(2, 42), (85, 58)], fill=rim, width=2)
    
    d.polygon([(155, 52), (188, 28), (218, 26), (205, 48), (160, 68)], fill=hide)
    for nx in range(165, 195, 8):
        d.line([(nx, 42), (nx - 4, 30)], fill=feather, width=2)
    d.ellipse((200, 32, 204, 36), fill=(25, 20, 15, 255))
    d.ellipse((202, 33, 203, 34), fill=(255, 240, 160, 255))
    
    d.polygon([(162, 65), (175, 82), (180, 78), (168, 62)], fill=shade)
    d.polygon([(118, 72), (125, 122), (140, 122), (135, 72)], fill=shade)
    d.polygon([(142, 72), (148, 122), (162, 122), (155, 72)], fill=shade)
    return np.asarray(im)


def _cosmic_demigod():
    path = os.path.join(os.path.dirname(__file__), "cache", "buddha_cosmic_demigod.jpg")
    im = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    return np.asarray(im).astype(np.float32)


def _retro_telemetry_sleep():
    path = os.path.join(os.path.dirname(__file__), "cache", "buddha_retro_studio_sleep.jpg")
    im = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    return np.asarray(im).astype(np.float32)


def _face_meditating():
    path = os.path.join(os.path.dirname(__file__), "cache", "buddha_human_face_meditating.jpg")
    im = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    return np.asarray(im).astype(np.float32)


def _face_awake():
    path = os.path.join(os.path.dirname(__file__), "cache", "buddha_human_face_awake.jpg")
    im = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    return np.asarray(im).astype(np.float32)


def _lab(awake=False):
    return _retro_telemetry_sleep()


def _buddha():
    """2D Bodhisattva portrait with iconographic features and VR visor."""
    im = Image.new("RGBA", (760, 820), (0, 0, 0, 0))
    halo = Image.new("RGBA", im.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    for r, a in ((305, 18), (270, 22), (232, 28)):
        hd.ellipse((380-r, 242-r, 380+r, 242+r), fill=(109, 179, 232, a))
    im.alpha_composite(halo.filter(ImageFilter.GaussianBlur(36)))
    d = ImageDraw.Draw(im)
    bronze, light, shade = (74,117,119,255), (139,179,164,255), (37,67,76,255)
    gold = (209,167,101,245)
    d.ellipse((75,650,685,790), fill=(31,35,54,238), outline=(199,160,100,230), width=5)
    d.polygon([(110,707),(222,612),(350,647),(382,708),(520,617),(648,710),(610,755),(160,755)], fill=bronze)
    d.arc((116,657,646,786), 190, 350, fill=(175,194,164,230), width=8)
    d.polygon([(273,390),(214,423),(154,565),(194,681),(566,681),(607,564),(541,423),(475,390)], fill=bronze)
    d.polygon([(276,403),(326,472),(282,676),(210,662),(180,574),(238,442)], fill=light)
    d.polygon([(487,401),(545,456),(584,566),(548,661),(474,672),(439,482)], fill=shade)
    for pts in ([(241,445),(308,506),(251,637)],[(270,429),(348,502),(321,649)],[(510,441),(433,513),(496,639)],[(475,431),(404,502),(430,645)]):
        d.line(pts, fill=(195,180,132,205), width=6, joint="curve")
    d.ellipse((272,594,489,658), fill=(161,181,151,255), outline=(217,188,132,220), width=4)
    d.arc((297,603,458,648), 195, 345, fill=(67,91,87,220), width=4)
    d.polygon([(333,355),(427,355),(442,422),(380,457),(318,422)], fill=bronze)
    d.ellipse((273,127,487,403), fill=light, outline=gold, width=4)
    d.ellipse((244,220,298,341), fill=bronze, outline=gold, width=4)
    d.ellipse((462,220,516,341), fill=bronze, outline=gold, width=4)
    d.ellipse((318,78,442,196), fill=shade, outline=gold, width=4)
    for row in range(4):
        for col in range(7):
            x, y = 299 + col*27 + (row%2)*12, 145 + row*22
            d.ellipse((x,y,x+15,y+15), fill=(48,73,79,255), outline=(173,151,108,190), width=2)
    d.ellipse((362,65,398,105), fill=(101,129,119,255), outline=gold, width=3)  # ushnisha
    d.arc((251,132,509,319), 188, 352, fill=(18,26,39,255), width=20)
    d.rounded_rectangle((263,222,497,294), radius=23, fill=(12,22,36,255), outline=(188,202,192,255), width=5)
    d.rounded_rectangle((278,235,482,279), radius=15, fill=(28,95,122,255), outline=(96,200,222,255), width=3)
    d.line((292,241,370,241), fill=(191,232,233,220), width=3)
    d.line((278,285,242,309), fill=(180,191,177,230), width=8)
    d.line((482,285,518,309), fill=(180,191,177,230), width=8)
    d.ellipse((371,201,389,219), fill=gold)  # urna, partly above the visor
    d.line((381,291,373,329,381,341), fill=(46,73,73,255), width=6)
    d.arc((345,348,414,374), 15, 165, fill=(52,73,71,255), width=5)
    d.line((277,281,274,366), fill=gold, width=4)
    d.line((483,281,486,366), fill=gold, width=4)
    d.ellipse((264,341,284,380), outline=gold, width=4)
    d.ellipse((476,341,496,380), outline=gold, width=4)
    return np.asarray(im)


def _ship_sprite():
    im = Image.new("RGBA", (220, 90), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([18, 28, 202, 70], fill=(226, 214, 190, 235))
    d.ellipse([78, 12, 146, 40], fill=(190, 204, 214, 230))
    for x in (48, 110, 168):
        d.ellipse([x - 6, 46, x + 6, 58], fill=(255, 210, 140, 245))
    return np.asarray(im)


def _ship_sizes(sprite):
    im = Image.fromarray(sprite)
    out = []
    for scale in (0.62, 0.9, 1.2, 1.55):
        w = max(8, int(im.width * scale))
        h = max(8, int(im.height * scale))
        out.append(np.asarray(im.resize((w, h), Image.Resampling.BILINEAR)))
    return out


def _crystal():
    im = Image.new("RGBA", (70, 70), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(35, 8), (16, 40), (34, 62)], fill=(230, 242, 246, 210))
    d.polygon([(35, 8), (54, 38), (36, 62)], fill=(255, 250, 244, 230))
    d.line([(35, 8), (16, 40)], fill=(255, 220, 150, 230), width=2)
    d.line([(35, 8), (54, 38)], fill=(255, 220, 150, 230), width=2)
    return np.asarray(im)


def _logo():
    possible_paths = [
        LOGO_PATH,
        os.path.join(ROOT_DIR, "assets", "logo-circle.png"),
        os.path.join(ROOT_DIR, "public", "images", "logo-circle.png"),
        os.path.join(os.path.dirname(ROOT_DIR), "public", "images", "logo-circle.png"),
        r"e:\betweenthenotes\public\images\logo-circle.png",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                src = Image.open(p).convert("RGBA").resize((460, 460), Image.Resampling.LANCZOS)
                return np.asarray(src)
            except Exception:
                continue
    # Procedural circular crest fallback if image asset is unavailable
    im = Image.new("RGBA", (460, 460), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([10, 10, 450, 450], outline=(230, 242, 246, 220), width=4)
    d.ellipse([25, 25, 435, 435], fill=(22, 26, 32, 200))
    return np.asarray(im)


def _stripes():
    im = Image.new("RGBA", (WIDTH, HORIZON), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(48, HORIZON - 70, 18):
        d.rectangle([1668, y, 1864, y + 8], fill=(32, 148, 168, 170))
    return np.asarray(im)


def _vignette():
    yy = np.linspace(-1.0, 1.05, HEIGHT, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32)[None, :]
    return (1.0 - 0.28 * np.clip(xx ** 2 * 0.55 + yy ** 2, 0, 1))[..., None]


def _nebula():
    """Soft, painterly 2D nebula plate behind the final headset portrait."""
    im = Image.new("RGBA", (WIDTH, HORIZON), (0, 0, 0, 0))
    cloud = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(cloud)
    clouds = [
        (250, 310, 760, 170, (41, 106, 214, 96)),
        (620, 285, 790, 210, (155, 50, 196, 108)),
        (990, 270, 650, 190, (245, 103, 102, 84)),
        (1370, 330, 780, 230, (244, 145, 55, 88)),
        (1050, 430, 880, 150, (54, 165, 190, 66)),
    ]
    for cx, cy, rx, ry, color in clouds:
        d.ellipse((cx-rx,cy-ry,cx+rx,cy+ry), fill=color)
    im.alpha_composite(cloud.filter(ImageFilter.GaussianBlur(64)))
    wisps = Image.new("RGBA", im.size, (0, 0, 0, 0))
    wd = ImageDraw.Draw(wisps)
    for i in range(14):
        y = 100 + i * 24
        start = -120 + i * 31
        wd.arc((start,y-110,start+1020,y+170), 188, 352, fill=(161+min(70,i*5),108,224,34), width=22)
    im.alpha_composite(wisps.filter(ImageFilter.GaussianBlur(24)))
    dust = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dd = ImageDraw.Draw(dust)
    rng = np.random.default_rng(31)
    for x, y, r in zip(rng.integers(20, WIDTH-20, 170), rng.integers(10, HORIZON-30, 170), rng.integers(1, 4, 170)):
        dd.ellipse((int(x-r),int(y-r),int(x+r),int(y+r)), fill=(232,220,255,int(rng.integers(60,190))))
    im.alpha_composite(dust)
    return np.asarray(im)


def _load_sprite(name):
    p = os.path.join(CACHE, name)
    if os.path.exists(p):
        im = Image.open(p).convert("RGBA")
        return _rgba(np.asarray(im))
    return None


def load():
    print("Building plates...", flush=True)
    day = _sky(lambda r: (168 * (1 - r) + 246 * r, 196 * (1 - r) + 214 * r, 214 * (1 - r) + 186 * r))
    warm = _sky(lambda r: (
        248 * (1 - r) + 236 * r,
        168 * ((1 - r) ** 1.15) + 96 * (r ** 0.9),
        52 * (1 - r) + 32 * r,
    ))
    night = _sky(lambda r: (12 * (1 - r) + 46 * r, 16 * (1 - r) + 34 * r, 40 * (1 - r) + 74 * r))
    water_day = np.zeros((HEIGHT - HORIZON, 1, 3), np.float32)
    water_warm = np.zeros_like(water_day)
    water_night = np.zeros_like(water_day)
    frac = np.linspace(0, 1, HEIGHT - HORIZON, dtype=np.float32)[:, None]
    water_warm[:, 0, 0] = 42 * (1 - frac[:, 0]) + 22 * frac[:, 0]
    water_warm[:, 0, 1] = 118 * (1 - frac[:, 0]) + 78 * frac[:, 0]
    water_warm[:, 0, 2] = 132 * (1 - frac[:, 0]) + 96 * frac[:, 0]
    water_day[:, 0, 0] = 28 * (1 - frac[:, 0]) + 12 * frac[:, 0]
    water_day[:, 0, 1] = 140 * (1 - frac[:, 0]) + 100 * frac[:, 0]
    water_day[:, 0, 2] = 176 * (1 - frac[:, 0]) + 148 * frac[:, 0]
    water_night[:, 0, 0] = 8 * (1 - frac[:, 0]) + 5 * frac[:, 0]
    water_night[:, 0, 1] = 22 * (1 - frac[:, 0]) + 14 * frac[:, 0]
    water_night[:, 0, 2] = 48 * (1 - frac[:, 0]) + 32 * frac[:, 0]
    ridges = (
        _ridge_mask(120, 800, 70, 700, 50, 0.0035, 0.6, 0.007),
        _ridge_mask(95, 650, 50, 600, 35, 0.0042, 1.2, 0.0085),
        _ridge_mask(75, 500, 35, 550, 22, 0.005, 2.1, 0.01),
        _ridge_mask(55, 400, 25, 450, 0, 0.006, 1.5, 0.012),
    )
    land = np.clip(ridges[0] + ridges[1] + ridges[2] + ridges[3], 0, 1)
    rng = np.random.default_rng(7)
    audio_feats = None
    root_dir = os.path.dirname(os.path.dirname(__file__))
    feat_path = os.path.join(root_dir, "audio_features_24fps.npy")
    if os.path.exists(feat_path):
        try:
            audio_feats = np.load(feat_path)
        except Exception:
            audio_feats = None

    plates = {
        "audio_feats": audio_feats,
        "day": day,
        "warm": warm,
        "night": night,
        "water_day": water_day,
        "water_warm": water_warm,
        "water_night": water_night,
        "ridges": ridges,
        "land": land,
        "sun": _sun_sprite(),
        "moon": _moon_sprite(),
        "titles": _title_cards(),
        "life": _life(),
        "buddha": _buddha(),
        "nebula": _nebula(),
        "ships": _ship_sizes(_ship_sprite()),
        "beasts": (_sauropod_sprite(), _hadrosaur_sprite(), _ankylosaur_sprite(), _raptor_sprite()),
        "seat": _seat(False),
        "seat_awake": _seat(True),
        "cosmic_demigod": _cosmic_demigod(),
        "lab": _retro_telemetry_sleep(),
        "face_meditating": _face_meditating(),
        "face_awake": _face_awake(),
        "logo": _logo(),
        "stripes": _stripes(),
        "vignette": _vignette(),
        "stars": (
            rng.integers(20, WIDTH - 20, 180),
            rng.integers(12, HORIZON - 70, 180),
            rng.uniform(0.35, 1.0, 180).astype(np.float32),
            rng.uniform(0, math.tau, 180).astype(np.float32),
        ),
        "flakes": (
            rng.uniform(0, WIDTH, 160).astype(np.float32),
            rng.uniform(0, 1, 160).astype(np.float32),
            rng.uniform(28, 70, 160).astype(np.float32),
            rng.uniform(0.6, 1.6, 160).astype(np.float32),
        ),
    }
    plates["sun_src"] = plates["sun"]
    moon = _rgba(plates["moon"])
    plates["moon"] = moon
    crimson = moon.copy()
    crimson[:, :, 0] = np.clip(crimson[:, :, 0] * 0.45 + 210 * crimson[:, :, 3], 0, 255)
    crimson[:, :, 1] = crimson[:, :, 1] * 0.42
    crimson[:, :, 2] = crimson[:, :, 2] * 0.38
    plates["moon_red"] = crimson
    plates["logo"] = _rgba(plates["logo"])
    plates["buddha"] = _rgba(plates["buddha"])
    plates["nebula"] = _rgba(plates["nebula"])
    plates["stripes"] = _crop_rgba(plates["stripes"])
    packed = []
    for words, rule, x, y in plates["titles"]:
        packed.append((_rgba(words), None if rule is None else _rgba(rule), x, y))
    plates["titles"] = packed
    plates["life"] = [_rgba(s) for s in plates["life"]]
    plates["ships"] = [_rgba(s) for s in plates["ships"]]

    # Load Flow Pokemon character sprites
    plates["pokemon_sauropod"] = _load_sprite("pokemon_sauropod.png")
    plates["pokemon_hadrosaur"] = _load_sprite("pokemon_hadrosaur.png")
    plates["pokemon_ankylosaur"] = _load_sprite("pokemon_ankylosaur.png")
    plates["pokemon_raptor"] = _load_sprite("pokemon_raptor.png")
    plates["pokemon_pterodactyl"] = _load_sprite("pokemon_pterodactyl.png")
    plates["pokemon_leviathan"] = _load_sprite("pokemon_leviathan.png")
    plates["pokemon_flying_fish"] = _load_sprite("pokemon_flying_fish.png")
    plates["pokemon_plesiosaur"] = _load_sprite("pokemon_plesiosaur.png")

    # Animated video sprites (Flow green screen video loops)
    plates["anim_sauropod"] = video_sprites.load_cached_video_sprite("sauropod_walk")
    plates["anim_pterodactyl"] = video_sprites.load_cached_video_sprite("pterodactyl_flight")
    plates["anim_leviathan"] = video_sprites.load_cached_video_sprite("leviathan_breach")
    plates["anim_hadrosaur"] = video_sprites.load_cached_video_sprite("hadrosaur_walk")
    plates["anim_ankylosaur"] = video_sprites.load_cached_video_sprite("ankylosaur_walk")
    plates["anim_raptor"] = video_sprites.load_cached_video_sprite("raptor_walk")
    plates["anim_mushroom_growth"] = video_sprites.load_cached_video_sprite("anim_mushroom_growth")
    plates["anim_mushroom_idle"] = video_sprites.load_cached_video_sprite("anim_mushroom_idle")

    screens_npz = os.path.join(CACHE, "screens_90s.npz")
    if os.path.exists(screens_npz):
        try:
            plates["screens_90s"] = np.load(screens_npz)
        except Exception:
            plates["screens_90s"] = None
    else:
        plates["screens_90s"] = None

    dino_list = []
    for spr_key, fallback in [
        ("pokemon_sauropod", _sauropod_sprite()),
        ("pokemon_hadrosaur", _hadrosaur_sprite()),
        ("pokemon_ankylosaur", _ankylosaur_sprite()),
        ("pokemon_raptor", _raptor_sprite()),
    ]:
        if plates[spr_key] is not None:
            dino_list.append(plates[spr_key])
        else:
            dino_list.append(_rgba(fallback))
    plates["beasts"] = dino_list

    plates["seat"] = _rgba(plates["seat"])
    plates["seat_awake"] = _rgba(plates["seat_awake"])

    # Load celestial shimmer star sprite (Nano Banana Flow generation)
    star_path = os.path.join(CACHE, "celestial_shimmer_star.png")
    if not os.path.exists(star_path):
        star_path = os.path.join(ROOT_DIR, "assets", "celestial_shimmer_star.png")
    if os.path.exists(star_path):
        plates["star_sprite"] = _rgba(np.asarray(Image.open(star_path).convert("RGBA")))
    else:
        plates["star_sprite"] = None

    print("Plates ready.", flush=True)
    return plates


def _rgba(arr):
    out = arr.astype(np.float32)
    if out[:, :, 3].max() > 1.0:
        out[:, :, 3] *= 1.0 / 255.0
    return out


def _crop_rgba(arr):
    alpha = arr[:, :, 3]
    ys, xs = np.nonzero(alpha)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return _rgba(arr[y0:y1, x0:x1]), x0, y0
