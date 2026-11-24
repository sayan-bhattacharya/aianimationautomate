"""Create a restrained, editorial end card for the SHE film with a top-down conical light beam."""
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from she_film.clock import HEIGHT, WIDTH


def _font(size, serif=False, italic=False, bold=False):
    if os.name == "nt":
        if serif and italic:
            names = [r"C:\Windows\Fonts\georgiai.ttf", r"C:\Windows\Fonts\timesi.ttf"]
        elif serif:
            names = [r"C:\Windows\Fonts\georgia.ttf", r"C:\Windows\Fonts\times.ttf"]
        elif bold:
            names = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\segoeuib.ttf"]
        else:
            names = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]
    elif serif and italic:
        names = [
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf",
        ]
    elif serif:
        names = [
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        ]
    elif bold:
        names = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]
    else:
        names = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
    for name in names:
        if os.path.exists(name):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _centered(draw, y, text, font, fill, tracking=0):
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + tracking * max(0, len(text) - 1)
    x = (WIDTH - total) / 2.0
    shadow = (4, 6, 12, 220)
    for ch, width in zip(text, widths):
        draw.text((round(x) + 2, y + 2), ch, font=font, fill=shadow)
        draw.text((round(x), y), ch, font=font, fill=fill)
        x += width + tracking


def _wordmark(draw, center_x, top):
    # Bespoke 'She' title mark: capital S carries the word, prominent golden serif styling
    S = _font(280, serif=True, italic=True)
    he = _font(200, serif=True, italic=True)
    gold = (238, 206, 140, 255)
    shadow = (6, 8, 16, 220)
    sw, sh = draw.textbbox((0, 0), "S", font=S)[2:]
    hw, hh = draw.textbbox((0, 0), "he", font=he)[2:]
    gap = -10
    width = sw + gap + hw
    x = center_x - width / 2.0
    draw.text((round(x) + 3, top + 3), "S", font=S, fill=shadow, stroke_width=0)
    draw.text((round(x + sw + gap) + 3, top + 74), "he", font=he, fill=shadow, stroke_width=0)
    draw.text((round(x), top), "S", font=S, fill=gold, stroke_width=0)
    draw.text((round(x + sw + gap), top + 72), "he", font=he, fill=gold, stroke_width=0)


def _signature_rule(draw, y):
    # Elegant centered hairline with diamond lozenge
    draw.line((540, y, 1380, y), fill=(224, 190, 127, 210), width=2)
    draw.line((700, y + 3, 1220, y + 3), fill=(224, 190, 127, 65), width=1)
    draw.polygon([(960, y - 8), (966, y), (960, y + 8), (954, y)], fill=(245, 222, 175, 255))


def build_outro_card(logo_path=None):
    """Return a 1920x1080 RGB slate with realistic top-down conical light beam,
    prominent 'She' emblem, and larger editorial typography."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)

    # 1. Base deep velvety indigo/black background
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    rgb[:, :, 0] = 8.0
    rgb[:, :, 1] = 12.0
    rgb[:, :, 2] = 20.0

    # 2. Realistic top-down conical light beam (volumetric light cone originating from top center)
    cone_apex_x, cone_apex_y = WIDTH / 2.0, -30.0
    dy = yy - cone_apex_y
    dx = np.abs(xx - cone_apex_x)
    # Conical angle spread: half-angle tan ~ 0.55 (spreads from apex down to ~1200px wide at bottom)
    cone_half_w = dy * 0.55 + 40.0
    cone_mask = np.clip(1.0 - (dx / np.maximum(1.0, cone_half_w)) ** 2, 0.0, 1.0) ** 1.8
    # Vertical falloff: strong at top, softly illuminating the full credits stack
    vert_falloff = np.clip(1.0 - (yy / (HEIGHT * 1.15)), 0.15, 1.0)
    # Soft ray-tracing streaks inside the cone
    ray_streaks = 1.0 + 0.12 * np.sin(np.arctan2(xx - cone_apex_x, dy) * 28.0)
    conical_beam = (cone_mask * vert_falloff * ray_streaks)[:, :, None]

    # Warm starlight beam color: soft champagne gold / radiant amber
    beam_color = np.array([215.0, 235.0, 255.0], dtype=np.float32)
    rgb += conical_beam * beam_color * 0.28
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")

    muted = (175, 195, 202, 240)
    gold = (235, 204, 142, 255)
    ivory = (248, 244, 236, 255)

    # 3. Prominent stylish "She" emblem / logo at top under the spotlight
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")
            logo = logo.resize((140, 140), Image.Resampling.LANCZOS)
            image.paste(logo, ((WIDTH - 140) // 2, 45), logo)
            draw = ImageDraw.Draw(image, "RGBA")
        except Exception:
            pass

    # 4. Larger, highly readable editorial credits typography
    _centered(draw, 195, "BETWEEN THE NOTES PRESENTS", _font(26, bold=True), muted, tracking=4.5)
    _wordmark(draw, WIDTH / 2, 232)
    _signature_rule(draw, 520)

    _centered(draw, 558, "ORIGINAL MUSIC", _font(24, bold=True), gold, tracking=3.8)
    _centered(draw, 598, "ATANDRA CHAKRABORTY", _font(46, serif=True), ivory, tracking=1.5)
    _centered(draw, 678, "A VISUAL MUSIC FILM BY", _font(22, bold=True), gold, tracking=3.2)
    _centered(draw, 716, "SAYAN BHATTACHARYA   |   ATANDRA CHAKRABORTY", _font(32, bold=True), ivory, tracking=1.0)
    _centered(draw, 800, "BETWEENTHENOTES.ORG", _font(24, bold=True), gold, tracking=3.0)
    _centered(draw, 850, "WHERE THE SILENCE SPEAKS", _font(20, bold=True), (185, 205, 210, 240), tracking=3.2)

    # Convert RGB -> BGR to match frame.py's OpenCV BGR pipeline
    return np.asarray(image)[:, :, ::-1].astype(np.float32)
