"""Create a restrained, editorial end card for the SHE film."""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from she_film.clock import HEIGHT, WIDTH


def _font(size, serif=False, italic=False, bold=False):
    if os.name == "nt":
        if serif and italic:
            names = [r"C:\Windows\Fonts\georgiai.ttf"]
        elif serif:
            names = [r"C:\Windows\Fonts\georgia.ttf"]
        elif bold:
            names = [r"C:\Windows\Fonts\arialbd.ttf"]
        else:
            names = [r"C:\Windows\Fonts\arial.ttf"]
    elif serif:
        names = (["/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf"] if italic else
                 ["/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"])
    else:
        names = (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else
                 ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for name in names:
        if os.path.exists(name):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _centered(draw, y, text, font, fill, tracking=0):
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + tracking * max(0, len(text)-1)
    x = (WIDTH-total)/2.0
    for ch, width in zip(text, widths):
        draw.text((round(x), y), ch, font=font, fill=fill)
        x += width + tracking


def _wordmark(draw, center_x, top):
    # A bespoke lockup: the capital S carries the word; he sits slightly smaller
    # and lower, so the title reads as a quiet signature rather than a heading.
    S = _font(228, serif=True, italic=True)
    he = _font(160, serif=True, italic=True)
    gold = (224, 190, 127, 255)
    sw, sh = draw.textbbox((0, 0), "S", font=S)[2:]
    hw, hh = draw.textbbox((0, 0), "he", font=he)[2:]
    gap = -8
    width = sw + gap + hw
    x = center_x - width/2
    draw.text((round(x), top), "S", font=S, fill=gold, stroke_width=0)
    # Set the lowercase ending a touch lower to give the mark a natural rhythm.
    draw.text((round(x+sw+gap), top+61), "he", font=he, fill=gold, stroke_width=0)


def _signature_rule(draw, y):
    # One perfectly straight hairline with a centered, engraved lozenge.
    draw.line((600, y, 1320, y), fill=(224,190,127,190), width=2)
    draw.line((730, y+3, 1190, y+3), fill=(224,190,127,52), width=1)
    draw.polygon([(960,y-7),(965,y),(960,y+7),(955,y)], fill=(239,213,164,255))


def build_outro_card(logo_path=None):
    """Return a clean 1920x1080 RGB slate with a custom 'She' wordmark."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    # Quiet ink-blue gradient with only a faint warm lift behind the typography.
    warmth = np.exp(-(((xx-WIDTH/2)/780.0)**2 + ((yy-460.0)/430.0)**2)*2.0)
    vertical = np.clip(yy/HEIGHT,0.0,1.0)
    rgb=np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8)
    rgb[:,:,0]=np.clip(7 + 4*warmth + 2*vertical,0,255)
    rgb[:,:,1]=np.clip(11 + 5*warmth + 3*vertical,0,255)
    rgb[:,:,2]=np.clip(19 + 8*warmth + 5*vertical,0,255)
    image=Image.fromarray(rgb)
    draw=ImageDraw.Draw(image,"RGBA")

    muted=(166,185,188,238)
    gold=(224,190,127,255)
    ivory=(239,234,223,255)
    if logo_path and os.path.exists(logo_path):
        logo=Image.open(logo_path).convert("RGBA")
        logo=logo.resize((112,112),Image.Resampling.LANCZOS)
        image.paste(logo,((WIDTH-112)//2,62),logo)
        draw=ImageDraw.Draw(image,"RGBA")
    _centered(draw,205,"BETWEEN THE NOTES PRESENTS",_font(18,bold=True),muted,tracking=4)
    _wordmark(draw,WIDTH/2,260)
    _signature_rule(draw,526)

    _centered(draw,588,"ORIGINAL MUSIC",_font(16,bold=True),gold,tracking=3)
    _centered(draw,622,"ATANDRA CHAKRABORTY",_font(29,serif=True),ivory,tracking=1)
    _centered(draw,689,"A VISUAL MUSIC FILM BY",_font(15,bold=True),gold,tracking=2.5)
    _centered(draw,721,"SAYAN BHATTACHARYA  |  ATANDRA CHAKRABORTY",_font(21),ivory,tracking=0.5)
    _centered(draw,798,"BETWEENTHENOTES.ORG",_font(15,bold=True),gold,tracking=2.2)
    _centered(draw,846,"WHERE THE SILENCE SPEAKS",_font(13,bold=True),(158,177,178,225),tracking=2.4)
    return np.asarray(image).astype(np.float32)
