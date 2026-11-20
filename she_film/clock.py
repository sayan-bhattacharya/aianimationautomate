"""Full-song story clock for SHE. One readable lighting state at a time."""
import math

WIDTH = 1920
HEIGHT = 1080
HORIZON = 580
FPS = 24
DURATION = 449.064


def clamp(p, lo=0.0, hi=1.0):
    return lo if p < lo else hi if p > hi else p


def smooth(p):
    p = clamp(p)
    return p * p * (3.0 - 2.0 * p)


def lerp(a, b, p):
    return a + (b - a) * p


def _arc(t, start, end, x0=180.0, x1=1740.0, high=390.0):
    p = clamp((t - start) / (end - start))
    alt = math.sin(p * math.pi) * (high + 32.0) - 32.0
    return None if alt < -28.0 else (lerp(x0, x1, p), alt)


def sun(t):
    """Two long day arcs, separated from moonlit night by true twilight."""
    if 14.0 <= t < 122.0:
        return _arc(t, 14.0, 122.0, 150.0, 1780.0, 390.0)
    if 196.0 <= t < 300.0:
        return _arc(t, 196.0, 300.0, 140.0, 1780.0, 420.0)
    return None


def moon(t):
    """The two night journeys; the second blooms crimson during the fade."""
    if 132.0 <= t < 192.0:
        return _arc(t, 132.0, 192.0, 220.0, 1660.0, 380.0)
    if 310.0 <= t < 428.0:
        return _arc(t, 310.0, 428.0, 260.0, 1650.0, 420.0)
    return None


def crimson(t):
    return smooth((t - 306.0) / 38.0) if t >= 306.0 else 0.0


def sky_weights(t):
    """Day, warm twilight, and night exposures, crossfaded with organic golden-hour curve."""
    # Act 1 Dawn & First Sunrise (10s -> 20s)
    if t < 10.0:
        return 0.08, 0.92, 0.0
    if 10.0 <= t < 20.0:
        p = smooth((t - 10.0) / 10.0)
        return lerp(0.08, 0.90, p), lerp(0.92, 0.10, p), 0.0

    # Act 1 Day (20s -> 116s)
    if 20.0 <= t < 116.0:
        return 0.90, 0.10, 0.0

    # Act 1 Sunset (116s -> 130s): 14s warm golden hour and twilight into night
    if 116.0 <= t < 130.0:
        p = smooth((t - 116.0) / 14.0)
        glow = 0.40 * math.sin(math.pi * p)
        return 0.90 * (1.0 - p), 0.10 * (1.0 - p) + 0.04 * p + glow, 0.96 * p

    # Act 2 Night (130s -> 190s)
    if 130.0 <= t < 190.0:
        return 0.0, 0.04, 0.96

    # Act 3 Sunrise 2 (190s -> 202s): 12s golden dawn crossfade as starcraft ascends
    if 190.0 <= t < 202.0:
        p = smooth((t - 190.0) / 12.0)
        glow = 0.45 * math.sin(math.pi * p)
        return 0.90 * p, 0.04 * (1.0 - p) + 0.10 * p + glow, 0.96 * (1.0 - p)

    # Act 3 Day (202s -> 294s)
    if 202.0 <= t < 294.0:
        return 0.90, 0.10, 0.0

    # Act 4 Sunset 2 (294s -> 308s): 14s dusk into crimson night
    if 294.0 <= t < 308.0:
        p = smooth((t - 294.0) / 14.0)
        glow = 0.38 * math.sin(math.pi * p)
        return 0.90 * (1.0 - p), 0.10 * (1.0 - p) + 0.04 * p + glow, 0.96 * p

    # Act 5 Night & Cosmic Transmutation (308s -> 428s)
    if 308.0 <= t < 428.0:
        return 0.0, 0.04, 0.96

    return 0.0, 0.12, 0.88


def _cut(t, start, hold, end):
    if t < start or t >= end:
        return 0.0
    return 1.0 if t <= hold else 1.0 - smooth((t - hold) / (end - hold))


def title_lines(t):
    # Tightened typography: 30% lesser on-screen time, crisp and clear
    p = _cut(t, 8.0, 18.0, 24.0)
    film = _cut(t, 9.5, 19.5, 25.5)
    credits = _cut(t, 11.0, 21.0, 27.0)
    return ((p, p), (0.0, film), (film, film), (0.0, credits), (0.0, credits), (0.0, credits))


def visit(t):
    """Alien saucer descends from the dark lunar limb, lands, then launches."""
    if not 166.0 <= t < 207.0:
        return None
    land_x, land_y = 1120.0, float(HORIZON - 96)
    if t < 180.0:
        p = smooth((t - 166.0) / 14.0)
        return lerp(1580.0, land_x, p), lerp(30.0, land_y, p), lerp(0.1, 1.0, p)
    if t < 187.0:
        return land_x, land_y, 1.0
    p = smooth((t - 187.0) / 20.0)
    return lerp(land_x, 1740.0, p), lerp(land_y, 20.0, p), lerp(1.0, 0.12, p)


def colony(t):
    """Spore → mushrooms → broad vegetation. The forest holds for the herd."""
    age = t - 20.0
    if age <= 0 or t > 352:
        return 0.0, 0.0, 1
    alpha = smooth(age / 8.0) * (1.0 - 0.45 * smooth((t - 300.0) / 42.0))
    stage = min(7.0, age / 18.0)
    copies = 1 if t < 38 else 3 if t < 70 else 5
    return alpha, stage, copies


def birds(t):
    alpha, stage, _ = colony(t)
    if t < 242 or t > 352 or stage < 4.2:
        return 0.0
    return alpha * (1.0 - 0.45 * crimson(t))


def herd(t):
    """
    Dinosaur caravan marches gracefully East to West along the coastal shelf.
    Returns tuple of (dino_id, x) with fixed species identities:
      0: Hero Sauropod
      1: Hadrosaur
      2: Armoured Ankylosaur
      3: Feathered Raptor
    Each dinosaur maintains continuous steady stride with zero speed discrepancy or index swapping.
    """
    if not 240.0 <= t < 396.0:
        return ()
    
    # Caravan march speed: ~17.5 pixels/sec from East (2080+) to West (-350)
    pace = 17.5
    t_rel = t - 240.0
    
    dinos = [
        (0, 2080.0 - pace * t_rel),
        (1, 2360.0 - pace * t_rel),
        (2, 2620.0 - pace * t_rel),
        (3, 2860.0 - pace * t_rel),
    ]
    
    return tuple((d_id, x) for d_id, x in dinos if -350.0 <= x <= 2150.0)


def scene_light(t):
    # Slow aurora-like blooms, tied to the long final musical fade.
    if not 336 <= t <= 428:
        return 1.0, 0.0
    pulse = max(0.0, math.sin((t - 336.0) * math.pi / 18.0)) ** 2
    return 1.0 + 0.24 * pulse, 0.32 * pulse


def dream(t):
    # Professional, tighter transition into physical telemetry reality (9s dissolve)
    return smooth((t - 420.0) / 9.0) if t > 420 else 0.0


def waking(t):
    return smooth((t - 435.0) / 6.0) if t > 435 else 0.0


def growth(t):
    return colony(t)[1]


def snow_amount(t):
    # A restrained crystalline breath over the first moon passage.
    if t < 142 or t > 190:
        return 0.0
    return smooth((t - 142) / 8) * (1.0 - smooth((t - 170) / 14))


def ice(t):
    return 0.0


def water_stillness(t):
    """
    Vedantic Stillness: As the meditative Tanpura drone swells in Act 5 (t >= 318s),
    all surface wave ripples subside to absolute zero (1.0 = crystal still mirror).
    Derived from the Vedantic insight: emotions are waves in mind that distort reflection;
    when mind is completely still, water is still and reveals true pristine reflection.
    """
    if t < 318.0:
        return 0.0
    elif 318.0 <= t < 348.0:
        return smooth((t - 318.0) / 30.0)
    elif 348.0 <= t <= 422.0:
        return 1.0
    else:
        return max(0.0, 1.0 - smooth((t - 422.0) / 8.0))


def buddha(t):
    # Crisp, professional cosmic reveal transition
    return smooth((t - 350.0) / 16.0) * (1.0 - smooth((t - 422.0) / 6.0))


def audit():
    for i in range(int(DURATION * 2)):
        t = i / 2.0
        s, m = sun(t), moon(t)
        if s and m and s[1] > 0 and m[1] > 0:
            raise RuntimeError(f"sun and moon share the sky at t={t}")
