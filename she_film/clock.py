"""Full-song story clock for SHE. One readable lighting state at a time."""
import math

WIDTH = 1920
HEIGHT = 1080
HORIZON = 580
FPS = 24
DURATION = 445.0  # exact 7:25 master at 24 fps
TOTAL_FRAMES = int(round(DURATION * FPS))
OUTRO_TRANSITION_START = DURATION - 10.0
OUTRO_FADE_START = DURATION - 2.0

VISIT_START = 166.0
VISIT_LAND_START = 182.0
VISIT_LAND_END = 185.0
VISIT_DEPARTURE_END = 212.0
VISIT_DEPARTURE_FADE_START = 206.0
VISIT_LANDING_X = 1120.0
VISIT_DEPARTURE_X = 1720.0
VISIT_DEPARTURE_Y = 78.0
VISIT_LANDING_SURFACE_Y = 500.0  # open inland clearing between the fir groups
VISIT_LANDING_NEAR = 0.70
VISIT_HULL_BOTTOM_FRACTION = 70.0 / 90.0  # opaque hull bottom in the source saucer sprite
MUSHROOM_EXIT_END = VISIT_START + 10.0
FIR_EMERGENCE_START = MUSHROOM_EXIT_END
FIR_EMERGENCE_END = FIR_EMERGENCE_START + 8.0


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


def star_visibility(t):
    """Calculates star emergence: ensures the night sky appears FIRST cleanly at sunset,
    followed by gradual organic kindling and shimmering of the celestial star field."""
    # Act 1 Sunset -> Act 2 Night (116s -> 190s)
    # Sunset starts 116s. Twilight settles into deep night by 126s.
    # At 123s (2m03s), stars must NOT appear prematurely (night sky appears first).
    if t < 126.0:
        return 0.0
    if 126.0 <= t < 136.0:
        # Smooth emergence from 0.0 to 1.0 between 126s and 136s
        return float(smooth((t - 126.0) / 10.0))
    if 136.0 <= t < 188.0:
        # Full night sky brilliance
        return 1.0
    if 188.0 <= t < 200.0:
        # Sunrise 2 crossfade: stars fade as dawn breaks
        return float(smooth((200.0 - t) / 12.0))

    # Act 4 Sunset 2 -> Night (294s -> 380s)
    if 294.0 <= t < 306.0:
        return 0.0
    if 306.0 <= t < 314.0:
        return float(smooth((t - 306.0) / 8.0))
    if 314.0 <= t < 380.0:
        return 1.0
    if 380.0 <= t < 400.0:
        return float(smooth((400.0 - t) / 20.0))

    return 0.0


def _cut(t, start, hold, end):
    if t < start or t >= end:
        return 0.0
    return 1.0 if t <= hold else 1.0 - smooth((t - hold) / (end - hold))


def title_lines(t):
    """Independent smoothstep fade timings for the 6 opening title/credit layers
    relative to sequence start t0 = 8.0s (10.0s total span -> 8.0s to 18.0s)."""
    dt = t - 8.0
    if dt < 1.5 or dt > 10.0:
        return ((0.0, 0.0),) * 6

    def _win(a, b, c, d):
        if dt < a or dt > d:
            return 0.0
        if dt < b:
            return smooth((dt - a) / max(1e-4, b - a))
        if dt <= c:
            return 1.0
        return 1.0 - smooth((dt - c) / max(1e-4, d - c))

    p = _win(1.5, 2.7, 9.2, 10.0)
    she = _win(2.7, 4.0, 9.2, 10.0)
    c1 = _win(4.00, 4.75, 8.0, 9.2)
    c2 = _win(4.35, 5.10, 8.0, 9.2)
    c3 = _win(4.70, 5.45, 8.0, 9.2)
    c4 = _win(5.05, 5.80, 8.0, 9.2)
    return ((p, p), (0.0, she), (0.0, c1), (0.0, c2), (0.0, c3), (0.0, c4))


def visit(t):
    """Approach, settle on land, hold three seconds, then rise rightward and recede (-Z)."""
    if not VISIT_START <= t < VISIT_DEPARTURE_END:
        return None
    landed_height = 90.0 * (0.18 + 1.56 * VISIT_LANDING_NEAR)
    visible_bottom_offset = landed_height * (VISIT_HULL_BOTTOM_FRACTION - 0.5)
    landed_y = VISIT_LANDING_SURFACE_Y - visible_bottom_offset
    if t < VISIT_LAND_START:
        p = smooth((t - VISIT_START) / (VISIT_LAND_START - VISIT_START))
        return (lerp(-250.0, VISIT_LANDING_X, p), lerp(285.0, landed_y, p),
                lerp(0.12, VISIT_LANDING_NEAR, p), 1.0)
    if t < VISIT_LAND_END:
        return VISIT_LANDING_X, landed_y, VISIT_LANDING_NEAR, 1.0

    p = smooth((t - VISIT_LAND_END) / (VISIT_DEPARTURE_END - VISIT_LAND_END))
    # Match the reference arrow: a long diagonal ascent across the image plane.
    # `near` is the projected depth control; approaching zero makes the craft
    # shrink continuously as it travels away from camera along -Z.
    visibility = 1.0 - smooth((t - VISIT_DEPARTURE_FADE_START) /
                              (VISIT_DEPARTURE_END - VISIT_DEPARTURE_FADE_START))
    return (lerp(VISIT_LANDING_X, VISIT_DEPARTURE_X, p),
            lerp(landed_y, VISIT_DEPARTURE_Y, p),
            VISIT_LANDING_NEAR * (1.0 - p), visibility)


def arrival_cue(t):
    """A restrained water glint advances ahead of the craft, then fades at dawn."""
    if not 160.0 <= t < VISIT_DEPARTURE_END:
        return None
    if t < VISIT_LAND_START:
        p = smooth((t - 160.0) / (VISIT_LAND_START - 160.0))
        x = lerp(220.0, VISIT_LANDING_X, p)
        weight = 0.16 * smooth((t - 160.0) / 4.0)
    elif t < VISIT_LAND_END:
        x, weight = VISIT_LANDING_X, 0.16
    else:
        p = smooth((t - VISIT_LAND_END) / (VISIT_DEPARTURE_END - VISIT_LAND_END))
        x = lerp(VISIT_LANDING_X, VISIT_DEPARTURE_X, p)
        weight = 0.16 * (1.0 - smooth((t - VISIT_DEPARTURE_FADE_START) /
                                     (VISIT_DEPARTURE_END - VISIT_DEPARTURE_FADE_START)))
    return x, weight


def colony(t):
    """Spore → mushrooms → broad vegetation. The forest holds for the herd."""
    age = t - 20.0
    if age <= 0 or t > 352:
        return 0.0, 0.0, 1
    alpha = smooth(age / 8.0) * (1.0 - 0.45 * smooth((t - 300.0) / 42.0))
    stage = min(7.0, age / 18.0)
    copies = 1 if t < 38 else 3 if t < 70 else 5
    return alpha, stage, copies


def mushroom_cycle_alpha(t):
    """The mushroom world yields to pines at the visitor, then returns after decay."""
    if t < 20.0:
        return 0.0
    if t < VISIT_START:
        return colony(t)[0]
    # Mushrooms leave first; a brief, clear ridge pause separates the states
    # before the firs emerge, avoiding a long double-exposure of both silhouettes.
    if t < MUSHROOM_EXIT_END:
        return colony(VISIT_START)[0] * (1.0 - smooth((t - VISIT_START) / (MUSHROOM_EXIT_END - VISIT_START)))
    if 404.0 <= t < 420.0:
        return smooth((t - 404.0) / 16.0)
    if 420.0 <= t < 428.0:
        # Carry the returning mushrooms gently into the luminous Buddha / studio
        # transition instead of cutting their alpha at the 7-minute boundary.
        return 1.0 - smooth((t - 420.0) / 8.0)
    return 0.0


def forest_cycle(t):
    """Whorl-firs emerge with the craft and hold through the dinosaur passage."""
    if t < FIR_EMERGENCE_START or t >= 404.0:
        return 0.0, 0.0
    grow = smooth((t - FIR_EMERGENCE_START) / (FIR_EMERGENCE_END - FIR_EMERGENCE_START))
    # Ethereal fadeout of trees from 6m30s (390s) onward so Buddha takes full prominence
    decay = smooth((t - 390.0) / 14.0)
    return grow * (1.0 - decay), decay


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
    if not 232.0 <= t < 405.0:
        return ()
    
    # Calm, constant caravan pace. The longer exit window lets the full group
    # clear the frame before the forest begins its gradual decay.
    pace = 19.5
    t_rel = t - 240.0
    
    dinos = [
        (0, 2080.0 - pace * t_rel),
        (1, 2360.0 - pace * t_rel),
        (2, 2620.0 - pace * t_rel),
        (3, 2860.0 - pace * t_rel),
    ]
    
    return tuple((d_id, x) for d_id, x in dinos if -500.0 <= x <= 2320.0)


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
    # Keep the landscape readable through forest decay and mushroom return, then
    # let the cosmic Buddha resolve as their final, luminous afterimage.
    if t < 350.0:
        return 0.0
    if t < 366.0:
        return 0.18 * smooth((t - 350.0) / 16.0)
    if t < 418.0:
        return 0.18
    if t < 422.0:
        return lerp(0.18, 1.0, smooth((t - 418.0) / 4.0))
    return 1.0 - smooth((t - 422.0) / 6.0)


def audit():
    for i in range(int(DURATION * 2)):
        t = i / 2.0
        s, m = sun(t), moon(t)
        if s and m and s[1] > 0 and m[1] > 0:
            raise RuntimeError(f"sun and moon share the sky at t={t}")
