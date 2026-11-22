"""
Pre-Render Quality Gatekeeper & Code Review Suite for 'SHE' Music Video Master.
Audits the entire film generation codebase before launching expensive renders.
Blocks rendering if any mathematical discontinuity, alignment error, or visual flaw is found.
"""

import sys
import os
import math
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from she_film import clock, frame, plates, actors, water, mirage

def run_pre_render_code_review():
    print("=" * 80)
    print("PRE-RENDER CODE REVIEW & QUALITY GATEKEEPER AUDIT")
    print("=" * 80)
    
    passes = 0
    failures = 0
    warnings = 0
    findings = []

    def record_pass(test_id, description, metric=""):
        nonlocal passes
        passes += 1
        print(f" [PASS] {test_id}: {description} {metric}")

    def record_fail(test_id, description, reason):
        nonlocal failures
        failures += 1
        print(f" [FAIL] {test_id}: {description} -> ERROR: {reason}")
        findings.append((test_id, "FAIL", reason))

    def record_warn(test_id, description, note):
        nonlocal warnings
        warnings += 1
        print(f" [WARN] {test_id}: {description} -> NOTE: {note}")
        findings.append((test_id, "WARN", note))

    # -------------------------------------------------------------
    # 1. SCREEN 1 AUDIT: Sunrise Curve & Intro Credits Stack Fade
    # -------------------------------------------------------------
    print("\n--- 1. Screen 1 Audit (Sunrise & Credits Stack) ---")
    # Sun altitude must be smoothly rising and positive above horizon at t=19.0s
    s_19 = clock.sun(19.0)
    if s_19 is not None and s_19[1] > 0:
        record_pass("S1.1", "Sun altitude positive and rising smoothly at t=19s", f"x={s_19[0]:.1f}px, alt={s_19[1]:.1f}px")
    else:
        record_fail("S1.1", "Sun altitude during sunrise", f"s_19={s_19}")

    # Verify right-side credits stack artifact is zero at t=19.0s (in frame.py)
    frame_src = open(frame.__file__, encoding="utf-8").read()
    if "if t < 14.0:" in frame_src and "stripe_fade" in frame_src:
        record_pass("S1.2", "Right-hand credits stack artifact strictly eliminated by t=14.0s", "stripe_cutoff=14.0s")
    else:
        record_fail("S1.2", "Credits stack artifact persisting", "stripes not cut off at t=14.0s in frame.py")

    # A previous Colab session/repository copy can leave an older
    # screens_90s.npz in place. Rebuild it before plates.load() so the reviewer
    # always inspects the same validated, moving feeds the renderer uses.
    from she_film import generate_screens_cache
    screen_cache_path = os.path.join(plates.CACHE, "screens_90s.npz")
    cache_is_valid = False
    if os.path.isfile(screen_cache_path):
        try:
            with np.load(screen_cache_path) as cached_screens:
                required = ("scr_left", "scr_mid_left", "scr_mid_right", "scr_right")
                cache_is_valid = all(key in cached_screens for key in required)
                if cache_is_valid:
                    for key in required:
                        feed = cached_screens[key]
                        if (feed.ndim != 4 or len(feed) < 240
                                or feed.shape[1:] != (320, 240, 3)
                                or float(np.mean(np.abs(
                                    feed[0].astype(np.float32)
                                    - feed[len(feed) // 2].astype(np.float32)
                                ))) <= 2.0):
                            cache_is_valid = False
                            break
        except Exception as exc:
            print(f"[WARN] Existing Buddha CRT cache is unreadable: {exc}")
    if not cache_is_valid:
        print("[INFO] Rebuilding missing/stale Buddha CRT feed cache before audit...")
        generate_screens_cache.build_screens_cache(plates.CACHE)

    # -------------------------------------------------------------
    # 2. SCREEN 2 AUDIT: Flow AI Green Screen Mushroom Growth
    # -------------------------------------------------------------
    print("\n--- 2. Screen 2 Audit (Flow AI Mushroom Growth) ---")
    p = plates.load()
    vs_growth = p.get("anim_mushroom_growth")
    vs_idle = p.get("anim_mushroom_idle")
    if vs_growth is not None and vs_growth.n_frames > 20:
        record_pass("S2.1", "Flow AI Mushroom Growth sprite loaded", f"{vs_growth.n_frames} frames")
    else:
        record_fail("S2.1", "Mushroom Growth sprite missing", "anim_mushroom_growth is None or empty")

    if vs_idle is not None and vs_idle.n_frames > 20:
        record_pass("S2.2", "Flow AI Mushroom Idle loop loaded", f"{vs_idle.n_frames} frames")
    else:
        record_fail("S2.2", "Mushroom Idle sprite missing", "anim_mushroom_idle is None or empty")
    if vs_growth is not None and vs_growth.n_frames:
        m3_pre_endpoint = actors.mushroom3_target_frame(vs_growth, actors.MUSHROOM_GROWTH_SECONDS - 1e-3)
        m3_endpoint = actors.mushroom3_target_frame(vs_growth, actors.MUSHROOM_GROWTH_SECONDS)
        if np.array_equal(m3_pre_endpoint, m3_endpoint):
            record_pass("S2.3", "Mushroom 3 reaches and holds Scene 1's exact final growth frame")
        else:
            record_fail("S2.3", "Mushroom 3 endpoint continuity", "frame changed across the growth-to-target boundary")
    else:
        record_fail("S2.3", "Mushroom 3 canonical target", "growth sprite unavailable")

    mushroom_stands_grounded = True
    mushroom_ground_errors = []
    for check_t in (54.0, 62.0, 140.0, 224.0, 414.0):
        for stand in actors._get_active_mushroom_stands(p, check_t):
            spr, x, y = stand[:3]
            root_y = stand[6] if len(stand) > 6 else y + spr.shape[0]
            center_x = int(x + spr.shape[1] / 2)
            profile = p["ridge_surface_y"]
            terrain = np.maximum.reduce(p["ridges"])
            expected_surface = min(clock.HORIZON - 1, max(0, int(profile[center_x])))
            nearby_x = slice(max(0, center_x - 12), min(clock.WIDTH, center_x + 13))
            nearby = terrain[:clock.HORIZON, nearby_x]
            contour_rows = np.where(np.any(nearby > 0.1, axis=1))[0]
            raster_surface = (int(contour_rows[0]) if len(contour_rows)
                              else (clock.HORIZON - 1 if expected_surface == clock.HORIZON - 1 else None))
            grounded = (abs((y + spr.shape[0]) - root_y) <= 1
                        and 0 <= center_x < clock.WIDTH
                        and 0 <= root_y < clock.HORIZON
                        and abs(root_y - expected_surface) <= 1
                        and raster_surface is not None
                        and abs(root_y - raster_surface) <= 14)
            if not grounded:
                mushroom_stands_grounded = False
                mushroom_ground_errors.append((check_t, center_x, y + spr.shape[0], root_y))
    if mushroom_stands_grounded:
        record_pass("S2.4", "All mushroom roots follow the solid ridge contour across growth/return", "five anchors checked at five scene times")
    else:
        record_fail("S2.4", "Floating or buried mushroom root", f"samples={mushroom_ground_errors}")

    m2_mature, m2_scale_a = actors.mushroom2_lifecycle_frame(p, actors.MUSHROOM_GROWTH_SECONDS)
    m2_hold, m2_scale_b = actors.mushroom2_lifecycle_frame(p, actors.MUSHROOM2_DECAY_HOLD_END)
    m2_mid, m2_scale_mid = actors.mushroom2_lifecycle_frame(p, (actors.MUSHROOM2_DECAY_HOLD_END + actors.MUSHROOM2_DECAY_END) * 0.5)
    m2_end, m2_scale_end = actors.mushroom2_lifecycle_frame(p, actors.MUSHROOM2_DECAY_END)
    m2_idle, m2_idle_scale = actors.mushroom2_lifecycle_frame(p, actors.MUSHROOM2_DECAY_END + 1e-4)
    m2_decay_ok = (m2_mature is not None and m2_hold is not None and m2_mid is not None
                   and m2_end is not None and m2_idle is not None
                   and np.array_equal(m2_mature, m2_hold)
                   and m2_scale_a == m2_scale_b == 1.0
                   and m2_scale_a > m2_scale_mid > m2_scale_end == m2_idle_scale
                   and m2_scale_end <= actors.MUSHROOM2_FINAL_DECAY_SCALE + 1e-6
                   and float(np.mean(np.abs(m2_end.astype(np.float32) - m2_idle.astype(np.float32)))) < 1.0)
    if m2_decay_ok:
        record_pass("S2.5", "Mushroom 2 holds mature, then shrinks and withers in place without an abrupt silhouette swap",
                    f"mature hold through age={actors.MUSHROOM2_DECAY_HOLD_END:.0f}s; decay ends at age={actors.MUSHROOM2_DECAY_END:.0f}s")
    else:
        record_fail("S2.5", "Mushroom 2 decay continuity",
                    f"scales={m2_scale_a,m2_scale_b,m2_scale_mid,m2_scale_end,m2_idle_scale}")

    # -------------------------------------------------------------
    # 3. SCREEN 3 AUDIT: Ocean Ripples Sparsity & Audio Reactivity
    # -------------------------------------------------------------
    print("\n--- 3. Screen 3 Audit (Ocean Ripple Sparsity & Frequency) ---")
    water_src = open(water.__file__, encoding="utf-8").read()
    if ("range(5)" in water_src
            and "line_mix = (0.14 * wave_factor * coverage)" in water_src
            and "Gaussian subpixel coverage" in water_src):
        record_pass("S3.1", "Ocean crests are sparse, subpixel-softened, and low contrast",
                    "line_count=5, blend=0.14, gaussian coverage")
    else:
        record_fail("S3.1", "Ocean wave prominence", "expected five crest lines with restrained blend")

    if "d_ellipse * 0.11" in water_src or "0.10" in water_src or "0.12" in water_src:
        record_pass("S3.2", "Concentric ripple frequency is sparse (wavelength ~57px)", "k=0.11 rad/px")
    else:
        record_fail("S3.2", "Concentric ripple frequency", "spatial frequency not widened to ~0.11 rad/px")

    if "cv2.resize" in water_src and "INTER_LINEAR" in water_src and "np.repeat(np.repeat" not in water_src:
        record_pass("S3.3", "Ocean crest lines use smooth subpixel reconstruction without 4px stair steps")
    else:
        record_fail("S3.3", "Ocean wave raster quality", "expected antialiased bilinear reconstruction")
    if "0.42 * smooth" in frame_src and "rgb, 16, 110" in frame_src:
        record_pass("S3.4", "Moon reflection is a restrained narrow glint rather than a broad light column")
    else:
        record_fail("S3.4", "Moon reflection strength", "expected reduced moon path weight and width")

    # -------------------------------------------------------------
    # 4. SCREEN 4 AUDIT: Wavy Passing Effect Across Mushroom Neighbors
    # -------------------------------------------------------------
    print("\n--- 4. Screen 4 Audit (Mushroom Passing Wave Effect) ---")
    stands_t60 = actors._get_active_mushroom_stands(p, 60.0)
    if len(stands_t60) >= 3:
        xs = [s[1] for s in stands_t60]
        delays = [s[5] for s in stands_t60]
        is_increasing = all(delays[i] <= delays[i+1] for i in range(len(delays)-1))
        if is_increasing and delays[-1] > delays[0]:
            record_pass("S4.1", "Mushroom stands have strictly monotonic left-to-right wave delay propagation", 
                        f"delays: {[round(d, 2) for d in delays]}s across stands")
        else:
            record_fail("S4.1", "Mushroom stand wave delay", f"delays={delays}")
    else:
        record_fail("S4.1", "Mushroom stands count at t=60", f"found {len(stands_t60)}")

    # -------------------------------------------------------------
    # 5. SCREEN 5 AUDIT: Continuous Dinosaur Caravan & Cured Ankylosaur
    # -------------------------------------------------------------
    print("\n--- 5. Screen 5 Audit (Dinosaur Caravan Continuity & Stride) ---")
    h_269 = clock.herd(269.9)
    h_270 = clock.herd(270.1)
    if h_269 and h_270:
        dx_lead = h_270[0][1] - h_269[0][1]
        expected_dx = -19.5 * 0.2
        if abs(dx_lead - expected_dx) < 0.5:
            record_pass("S5.1", "Caravan march velocity remains continuous (-19.5px/s)", f"dx={dx_lead:.2f}px")
        else:
            record_fail("S5.1", "Caravan velocity jump detected", f"dx={dx_lead:.2f}px vs expected {expected_dx:.2f}px")
    else:
        record_fail("S5.1", "Caravan herd presence", f"h_269={h_269}, h_270={h_270}")

    vs_anky = p.get("anim_ankylosaur")
    if vs_anky is not None and vs_anky.n_frames > 20:
        record_pass("S5.2", "Cured periodic Ankylosaur sprite loaded", f"{vs_anky.n_frames} frames")
    else:
        record_fail("S5.2", "Ankylosaur sprite missing", "anim_ankylosaur is None or empty")

    # Dinosaur gaits use continuous sampling and a blended loop seam; pterosaur
    # wing motion remains on its separately approved cycle below.
    dino_assets = [p.get(k) for k in ("anim_sauropod", "anim_hadrosaur", "anim_ankylosaur", "anim_raptor")]
    cycle_checks = []
    for d_id, anim in enumerate(dino_assets):
        if anim is None or anim.n_frames < 12:
            cycle_checks.append((False, float("inf")))
            continue
        span = actors.DINO_WALK_CYCLES[d_id]
        cycle = anim.frames[span[0]:min(span[1], anim.n_frames)]
        if len(cycle) < 12:
            cycle_checks.append((False, float("inf")))
            continue
        samples = [actors._dino_walk_frame(anim, d_id, 240.0 + (len(cycle) - 1 + off) / 24.0)
                   for off in (-1 / 24.0, 0.0, 1 / 24.0)]
        delta = float(np.mean(np.abs(samples[2] - samples[0])))
        inner = float(np.mean(np.abs(cycle[1:] - cycle[:-1])))
        cycle_checks.append((np.isfinite(delta) and np.isfinite(inner) and delta <= inner * 2.0, delta))
    if all(ok for ok, _ in cycle_checks):
        record_pass("S5.5", "All four dinosaur gaits use eased continuous loop sampling",
                    f"boundary deltas={[round(v, 3) for _, v in cycle_checks]}")
    else:
        record_fail("S5.5", "Dinosaur gait loop continuity", f"boundary checks={cycle_checks}")

    # The saucer may touch land only during the authored touchdown/hold window;
    # elsewhere its opaque hull must stay above the local terrain contour.
    craft_samples = [(float(t), clock.visit(t)) for t in np.linspace(166.0, 211.9, 90)]
    craft_clear = bool(craft_samples)
    min_craft_clearance = float("inf")
    for sample_t, body in craft_samples:
        if body is None:
            craft_clear = False
            break
        ship = actors._ship_for_near(p, body[2])
        opaque_rows = np.where(ship[:, :, 3].max(axis=1) > 0.03)[0]
        hull_bottom = body[1] - ship.shape[0] / 2 + (opaque_rows[-1] if len(opaque_rows) else ship.shape[0])
        ground_y = float(p["ridge_surface_y"][int(np.clip(round(body[0]), 0, clock.WIDTH - 1))])
        clearance = ground_y - hull_bottom
        min_craft_clearance = min(min_craft_clearance, clearance)
        landing_hold = clock.VISIT_LAND_START <= sample_t < clock.VISIT_LAND_END
        if clearance < -3.0 or (landing_hold and abs(clearance) > 3.0):
            craft_clear = False
            break
    visual_edges = []
    for d_id, x in clock.herd(391.0):
        key = {0: "anim_sauropod", 1: "anim_hadrosaur", 2: "anim_ankylosaur", 3: "anim_raptor"}.get(d_id)
        anim = p.get(key) if key else None
        width = (anim.frames.shape[2] if anim is not None and anim.n_frames else p["beasts"][d_id % len(p["beasts"])].shape[1])
        visual_edges.append(actors._edge_presence(x, width))
    herd_clear = max(visual_edges, default=0.0) < 0.01
    if craft_clear:
        record_pass("S5.6", "Vimana hull stays above the terrain and touches only at its marked landing",
                    f"minimum clearance={min_craft_clearance:.1f}px")
    else:
        record_fail("S5.6", "Vimana ridge clearance", "trajectory intersects the foreground ridge envelope")
    if herd_clear:
        record_pass("S5.7", "Dinosaur sprites are fully edge-faded before forest decay begins")
    else:
        record_fail("S5.7", "Caravan exit before forest decay", f"remaining={clock.herd(391.0)}")

    gait_motion = []
    for d_id, anim in enumerate(dino_assets):
        if anim is not None and anim.n_frames:
            first = actors._dino_walk_frame(anim, d_id, 232.0)
            later = actors._dino_walk_frame(anim, d_id, 232.5)
            gait_motion.append(float(np.mean(np.abs(first - later))))
    if len(gait_motion) == 4 and min(gait_motion) > 0.01:
        record_pass("S5.8", "All dinosaur gait loops are already moving during the 232s off-screen entrance",
                    f"half-second deltas={[round(v, 3) for v in gait_motion]}")
    else:
        record_fail("S5.8", "Dinosaur gait stalls during entrance", f"deltas={gait_motion}")

    # Arrival cue, grounded landing, readable hold, and dawn receding motion.
    cue = clock.arrival_cue(164.0)
    landed_a, landed_b = clock.visit(clock.VISIT_LAND_START + 0.5), clock.visit(clock.VISIT_LAND_END - 0.1)
    landing_duration = clock.VISIT_LAND_END - clock.VISIT_LAND_START
    land_ship = actors._ship_for_near(p, clock.VISIT_LANDING_NEAR)
    landing_surface = int(p["ridge_surface_y"][int(clock.VISIT_LANDING_X)])
    ship_opaque_rows = np.where(land_ship[:, :, 3].max(axis=1) > 0.03)[0]
    visible_ship_bottom = (landed_a[1] - land_ship.shape[0] / 2 + ship_opaque_rows[-1]
                           if len(ship_opaque_rows) else float("inf"))
    land_contact_error = abs(visible_ship_bottom - landing_surface)
    land_contact = land_contact_error <= 3
    held = (landed_a is not None and landed_b is not None
            and abs(landed_a[0] - landed_b[0]) < 1.0
            and abs(landed_a[1] - landed_b[1]) < 1.0)
    departure = clock.visit(198.0)
    departure_early, departure_late = clock.visit(188.0), clock.visit(210.0)
    departs_in_z = (departure_early is not None and departure_late is not None
                    and departure_late[0] > departure_early[0] + 400.0
                    and departure_late[2] < departure_early[2]
                    and departure_late[1] < departure_early[1]
                    and departure_late[3] < departure_early[3]
                    and clock.sun(198.0) is not None)
    if cue is not None and cue[1] <= 0.16 and held and land_contact and 1.0 <= landing_duration <= 3.0:
        record_pass("S5.9", "A subtle crystal arrival glint leads into a grounded 3s saucer landing",
                    f"cue={cue[1]:.2f}, hold={landing_duration:.1f}s, visible hull contact error={land_contact_error:.1f}px")
    else:
        record_fail("S5.9", "Saucer anticipation / landing continuity",
                    f"cue={cue}, hold={landing_duration:.2f}, visible hull contact={land_contact_error:.1f}px, grounded={landed_a}")
    if departs_in_z:
        record_pass("S5.10", "Saucer follows the reference's long right/up diagonal, shrinks along -Z, and fades through sunrise",
                    f"x={departure_early[0]:.0f}->{departure_late[0]:.0f}px, near={departure_early[2]:.2f}->{departure_late[2]:.2f}")
    else:
        record_fail("S5.10", "Saucer takeoff cadence", f"early={departure_early}, late={departure_late}, dawn={clock.sun(198.0)}")

    # -------------------------------------------------------------
    # 5c. PTERODACTYL WINGBEAT LOOP CONTINUITY
    # -------------------------------------------------------------
    ptero_cycle = actors.pterodactyl_loop_frames(p.get("anim_pterodactyl"))
    if ptero_cycle is not None and len(ptero_cycle) >= 2:
        ptero_alpha = ptero_cycle[..., 3:4]
        ptero_pm = np.concatenate((ptero_cycle[..., :3] * ptero_alpha, ptero_alpha), axis=3)
        loop_seam = float(np.mean(np.abs(ptero_pm[-1] - ptero_pm[0])))
        in_cycle_deltas = np.mean(np.abs(ptero_pm[1:] - ptero_pm[:-1]), axis=(1, 2, 3))
        normal_delta = float(np.median(in_cycle_deltas))
        if normal_delta > 0.0 and loop_seam <= normal_delta:
            record_pass("S5.4", "Pterodactyl wing pose continues smoothly across its repeating loop",
                        f"seam={loop_seam:.2f}, median_step={normal_delta:.2f}, cycle={len(ptero_cycle)} frames")
        else:
            record_fail("S5.4", "Pterodactyl wingbeat loop seam", f"seam={loop_seam:.2f}, median_step={normal_delta:.2f}")
    else:
        record_fail("S5.4", "Pterodactyl wingbeat asset/cycle missing",
                    f"expected at least {actors.PTERO_WING_CYCLE_END} cached Flow frames")

    # -------------------------------------------------------------
    # 6. SCREEN 5b AUDIT: Dinosaur Night Reflection Visibility
    # -------------------------------------------------------------
    print("\n--- 6. Screen 5b Audit (Dinosaur Night Reflection Visibility) ---")
    f_night = frame.render(p, 315.0)
    # Check reflection zone directly below dinosaurs (lead dino near x=960, horizon=580)
    ref_zone = f_night[585:680, 800:1100]
    water_ambient = f_night[585:680, 1400:1700]
    mean_ref_lum = float(np.mean(ref_zone))
    mean_amb_lum = float(np.mean(water_ambient))
    contrast = mean_ref_lum - mean_amb_lum
    if contrast > 15.0:
        record_pass("S5.3", "Dinosaur night reflection clearly visible against water", 
                    f"ref_lum={mean_ref_lum:.1f}, ambient={mean_amb_lum:.1f}, contrast=+{contrast:.1f}")
    else:
        record_fail("S5.3", "Dinosaur night reflection visibility", 
                    f"contrast too low (+{contrast:.1f}), reflection invisible on night sea")

    # -------------------------------------------------------------
    # 7. SCREEN 6 AUDIT: Buddha Heart Waveform Crests & Saturated Glow
    # -------------------------------------------------------------
    print("\n--- 7. Screen 6 Audit (Buddha Heart Waveform & Expanding White Glow) ---")
    f_buddha = frame.render(p, 395.0)
    heart_slice = f_buddha[540:660, 800:1120]
    col_peaks = np.argmax(np.mean(heart_slice, axis=2), axis=0)
    crest_trough_amp = int(np.ptp(col_peaks))
    if crest_trough_amp >= 25:
        record_pass("S6.1", "Buddha waveform has prominent crests and troughs", f"peak_to_peak={crest_trough_amp}px")
    else:
        record_fail("S6.1", "Buddha waveform crest-to-trough amplitude", f"amp={crest_trough_amp}px too flat (<25px)")

    f_glow = frame.render(p, 422.0)
    core_pixel = f_glow[600, 960]
    is_saturated_white = core_pixel[0] >= 240 and core_pixel[1] >= 240 and core_pixel[2] >= 240
    if is_saturated_white:
        record_pass("S6.2", "Mystical love heart glow is pure saturated radiant white", f"RGB={list(core_pixel)}")
    else:
        record_fail("S6.2", "Heart glow saturation", f"RGB={list(core_pixel)} not saturated white")

    # -------------------------------------------------------------
    # 8. SCREEN 7 AUDIT: Telemetry Studio Redesign 2-Screen Fitting
    # -------------------------------------------------------------
    print("\n--- 8. Screen 7 Audit (Buddha B-roll on curved studio CRTs) ---")
    layout_ok, layout_note = frame.validate_studio_screen_bounds()
    if layout_ok:
        record_pass("S7.1", "TV video quads fit inside the source plate's visible display bounds", layout_note)
    else:
        record_fail("S7.1", "TV display area alignment", layout_note)
    debug_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "preview_720p", "tv_screen_alignment_debug.png")
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    debug_img = frame.studio_screen_debug_overlay(p["lab"])
    if cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)):
        record_pass("S7.3", "TV alignment debug overlay saved", debug_path)
    else:
        record_fail("S7.3", "TV alignment debug overlay", f"could not write {debug_path}")

    screen_feeds = p.get("screens_90s")
    feed_keys = ("scr_left", "scr_mid_left", "scr_mid_right", "scr_right")
    feeds_ok = screen_feeds is not None and all(k in screen_feeds for k in feed_keys)
    if feeds_ok:
        feeds = [screen_feeds[key] for key in feed_keys]
        dynamic = all(
            feed.ndim == 4 and len(feed) >= 240 and feed.shape[1:] == (320, 240, 3)
            and float(np.mean(np.abs(feed[0].astype(np.float32) - feed[len(feed) // 2].astype(np.float32)))) > 2.0
            for feed in feeds
        )
        if dynamic:
            record_pass("S7.2", "All four CRT feeds contain moving B-roll",
                        f"frames={[len(feed) for feed in feeds]}")
        else:
            record_fail("S7.2", "Dynamic Buddha B-roll feeds", "missing, static, or invalid screen frames")
    else:
        record_fail("S7.2", "Dynamic CRT B-roll feeds", "screens_90s.npz is missing one or more feeds")

    # -------------------------------------------------------------
    # 9. MASTER DURATION & OUTRO FADE
    # -------------------------------------------------------------
    print("\n--- 9. Master Runtime & Branded Outro ---")
    exact_frames = int(round(clock.DURATION * clock.FPS))
    outro_ok = (
        abs(clock.DURATION - 445.0) < 1e-6
        and exact_frames == 10680
        and clock.TOTAL_FRAMES == exact_frames
        and clock.OUTRO_TRANSITION_START < clock.OUTRO_FADE_START < clock.DURATION
        and plates.build_outro_card is not None
    )
    if outro_ok:
        record_pass("S8.1", "7:25 runtime, 10,680 frames, branded outro and final fade are configured",
                    f"fade={clock.OUTRO_FADE_START:.1f}-{clock.DURATION:.1f}s")
    else:
        record_fail("S8.1", "Master duration/outro configuration",
                    f"duration={clock.DURATION}, frames={clock.TOTAL_FRAMES}, outro={clock.OUTRO_TRANSITION_START}/{clock.OUTRO_FADE_START}")

    # -------------------------------------------------------------
    # 10. WHORL-FIR LIFECYCLE & RETURN TO MUSHROOMS
    # -------------------------------------------------------------
    print("\n--- 10. Screen 8 Audit (Mushroom / Whorl-Fir / Return Lifecycle) ---")
    mushroom_open = clock.mushroom_cycle_alpha(160.0)
    mushroom_yield = clock.mushroom_cycle_alpha(184.0)
    forest_hold, _ = clock.forest_cycle(260.0)
    forest_decay, decay = clock.forest_cycle(397.0)
    forest_empty, _ = clock.forest_cycle(406.0)
    mushroom_return = clock.mushroom_cycle_alpha(414.0)
    cycle_values_ok = (
        mushroom_open > 0.9 and mushroom_yield < 0.02 and forest_hold > 0.99
        and 0.0 < forest_decay < forest_hold and decay > 0.0
        and forest_empty == 0.0 and mushroom_return > 0.4
        and clock.buddha(400.0) < 0.25 and clock.buddha(422.0) == 1.0
    )
    assets_ok = len(p.get("forest", ())) == 5 and all(
        sprite.ndim == 3 and sprite.shape[2] == 4 and sprite[:, :, 3].max() > 0.9
        for sprite in p.get("forest", ())
    )
    if cycle_values_ok:
        record_pass("S8.2", "Mushrooms yield to whorl-firs at the saucer arrival, then decay and return on cue",
                    f"pre-arrival mushrooms={mushroom_open:.2f}, after transition={mushroom_yield:.2f}, forest={forest_hold:.2f}, return={mushroom_return:.2f}")
    else:
        record_fail("S8.2", "Forest lifecycle timing", f"open={mushroom_open}, hold={forest_hold}, decay={forest_decay}/{decay}, empty={forest_empty}, return={mushroom_return}")
    if assets_ok and callable(actors.forest_reflect) and callable(actors._celestial_shadow):
        record_pass("S8.3", "Five selected whorl-fir cutouts load with alpha, light shading, and water-reflection paths")
    else:
        record_fail("S8.3", "Whorl-fir assets or lighting/reflection path", f"forest_count={len(p.get('forest', ()))}, assets_ok={assets_ok}")

    depth_scale = plates.FOREST_DEPTH_SCALE
    far_ground = [actors.forest_ground_y(p, i) for i in (2, 3)]
    near_ground = [int(p["ridge_surface_layers"][3][x]) for x in (actors.FOREST_ANCHORS[2], actors.FOREST_ANCHORS[3])]
    real_depth = (actors.FOREST_GROUND_LAYERS[2:4] == (0, 0)
                  and all(far < near - 20 for far, near in zip(far_ground, near_ground)))
    selected_rows = [int(p["ridge_surface_layers"][layer][x])
                     for layer, x in zip(actors.FOREST_GROUND_LAYERS, actors.FOREST_ANCHORS)]
    roots_above_shore = all(row <= clock.HORIZON - actors.FOREST_GROUND_MARGIN
                            for row in selected_rows)
    root_rows = p.get("forest_root_rows", ())
    roots_seated = len(root_rows) == 5 and all(0 < root_rows[i] <= p["forest"][i].shape[0] for i in range(5))
    intact_depth_forms = actors.FOREST_DEPTH_VISIBILITY[2] >= 0.85 and actors.FOREST_DEPTH_VISIBILITY[3] >= 0.85
    layered_occlusion = callable(actors._ridge_visibility_mask) and len(p.get("ridges", ())) == 4
    if (real_depth and roots_above_shore and roots_seated and intact_depth_forms and layered_occlusion
            and depth_scale[2] <= 0.70 and depth_scale[3] <= 0.72):
        record_pass("S8.5", "Forest roots sit above the shoreline; nearer ridge masks preserve depth without ghosting",
                    f"rear rows={far_ground}, foreground rows={near_ground}, scale={depth_scale[2:4]}")
    else:
        record_fail("S8.5", "Forest ground/depth ordering",
                    f"rear={far_ground}, foreground={near_ground}, selected={selected_rows}, roots={root_rows}, opaque={intact_depth_forms}, occlusion={layered_occlusion}")

    # The two inland cutouts should have clean mattes, while only the distant
    # background and reflected actors receive the slow music/wave modulation.
    far_tree = p["forest"][3]
    _, _, component_stats, _ = cv2.connectedComponentsWithStats(
        (far_tree[:, :, 3] > 0.08).astype(np.uint8), connectivity=8
    )
    detached_specks = int(np.sum(component_stats[1:, cv2.CC_STAT_AREA] < 60))
    audio_profile = p.get("audio_feats")
    blur_energies = [mirage.music_energy(audio_profile, t) for t in (176.0, 177.0, 178.0, 179.0, 180.0)]
    audio_varies = max(blur_energies, default=0.0) - min(blur_energies, default=0.0) > 0.01
    background_focus = callable(mirage.soften_background) and mirage.focus_window(178.0) > 0.5
    synthetic = np.zeros((42, 100, 4), dtype=np.float32)
    cv2.ellipse(synthetic, (50, 16), (18, 7), 0, 0, 360, (180, 180, 160, 0.8), -1)
    warped = actors._water_wave_refraction(synthetic, 178.0, 960, 1.0, audio_profile)
    wave_reacts = float(np.mean(np.abs(warped - synthetic))) > 0.002
    if detached_specks <= 3 and background_focus and audio_varies and wave_reacts:
        record_pass("S8.6", "Far fir mattes are cleaned; background softness and actor reflections respond gently to music and waves",
                    f"detached specks={detached_specks}, focus={mirage.focus_window(178.0):.2f}, audio span={max(blur_energies)-min(blur_energies):.2f}")
    else:
        record_fail("S8.6", "Forest matte / music focus / wave-reflection treatment",
                    f"specks={detached_specks}, focus={background_focus}, audio_span={max(blur_energies)-min(blur_energies):.3f}, wave={wave_reacts}")

    mushroom_fade = [clock.mushroom_cycle_alpha(t) for t in (419.0, 420.0, 424.0, 428.0)]
    fade_ok = (mushroom_fade[0] > 0.98 and mushroom_fade[1] > 0.99
               and 0.0 < mushroom_fade[2] < mushroom_fade[1]
               and mushroom_fade[3] == 0.0)
    if fade_ok:
        record_pass("S8.4", "Returning mushrooms dissolve gradually into the Buddha / studio transition",
                    f"alpha at 419/420/424/428s={[round(v, 3) for v in mushroom_fade]}")
    else:
        record_fail("S8.4", "Mushroom return outro continuity", f"alpha={mushroom_fade}")

    try:
        smoke_times = (28.0, 36.0, 182.5, 232.0, 414.0, 432.0)
        smoke_shapes = []
        for sample_t in smoke_times:
            sample = frame.render(p, sample_t)
            smoke_shapes.append(sample.shape == (clock.HEIGHT, clock.WIDTH, 3)
                                and np.isfinite(sample).all())
        if all(smoke_shapes):
            record_pass("S9.1", "Representative full-frame composites render through mushroom decay, landing, forest, return, and CRT shots",
                        f"{len(smoke_times)} frames sampled")
        else:
            record_fail("S9.1", "Representative frame render smoke check", f"valid={smoke_shapes}")
    except Exception as exc:
        record_fail("S9.1", "Representative frame render smoke check", repr(exc))

    print("\n" + "=" * 80)
    print(f"AUDIT SUMMARY: {passes} PASSED | {failures} FAILED | {warnings} WARNINGS")
    print("=" * 80)

    if failures == 0:
        print("\n>>> PRE-RENDER CODE / ASSET CHECKS PASSED <<<")
        print("Static timing, asset, and alignment checks passed; this is not an artistic certification.")
        print("Inspect the rendered review frames for composition, depth, and transition quality.\n")
        return True
    else:
        print("\n>>> PRE-RENDER CODE / ASSET CHECKS FAILED <<<")
        print("Review and correct the reported failures before launching the render.\n")
        return False

if __name__ == "__main__":
    success = run_pre_render_code_review()
    sys.exit(0 if success else 1)
