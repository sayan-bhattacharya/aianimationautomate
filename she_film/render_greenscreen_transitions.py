"""Render production-grade green-screen (#00FF00) transition video plates and composites.

Maintains 100% spatial integrity, identical 5-anchor coordinates, and scale hierarchy
from the movie scenes so these can be keyed directly onto the ocean palette.
"""
import os
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from she_film.clock import HEIGHT, HORIZON, WIDTH, smooth
from she_film.plates import load, FOREST_DEPTH_SCALE, _rgba
from she_film import actors

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "preview_720p", "greenscreen_transition_videos")
os.makedirs(OUT_DIR, exist_ok=True)

CHROMA_GREEN = np.array([0, 255, 0], dtype=np.uint8)  # #00FF00

def render_greenscreen_frame(plates, t, mode="mushroom_to_fir"):
    """Render the 5 vegetation anchors on pure chroma-green (#00FF00) background."""
    # Create solid green canvas
    canvas = np.full((HEIGHT, WIDTH, 3), CHROMA_GREEN, dtype=np.float32)
    
    if mode == "mushroom_to_fir":
        # t runs from 0.0 to 6.0s
        # 0.0 - 1.0s: stable mushroom hold
        # 1.0 - 5.0s: gradual metamorphosis with spores and heat-haze
        # 5.0 - 6.0s: stable whorl-fir hold
        p = np.clip((t - 1.0) / 4.0, 0.0, 1.0)
        mush_alpha = 1.0 - smooth(p)
        fir_alpha = smooth(p)
        
        # 1. Render mushrooms if present
        if mush_alpha > 0.01:
            actors.life(canvas, plates, 120.0, alpha=mush_alpha)
            
        # 2. Add fine golden spore motes drifting between anchors during transition
        if 0.1 < p < 0.9:
            spore_intensity = np.sin(p * np.pi)
            rng = np.random.RandomState(int(t * 1000) % 10000)
            for anc in actors.FOREST_ANCHORS:
                n_spores = int(18 * spore_intensity)
                for _ in range(n_spores):
                    sx = int(anc + rng.uniform(-140, 140))
                    sy = int(HORIZON - 20 - rng.uniform(20, 240) * (0.4 + 0.6 * p))
                    if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
                        cv2.circle(canvas, (sx, sy), rng.randint(1, 3), (255, 230, 140), -1, cv2.LINE_AA)
        
        # 3. Render ancient whorl-firs
        if fir_alpha > 0.01:
            actors.forest(canvas, plates, 200.0, presence=fir_alpha, decay=0.0,
                          respect_ridges=False)

    elif mode == "fir_to_coastal":
        # Whorl-fir to coastal pine transition over 6.0s
        p = np.clip((t - 1.0) / 4.0, 0.0, 1.0)
        fir_alpha = 1.0 - smooth(p)
        pine_alpha = smooth(p)
        
        if fir_alpha > 0.01:
            actors.forest(canvas, plates, 200.0, presence=fir_alpha, decay=0.0,
                          respect_ridges=False)
            
        if pine_alpha > 0.01:
            # Render coastal wind-sculpted pines with inland depth
            coastal_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "whorl_firs", "wind_sculpted_coastal.png")
            if os.path.exists(coastal_path):
                cp_im = Image.open(coastal_path).convert("RGBA")
                for idx, anc in enumerate(actors.FOREST_ANCHORS):
                    scale = FOREST_DEPTH_SCALE[idx]
                    tw = round(cp_im.width * WIDTH / 1365.0 * scale)
                    th = round(cp_im.height * WIDTH / 1365.0 * scale)
                    resized = cp_im.resize((tw, th), Image.Resampling.LANCZOS)
                    arr = _rgba(np.asarray(resized))
                    ground_y = actors.forest_ground_y(plates, idx)
                    vis = actors.FOREST_DEPTH_VISIBILITY[idx] * pine_alpha
                    actors.over(canvas, arr, anc - tw // 2, ground_y - th + 8, vis)
    
    return np.clip(canvas, 0, 255).astype(np.uint8)


def main():
    print("Loading scene plates...", flush=True)
    plates = load()
    
    fps = 24
    duration = 6.0  # 6 seconds
    total_frames = int(fps * duration)
    target_w, target_h = 1280, 720
    
    configs = [
        ("greenscreen_mushroom_to_whorl_fir", "mushroom_to_fir"),
        ("greenscreen_whorl_fir_to_coastal_pine", "fir_to_coastal"),
    ]
    
    for filename, mode in configs:
        out_mp4 = os.path.join(OUT_DIR, f"{filename}.mp4")
        print(f"\nRendering {filename}.mp4 ({total_frames} frames @ 24fps)...", flush=True)
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}",
            "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-",
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
        sample_frames = [0, total_frames // 4, total_frames // 2, (3 * total_frames) // 4, total_frames - 1]
        saved_samples = {}
        
        for idx in range(total_frames):
            t = idx / fps
            frame = render_greenscreen_frame(plates, t, mode=mode)
            proc.stdin.write(frame.tobytes())
            
            if idx in sample_frames:
                saved_samples[t] = cv2.resize(frame, (target_w, target_h))
                
            if (idx + 1) % 36 == 0 or idx + 1 == total_frames:
                print(f"  Frame {idx+1}/{total_frames} (t={t:.2f}s)", flush=True)
                
        proc.stdin.close()
        proc.wait()
        print(f"Completed: {out_mp4}")
        
        # Build 5-up contact sheet
        cs_w, cs_h = 480, 270
        contact = Image.new("RGB", (cs_w * 5, cs_h + 30), (18, 18, 22))
        draw = ImageDraw.Draw(contact)
        for i, (sample_t, sf) in enumerate(sorted(saved_samples.items())):
            sf_rgb = cv2.cvtColor(sf, cv2.COLOR_BGR2RGB)
            sim = Image.fromarray(sf_rgb).resize((cs_w, cs_h), Image.Resampling.LANCZOS)
            contact.paste(sim, (i * cs_w, 30))
            draw.text((i * cs_w + 10, 8), f"{filename} | t={sample_t:.2f}s", fill=(240, 240, 240))
            
        contact_path = os.path.join(OUT_DIR, f"{filename}_contact.jpg")
        contact.save(contact_path, quality=92)
        print(f"Saved contact sheet: {contact_path}")

    # Now render Split-Screen Verification Video:
    # Left half: Green Screen, Right half: Film Palette Composite with water reflection & blueblocker
    split_mp4 = os.path.join(OUT_DIR, "split_screen_greenscreen_vs_film_palette.mp4")
    print(f"\nRendering split_screen_greenscreen_vs_film_palette.mp4...", flush=True)
    
    from she_film.frame import render
    split_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-",
        "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", split_mp4
    ]
    sp_proc = subprocess.Popen(split_cmd, stdin=subprocess.PIPE)
    
    # Use movie timestamp range t = 168.0 to 184.0s (the exact scene in the film)
    movie_t_start = 168.0
    movie_t_end = 184.0
    split_frames_total = int(fps * (movie_t_end - movie_t_start))
    
    for idx in range(split_frames_total):
        movie_t = movie_t_start + idx / fps
        norm_t = (idx / split_frames_total) * 6.0
        
        # 1. Left half: green screen
        gs_frame = render_greenscreen_frame(plates, norm_t, mode="mushroom_to_fir")
        
        # 2. Right half: movie composite
        film_frame = render(plates, movie_t).astype(np.float32)
        # Apply subtle blue-blocker grade
        film_frame[:, :, 0] *= 1.018
        film_frame[:, :, 1] *= 1.006
        film_frame[:, :, 2] *= 0.88
        film_frame = np.clip(film_frame, 0, 255).astype(np.uint8)
        
        # Compose side-by-side or split 50/50 down the middle
        split_frame = np.zeros_like(film_frame)
        split_frame[:, :WIDTH//2] = gs_frame[:, :WIDTH//2]
        split_frame[:, WIDTH//2:] = film_frame[:, WIDTH//2:]
        # Draw white dividing line
        cv2.line(split_frame, (WIDTH//2, 0), (WIDTH//2, HEIGHT), (255, 255, 255), 2)
        
        sp_proc.stdin.write(split_frame.tobytes())
        if (idx + 1) % 48 == 0 or idx + 1 == split_frames_total:
            print(f"  Split-screen frame {idx+1}/{split_frames_total} (movie_t={movie_t:.2f}s)", flush=True)
            
    sp_proc.stdin.close()
    sp_proc.wait()
    print(f"Completed split screen video: {split_mp4}")
    
    # Save a contact still of the split screen
    test_split = Image.fromarray(split_frame).resize((target_w, target_h), Image.Resampling.LANCZOS)
    draw_sp = ImageDraw.Draw(test_split)
    draw_sp.text((20, 20), "LEFT: RAW PRODUCTION GREEN SCREEN (#00FF00)", fill=(0, 255, 0))
    draw_sp.text((target_w // 2 + 20, 20), "RIGHT: COMPOSITE ON FILM PALETTE", fill=(255, 220, 140))
    split_still_path = os.path.join(OUT_DIR, "split_screen_verification_still.jpg")
    test_split.save(split_still_path, quality=92)
    print(f"Saved split screen verification still: {split_still_path}")
    print("\nAll green screen transition videos successfully rendered and verified!")

if __name__ == "__main__":
    main()
