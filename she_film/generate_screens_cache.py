"""
Generates the 2-screen sequential video cache for the ending telemetry studio scene:
  - Screen A (Left Wall): Between The Notes Logo (3s) -> Dinosaur Caravan March (19s)
  - Screen B (Right Wall): Cheetah (5s) -> Romcom (5s) -> Commercial (6s) -> White Noise (6s)
"""
import os
import cv2
import numpy as np

def build_screens_cache(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(output_dir, exist_ok=True)
    out_npz = os.path.join(output_dir, "screens_90s.npz")
    
    film_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(film_dir)
    screens_assets_dir = os.path.join(root_dir, "assets", "screens")
    
    # 1. Screen A: Logo (3s = 72 frames) -> Dino (19s = 456 frames)
    logo_path = os.path.join(screens_assets_dir, "logo_intro_3s.mp4")
    if not os.path.exists(logo_path):
        logo_path = os.path.join(root_dir, "logo_intro_3s.mp4")
        
    dino_path = os.path.join(screens_assets_dir, "dino_clean_19s.mp4")
    if not os.path.exists(dino_path):
        dino_path = os.path.join(root_dir, "preview_720p", "dinosaur_caravan_continuous_review_720p.mp4")
    
    logo_frames = []
    if os.path.exists(logo_path):
        cap = cv2.VideoCapture(logo_path)
        while len(logo_frames) < 72:
            ret, fr = cap.read()
            if not ret or fr is None:
                break
            logo_frames.append(cv2.resize(fr, (320, 240)))
        cap.release()
    else:
        logo_sprite = os.path.join(root_dir, "assets", "logo-circle.png")
        base = cv2.imread(logo_sprite) if os.path.exists(logo_sprite) else np.zeros((240, 320, 3), np.uint8)
        base = cv2.resize(base, (320, 240))
        logo_frames = [base.copy() for _ in range(72)]
        
    dino_frames = []
    if os.path.exists(dino_path):
        cap = cv2.VideoCapture(dino_path)
        # If using the clean 19s clip, read from frame 0; if using review 720p, offset by 240
        if "dino_clean" in dino_path:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 240)
        while len(dino_frames) < 456:
            ret, fr = cap.read()
            if not ret or fr is None:
                break
            dino_frames.append(cv2.resize(fr, (320, 240)))
        cap.release()
    else:
        dino_frames = [np.full((240, 320, 3), [140, 160, 180], np.uint8) for _ in range(456)]
        
    scr_a = (logo_frames + dino_frames)[:528]
    while len(scr_a) < 528:
        scr_a.append(scr_a[-1])
        
    # 2. Screen B: Cheetah (5s = 120) -> Romcom (5s = 120) -> Commercial (6s = 144) -> Noise (6s = 144)
    c1_path = os.path.join(screens_assets_dir, "clip-01.mp4")
    c2_path = os.path.join(screens_assets_dir, "clip-02.mp4")
    c3_path = os.path.join(screens_assets_dir, "clip-03.mp4")
    
    # Fallback to local flow-mcp-out if assets/screens does not have them yet
    flow_dir = r"C:\Users\sayan\flow-mcp-out\StudioScreens90s"
    if not os.path.exists(c1_path):
        c1_path = os.path.join(flow_dir, "clip-01.mp4")
    if not os.path.exists(c2_path):
        c2_path = os.path.join(flow_dir, "clip-02.mp4")
    if not os.path.exists(c3_path):
        c3_path = os.path.join(flow_dir, "clip-03.mp4")
    
    def read_frames(p, count, color):
        res = []
        if os.path.exists(p):
            cap = cv2.VideoCapture(p)
            for _ in range(count):
                ret, fr = cap.read()
                if not ret or fr is None:
                    break
                res.append(cv2.resize(fr, (320, 240)))
            cap.release()
        while len(res) < count:
            res.append(np.full((240, 320, 3), color, np.uint8))
        return res
        
    cheetah = read_frames(c1_path, 120, [45, 65, 30])
    romcom = read_frames(c2_path, 120, [35, 32, 45])
    comm = read_frames(c3_path, 144, [60, 48, 25])
    
    noise = []
    for i in range(144):
        static = np.random.randint(40, 220, (240, 320, 3), dtype=np.uint8)
        bar_y = (i * 12) % 240
        static[bar_y:min(240, bar_y + 18), :] = np.clip(static[bar_y:min(240, bar_y + 18), :].astype(int) + 50, 0, 255).astype(np.uint8)
        static[::2, :] = (static[::2, :].astype(float) * 0.82).astype(np.uint8)
        noise.append(static)
        
    scr_b = cheetah + romcom + comm + noise
    
    arr_a = np.array(scr_a, dtype=np.uint8)
    arr_b = np.array(scr_b, dtype=np.uint8)
    
    np.savez_compressed(out_npz, scr_left=arr_a, scr_right=arr_b, scr1=arr_a, scr2=arr_b, scr3=arr_a, scr4=arr_b)
    print(f"[CACHE] Built screens_90s.npz at {out_npz} ({os.path.getsize(out_npz)/1024/1024:.2f} MB)")
    return out_npz

if __name__ == "__main__":
    build_screens_cache()
