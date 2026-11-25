# SHE Full-Song Film Render Pipeline

SHE is the ambient music-film project in this repository. It is separate from the NOIR project and does not share its code, assets, or render workflow.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sayan-bhattacharya/aianimationautomate/blob/main/SHE_Full_Song_Master_Render_Colab.ipynb)

## Current render target

- Runtime: **7:25**, 10,680 frames at 24 fps
- Master: **1920?1080**, H.264 video with 320 kb/s AAC audio
- Render flow: 10 sequential chunks, with Google Drive checkpoints when Drive is mounted
- Teaser: 50 seconds, 1920?1080 landscape at 30 fps
- Current handoff: [revision notes](SHE_RENDER_REVISION_NOTES.md), [end-of-day critical review](SHE_END_OF_DAY_REVIEW_2026-11-25.md)

The notebook clones SHE code, assets, audio, and audio features from this repository's `main` branch. Colab resource availability and execution time vary by account/runtime. The renderer uses CPU-side frame generation and ffmpeg encoding; having a GPU visible does not mean the render is GPU-accelerated.

## Visual revision represented in the current notebook

The late Buddha/studio passage contains four moving CRT feeds. The first/left feed shows a painterly dinosaur aerial; the middle feeds show related dinosaur views; the far-right feed uses subdued analog noise. The 7:05 review point checks that arrangement, and the closing credits include `.STORE` around 7:21. Tree motion is slow and restrained; the dinosaur procession remains distant in the landscape.

## Automated checks and their limits

The pre-render gate passed 34 code/asset checks for the current revision, including mushroom grounding and timing, water restraint, dinosaur continuity, four moving CRT feeds, and the 7:25 runtime. This is a pre-render code gate, not a visual quality certificate. The notebook's final output check covers frame count, duration, resolution, and audio stream presence; it does not yet certify sync, freezes, black frames, loudness, or artistic quality. Inspect and listen to the rendered master before release.

Final video files are generated in Colab/Drive and are not stored in Git.
