"""Re-render one session-16 movie with blog wording ('CZ' -> '2p' inside string literals only, American spelling).
Usage: MOVIE=mov_auto_4_matcher python regen_movie.py   (archive is read-only; output -> /scratch/tmp/diagram/movies)"""
import sys, os, io, tokenize
from pathlib import Path
ARCH = Path('/root/capsule/data/claude-data_ophys-mfish-autocoreg_soma-print_260616/sessions')
S16 = ARCH / '16_intermediate_summary_materials'
for p in (str(S16), str(ARCH / '15_geom_features')): sys.path.insert(0, p)
import matplotlib; matplotlib.use('Agg')
import _common
import benchmark_data_loader as bdl
OUT = Path('/scratch/tmp/diagram/movies'); OUT.mkdir(parents=True, exist_ok=True)
_common.OUT_MOV = OUT
_common.S15 = ARCH / '15_geom_features'; _common.S11 = ARCH / 'v3_S11_roi_quality'
def _hcr_dir(sid):
    m = sorted(x for x in bdl.DATA_DIR.glob(f"HCR_{sid}_*_processed_*") if 'HCR-ROI-label' not in x.name)
    m = [x for x in m if (x / 'cell_body_segmentation' / 'cell_centroids.npy').exists()] or m
    return m[-1]
bdl._find_hcr_dir = _hcr_dir

def save_movie(frames, stem, fps=20):                 # GIF only (no ffmpeg in this env); same subsampling/size as the original
    import numpy as np, imageio.v2 as imageio
    from PIL import Image
    frames = [np.asarray(f) for f in frames]
    step = max(1, len(frames) // 60); gframes = []
    for f in frames[::step]:
        im = Image.fromarray(f)
        if im.width > 480: im = im.resize((480, round(im.height * 480 / im.width)), Image.BILINEAR)
        gframes.append(np.asarray(im))
    imageio.mimsave(OUT / f"{stem}.gif", gframes, duration=step / fps, loop=0)
    h, w = frames[0].shape[:2]                       # full-resolution MP4 (h264, yuv420p) for in-page playback with controls
    if h % 2 or w % 2:
        H, W = h + h % 2, w + w % 2
        frames = [np.pad(f, ((0, H - h), (0, W - w), (0, 0)), mode="edge") for f in frames]
    with imageio.get_writer(OUT / f"{stem}.mp4", fps=fps, codec="libx264", macro_block_size=None, pixelformat="yuv420p",
                            output_params=["-crf", "24", "-preset", "slow", "-movflags", "+faststart"]) as wr:
        for f in frames: wr.append_data(f)
    print(f"  saved {OUT / (stem + '.gif')}  ({len(frames)} frames)")
_common.save_movie = save_movie

src = (S16 / 'mov_grand_registration.py').read_text().replace('/root/capsule/code/sessions/', str(ARCH) + '/')
out = []; last = (1, 0); lines = src.splitlines(keepends=True)
toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
res = []
for tok in toks:
    s = tok.string
    if tok.type == tokenize.STRING and not s.startswith(('"""', "'''")):
        s = s.replace('2p GCaMP (CZ)', '2p z-stack').replace('CZ', '2p').replace('neighbour', 'neighbor').replace('colour', 'color')
    res.append((tok.type, s))
new = tokenize.untokenize(res)
g = {'__name__': '__main__', '__file__': str(S16 / 'mov_grand_registration.py')}
exec(compile(new, str(S16 / 'mov_grand_registration.py'), 'exec'), g)
