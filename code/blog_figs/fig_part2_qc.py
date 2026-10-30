"""Part-2 results figure: 2p ROIs by match status (two subjects). Rows: 837568, 839909. Columns: XY, XZ, YZ.
Legend of the first axis sits below it; no figure title. Also writes the 837568 sweep movie."""
import numpy as np, pandas as pd, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio

D = '/root/capsule/data/'
OUT = '/scratch/tmp/site/posts/2026-10-auto-coregistration-2-pan-neuronal/figures/'
SUBJ = [('837568', D + 'HCR_837568-01_2026-05-20_13-00-00_2p-3DFISH-autocoreg_2026-09-03_16-46-13/'),
        ('839909', D + 'HCR_839909_2026-07-23_13-00-00_2p-3DFISH-autocoreg_2026-09-03_16-08-39/')]
FS = 12
plt.rcParams.update({'font.size': FS, 'axes.titlesize': FS + 1, 'axes.labelsize': FS})
fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2), gridspec_kw=dict(width_ratios=[1, 1.3, 1.3], wspace=0.28, hspace=0.62))
for r, (sid, A) in enumerate(SUBJ):
    P = pd.read_csv(A + 'for_qc_app/positions.csv'); mt = P.hcr_id > 0
    print(sid, len(P), int(mt.sum()))
    for c, (u, v, xl, yl, inv) in enumerate([('cz_native_x_um', 'cz_native_y_um', 'x (µm)', 'y (µm)', False),
                                             ('cz_native_x_um', 'cz_native_z_um', 'x (µm)', 'z (µm)', True),
                                             ('cz_native_y_um', 'cz_native_z_um', 'y (µm)', 'z (µm)', True)]):
        a = axes[r, c]
        a.scatter(P.loc[~mt, u], P.loc[~mt, v], s=7, marker='x', c='#d400d4', linewidths=0.6, alpha=0.6, label=f'unmatched ({(~mt).sum():,})')
        a.scatter(P.loc[mt, u], P.loc[mt, v], s=7, c='#2ca02c', alpha=0.55, linewidths=0, label=f'matched ({mt.sum():,})')
        a.set_xlabel(xl); a.set_ylabel(yl)
        if inv: a.invert_yaxis()
        else: a.set_aspect('equal')
        a.set_title(['XY', 'XZ', 'YZ'][c] + (f'   ·   {sid}' if c == 0 else ''), loc='left' if c == 0 else 'center')
    axes[r, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=FS - 1, markerscale=2.2, frameon=False)
fig.savefig('/scratch/tmp/diagram/p2_results.png', dpi=130, bbox_inches='tight', facecolor='white')
Image.open('/scratch/tmp/diagram/p2_results.png').convert('RGB').save(OUT + 'results_two_subjects.jpg', quality=88)

# second movie: 837568 top-down ROI sweep (original frames, 2x, original speed)
fr = sorted(glob.glob(SUBJ[0][1] + 'qc/08_sweep_rois_topdown_frames/*.png')); print('frames', len(fr))
with imageio.get_writer(OUT + '837568_sweep_rois_topdown.mp4', fps=5, codec='libx264', macro_block_size=None, pixelformat='yuv420p',
                        output_params=['-crf', '27', '-preset', 'slow', '-movflags', '+faststart']) as w:
    for f in fr:
        im = Image.open(f).convert('RGB'); w.append_data(np.asarray(im.resize((im.width * 2 // 2 * 2, im.height * 2 // 2 * 2), Image.NEAREST)))
