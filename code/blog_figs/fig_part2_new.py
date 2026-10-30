"""Part-2 figures (rewritten post).

  pan_inhibitory_vs_pan_neuronal.jpg   same-scale 2p slabs: 790322 (pan-inhibitory) vs 837568 (pan-neuronal)
  839909_sz_estimation.jpg             A: NCC sweep is flat (mean ± s.d. over 5 slabs); B: OLS fit of HCR depth vs 2p depth on matched pairs
"""
import json, sys, numpy as np, pandas as pd, tifffile, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/2p2fish/src')
from autocoreg.io.hcr_image import depth_from_surface

D = '/root/capsule/data/'
OUT = '/scratch/tmp/site/posts/2026-10-auto-coregistration-2-pan-neuronal/figures/'
A = D + 'HCR_839909_2026-07-23_13-00-00_2p-3DFISH-autocoreg_2026-09-03_16-08-39/'
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13.5, 'axes.labelsize': 12})
XY_UM = 0.78                                      # both stacks: 512 px, same scan settings

# ---- Fig 1: sparse vs dense, same scale ------------------------------------------------------------------
def slab(path, z0, n=11):
    v = tifffile.imread(path); return v[z0:z0 + n].max(0)
z790 = slab(D + 'multiplane-ophys_790322_2025-07-10_15-25-55_cortical-zstack-registration_2025-11-11_21-01-40/cortical_zstack_0/channel_0_ref_0/cortical_zstack_0_2xREG.tif', 200)
z837 = slab(D + 'multiplane-ophys_837568_2026-02-13_14-38-22_cortical-zstack-registration_2026-08-03_22-44-14/cortical_zstack_0/channel_0_ref_0/cortical_zstack_0_2xREG.tif', 195)
fig, ax = plt.subplots(1, 2, figsize=(10.4, 5.4))
from scipy.ndimage import binary_erosion
def norm_valid(im):                                  # normalize inside the image content only (registration leaves zero-padded edges)
    valid = binary_erosion(im > 0, iterations=4)
    lo, hi = np.percentile(im[valid], [1, 99.7]); out = np.clip((im - lo) / (hi - lo), 0, 1) ** 0.7; out[~valid] = 0; return out
for a, im, ttl in ((ax[0], z790, 'Pan-inhibitory'), (ax[1], z837, 'Pan-neuronal')):
    a.imshow(norm_valid(im), cmap='gray', interpolation='nearest')
    a.set_title(ttl, fontweight='bold'); a.set_xticks([]); a.set_yticks([])
    L = 100 / XY_UM; a.plot([20, 20 + L], [im.shape[0] - 22] * 2, color='w', lw=4); a.text(20 + L / 2, im.shape[0] - 32, '100 µm', color='w', ha='center', fontsize=11)
fig.tight_layout(); fig.savefig('/scratch/tmp/diagram/p2_fig1.png', dpi=170, bbox_inches='tight', facecolor='white')
from PIL import Image
Image.open('/scratch/tmp/diagram/p2_fig1.png').convert('RGB').save(OUT + 'pan_inhibitory_vs_pan_neuronal.jpg', quality=88)

# ---- Fig 2: s_z ----------------------------------------------------------------------------------------
sz = json.load(open(A + 'registration/sz/839909.json')); rows = sz['slab_rows']
szs = np.array(sorted({r['sz'] for r in rows})); slabs = sorted({r['slab_idx'] for r in rows})
M = np.array([[next(r['ncc'] for r in rows if r['sz'] == v and r['slab_idx'] == k) for v in szs] for k in slabs])
mean, sd = np.nanmean(M, 0), np.nanstd(M, 0)
pose = json.load(open(A + 'registration/pose/839909.json')); seed = pose['hcr_l1_thickness_um'] / pose['cz_l1_thickness_um']
cz = json.load(open(A + 'registration/surfaces/839909_cz_onset_panneuronal.json')); hs = json.load(open(A + 'registration/surfaces/839909_hcr_top_iter07.json'))
P = pd.read_csv(A + 'for_qc_app/positions.csv'); m = P[P.hcr_id > 0]
cd = depth_from_surface(m[['cz_native_x_um', 'cz_native_y_um', 'cz_native_z_um']].to_numpy(), cz)
hd = depth_from_surface(m[['hcr_x_um', 'hcr_y_um', 'hcr_z_um']].to_numpy(), hs)
k = (cd >= 30) & (hd > 0); cd, hd = cd[k], hd[k]; ols = float((cd * hd).sum() / (cd * cd).sum())
print('seed', seed, 'ncc argmax', sz['sz_best'], 'peak_ratio', sz['peak_ratio'], 'ols(final pairs)', ols, 'n', k.sum(), 'slabs', len(slabs))
np.savez('/scratch/tmp/diagram/sz_regression_839909.npz', cd=cd, hd=hd, ols=ols, seed=seed)   # regression panel is drawn in fig_l1.py (Part 2 Figure 3)
fig, a = plt.subplots(1, 1, figsize=(6.2, 4.4))
a.fill_between(szs, mean - sd, mean + sd, color='#54278f', alpha=0.22, lw=0, label='± s.d. across slabs')
a.plot(szs, mean, '-o', color='#54278f', ms=3, lw=1.6, label='mean across slabs')
a.axvline(sz['sz_best'], color='#d73027', lw=2, label=f'NCC peak = {sz["sz_best"]:.2f}')
a.axvline(2.687, color='k', lw=2, ls='--', label='adaptive estimate = 2.69')
a.set_xlabel('axial scale  s_z  (2p → HCR)'); a.set_ylabel('slab FFT-NCC'); a.set_title('NCC sweep (flat)')
a.set_ylim(0, max(mean + sd) * 1.08); a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10, frameon=False)
for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
fig.savefig('/scratch/tmp/diagram/p2_fig2.png', dpi=170, bbox_inches='tight', facecolor='white')
Image.open('/scratch/tmp/diagram/p2_fig2.png').convert('RGB').save(OUT + '839909_sz_sweep.jpg', quality=88)
