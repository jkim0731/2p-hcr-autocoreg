"""Ground-truth transform summary for the blog (Figure 3): former panels A, C, E, F, G of session-16 fig_01, renumbered A-E.

  A  anisotropic expansion (XY vs Z)          [fig_01 A]
  B  pia tilt per modality                    [fig_01 C]
  C  in-plane rotation, deviation from 180°   [fig_01 E]
  D  2p center vs HCR FOV center              [fig_01 F]
  E  nonrigid residual after global affine    [fig_01 G / fig_02 A]
Data helpers (_gt.all_stats, fov_center_offset) come from the read-only archive.
"""
import sys
from pathlib import Path

ARCH = Path('/root/capsule/data/claude-data_ophys-mfish-autocoreg_soma-print_260616/sessions')
S16 = ARCH / '16_intermediate_summary_materials'
for p in (str(S16), str(ARCH / '15_geom_features')):
    sys.path.insert(0, p)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import _common
import benchmark_data_loader as bdl


def _hcr_dir(sid):                                   # skip *_HCR-ROI-label_* assets; prefer the round that has centroids
    m = sorted(x for x in bdl.DATA_DIR.glob(f"HCR_{sid}_*_processed_*") if 'HCR-ROI-label' not in x.name)
    m = [x for x in m if (x / 'cell_body_segmentation' / 'cell_centroids.npy').exists()] or m
    return m[-1]
bdl._find_hcr_dir = _hcr_dir
_common.S15 = ARCH / '15_geom_features'; _common.S11 = ARCH / 'v3_S11_roi_quality'
from _common import COLORS, SUBJECTS
from _gt import all_stats, fov_center_offset

OUT = '/scratch/tmp/site/posts/2026-10-auto-coregistration-1-pan-inhibitory/figures/fig_01_gt_expansion.jpg'
st = all_stats()
x = np.arange(len(SUBJECTS)); w = 0.38
FS = 12
plt.rcParams.update({'font.size': FS, 'axes.titlesize': FS + 1, 'axes.labelsize': FS})

fig = plt.figure(figsize=(16, 9.6))
gs = fig.add_gridspec(2, 6, hspace=0.55, wspace=1.1)
axes = {'A': fig.add_subplot(gs[0, 0:2]), 'B': fig.add_subplot(gs[0, 2:4]), 'C': fig.add_subplot(gs[0, 4:6]),
        'D': fig.add_subplot(gs[1, 1:3]), 'E': fig.add_subplot(gs[1, 3:5])}


def xt(ax):
    ax.set_xticks(x); ax.set_xticklabels(SUBJECTS, rotation=30, ha='right')


# A: anisotropic expansion
ax = axes['A']
xy = [d['xy_exp'] for d in st]; zz = [d['sz_exp'] for d in st]
ax.bar(x - w / 2, xy, w, label='XY expansion', color=COLORS['xy'])
ax.bar(x + w / 2, zz, w, label='Z expansion', color=COLORS['z'])
ax.axhline(1.0, color='k', lw=0.8, ls=':')
for xi, v in zip(x - w / 2, xy): ax.text(xi, v + 0.04, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
for xi, v in zip(x + w / 2, zz): ax.text(xi, v + 0.04, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
xt(ax); ax.set_ylabel('expansion factor (HCR / 2p)'); ax.set_title('A · Anisotropic expansion')
ax.set_ylim(0, max(zz) * 1.40); ax.legend(loc='upper right')

# B: pia tilt per modality
ax = axes['B']
czt = [d['cz_tilt'] for d in st]; hct = [d['hcr_tilt'] for d in st]
ax.bar(x - w / 2, czt, w, label='2p pia tilt', color=COLORS['cz'])
ax.bar(x + w / 2, hct, w, label='HCR pia tilt', color=COLORS['hcr'])
for xi, v in zip(x - w / 2, czt): ax.text(xi, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
for xi, v in zip(x + w / 2, hct): ax.text(xi, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
xt(ax); ax.set_ylabel('pia-plane tilt (deg)'); ax.set_title('B · Pia tilt per modality')
ax.set_ylim(0, max(hct) * 1.2); ax.legend(loc='upper right')

# C: in-plane rotation
ax = axes['C']
dev = [d['rot_dev180'] for d in st]
ax.bar(x, dev, 0.6, color='#2b8cbe')
for xi, v in zip(x, dev): ax.text(xi, v + 0.12, f'{v:.1f}°', ha='center', va='bottom', fontsize=8)
xt(ax); ax.set_ylabel('deviation from 180° (deg)'); ax.set_title('C · In-plane rotation (≈180°, varies)')
ax.set_ylim(0, max(dev) * 1.3)

# D: 2p center vs HCR FOV center
ax = axes['D']
offs = [fov_center_offset(sid) for sid in SUBJECTS]
cols = plt.cm.viridis(np.linspace(0, 0.9, len(SUBJECTS)))
for r in (100, 200):
    ax.add_patch(Circle((0, 0), r, fill=False, ec='0.6', lw=0.8, ls=':'))
    ax.text(0, r, f'{r} µm', color='0.5', fontsize=8, ha='center', va='bottom')
ax.scatter([0], [0], marker='+', c='k', s=90, zorder=5)
for o, c in zip(offs, cols):
    ax.scatter(o['dx'], o['dy'], s=70, color=c, zorder=4)
    ax.annotate(f"{o['sid']}", (o['dx'], o['dy']), color=c, fontsize=8, xytext=(5, 4), textcoords='offset points')
lim = 260; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel('Δx (µm)'); ax.set_ylabel('Δy (µm)'); ax.set_title('D · 2p center vs HCR FOV center')
med = float(np.median([o['dist'] for o in offs]))
ax.text(0.02, 0.02, f"offset {min(o['dist'] for o in offs):.0f}–{max(o['dist'] for o in offs):.0f} µm (median {med:.0f})\n"
        f"= {min(o['pct'] for o in offs):.0f}–{max(o['pct'] for o in offs):.0f}% of FOV half-width\n(HCR FOV ≈ 2.3 mm)",
        transform=ax.transAxes, fontsize=8, va='bottom', color='0.2')

# E: nonrigid residual after the global affine
ax = axes['E']
rms = [d['rms_um'] for d in st]; mx = [d['max_um'] for d in st]
ax.bar(x - w / 2, rms, w, label='RMS', color=COLORS['accent'])
ax.bar(x + w / 2, mx, w, label='max', color='#bcbddc')
for xi, v in zip(x - w / 2, rms): ax.text(xi, v + 1.5, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
for xi, v in zip(x + w / 2, mx): ax.text(xi, v + 1.5, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
xt(ax); ax.set_ylabel('residual after global affine (µm)')
ax.set_title('E · Nonrigid residual after the\nbest global anisotropic affine')
ax.set_ylim(0, max(mx) * 1.2); ax.legend(loc='upper left')

fig.suptitle('Ground-truth transform · 2p → HCR (light-sheet) · 6 benchmark subjects', y=0.97, fontsize=15, fontweight='bold')
fig.savefig('/scratch/tmp/diagram/fig_01_gt_expansion.png', dpi=200, bbox_inches='tight', facecolor='white')
from PIL import Image
im = Image.open('/scratch/tmp/diagram/fig_01_gt_expansion.png').convert('RGB'); im.thumbnail((1800, 1800)); im.save(OUT, quality=88)
print('saved', OUT)
