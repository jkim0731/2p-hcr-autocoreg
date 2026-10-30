"""L1/L2 boundary from cell density (subject 839909): side view + density profile, 2p and HCR. Data: l1_data.py."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
d = np.load('/scratch/tmp/diagram/l1_839909.npz')
OUT = '/scratch/tmp/site/posts/2026-10-auto-coregistration-2-pan-neuronal/figures/839909_l1_estimation.jpg'
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13.5, 'axes.labelsize': 12})
CY, OR = '#00b7d4', '#ff8c00'

def norm(a, lo=1, hi=99.7, g=0.7):
    v = a[a > 0]; l, h = np.percentile(v, [lo, hi]); return np.clip((a - l) / (h - l), 0, 1) ** g

fig = plt.figure(figsize=(25, 6))
gs = fig.add_gridspec(1, 6, width_ratios=[1.35, 0.75, 1.35, 0.75, 0.12, 1.75], wspace=0.34)   # col 4 = spacer for the HCR right-side y label
ax = [fig.add_subplot(gs[0, i]) for i in range(4)]
# ---- 2p
a, b, c, l12, ym = float(d['a']), float(d['b']), float(d['c']), float(d['l12']), float(d['ym_um'])
img = norm(d['band2p']); xs = np.arange(img.shape[1]) * 0.78
ax[0].imshow(img, cmap='gray', extent=[0, xs[-1], img.shape[0], 0], aspect='auto', interpolation='nearest')
ax[0].plot(xs, a * xs + b * ym + c, color=CY, lw=2.2, label='pia'); ax[0].plot(xs, a * xs + b * ym + l12, color=OR, lw=2.2, label='L1/L2 boundary')
ax[0].set_ylim(330, 0); ax[0].set_xlim(0, xs[-1]); ax[0].set_xlabel('x (µm)'); ax[0].set_ylabel('depth in z-stack (µm)'); ax[0].set_title('2p z-stack, side view')
ax[0].legend(loc='lower left', fontsize=11, labelcolor='w', facecolor='k', framealpha=0.5, edgecolor='none')
ax[1].plot(d['dens2'], d['cent2'], color='#54278f', lw=2); ax[1].fill_betweenx(d['cent2'], 0, d['dens2'], color='#54278f', alpha=0.2)
ax[1].axhline(c, color=CY, lw=2); ax[1].axhline(l12, color=OR, lw=2); ax[1].set_ylim(330, 0)
ax[1].annotate('', xy=(d['dens2'].max() * 0.55, l12), xytext=(d['dens2'].max() * 0.55, c), arrowprops=dict(arrowstyle='<->', lw=1.8))
ax[1].text(d['dens2'].max() * 0.6, (c + l12) / 2, f'L1\n{l12 - c:.0f} µm', va='center', fontsize=12)
ax[1].set_xlabel('ROIs per 4 µm'); ax[1].set_yticklabels([]); ax[1].set_title('2p ROI density')
# ---- HCR
zs = d['zs_h']; xh = d['xh']; imh = norm(d['bandh'].astype(float)); zmax = imh.shape[0] * 4.0
ax[2].imshow(imh, cmap='gray', extent=[0, xh[-1], zmax, 0], aspect='auto', interpolation='nearest')
on = float(d['hcr_onset'])
ax[2].plot(xh, zs, color=CY, lw=2.2); ax[2].plot(xh, zs + on, color=OR, lw=2.2)
zc = float(np.median(zs)); ax[2].set_ylim(zc + 300, zc - 60); ax[2].set_xlim(0, xh[-1]); ax[2].set_xlabel('x (µm)'); ax[2].set_ylabel('z (µm)'); ax[2].set_title('HCR 405, side view')
ax[3].plot(d['dens_h'], d['cent_h'], color='#54278f', lw=2); ax[3].fill_betweenx(d['cent_h'], 0, d['dens_h'], color='#54278f', alpha=0.2)
ax[3].axhline(0, color=CY, lw=2); ax[3].axhline(on, color=OR, lw=2); ax[3].set_ylim(300, -60)
m = d['dens_h'].max()
ax[3].annotate('', xy=(m * 0.5, on), xytext=(m * 0.5, 0), arrowprops=dict(arrowstyle='<->', lw=1.8))
ax[3].text(m * 0.55, on / 2, f'L1\n{on:.0f} µm', va='center', fontsize=12)
ax[3].set_xlabel('GFP+ ROIs per 6 µm'); ax[3].set_ylabel('depth below pia (µm)'); ax[3].yaxis.set_label_position('right'); ax[3].yaxis.tick_right(); ax[3].set_title('HCR ROI density')
for x_ in (ax[1], ax[3]):
    for s_ in ('top',): x_.spines[s_].set_visible(False)
# ---- s_z regression on matched pairs (data: fig_part2_new.py)
r = np.load('/scratch/tmp/diagram/sz_regression_839909.npz'); cd, hd, ols, seed = r['cd'], r['hd'], float(r['ols']), float(r['seed'])
b_ = fig.add_subplot(gs[0, 5])
b_.scatter(cd, hd, s=3, c='#444', alpha=0.25, linewidths=0, rasterized=True)
xx = np.array([0, cd.max()]); b_.plot(xx, ols * xx, color='k', lw=2, ls='--', label=f'OLS through origin: slope {ols:.2f}')
b_.plot(xx, seed * xx, color='#2c7fb8', lw=1.6, label=f'L1-ratio seed: {seed:.2f}')
b_.set_xlabel('2p depth below pia (µm)'); b_.set_ylabel('HCR depth below pia (µm)'); b_.set_title('Matched pairs')
b_.legend(loc='lower right', fontsize=11, frameon=False); b_.set_box_aspect(1)
for s_ in ('top', 'right'): b_.spines[s_].set_visible(False)
fig.savefig('/scratch/tmp/diagram/p2_l1.png', dpi=170, bbox_inches='tight', facecolor='white')
Image.open('/scratch/tmp/diagram/p2_l1.png').convert('RGB').save(OUT, quality=88)
print('saved; L1 2p', l12 - c, 'HCR', on, 'ratio', on / (l12 - c))
