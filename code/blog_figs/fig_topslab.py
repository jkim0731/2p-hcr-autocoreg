"""Top-slab 2-D registration, redrawn: HCR 488 top slab (bright green), 2p top slab (magenta), overlay (white = overlap).
Panels: HCR | 2p | overlay after rigid | overlay after piecewise-rigid. Subject 790322 (data via the session-16 archive helpers)."""
import archive_env
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from _common import SID, load_s
from matplotlib.patches import Rectangle

s = load_s()
from surface_registration_v2 import apply_registration, get_surface_registration
reg = get_surface_registration(s); best = reg['best_method']
rigid = apply_registration(s, reg, method='rigid'); bestw = apply_registration(s, reg, method=best)
hcr = np.asarray(rigid['hcr_target_crop'], float)
print('best', best, 'ncc rigid/best', reg['methods']['rigid']['ncc'], reg['methods'][best]['ncc'])
print({k: reg[k] for k in reg if 'slab' in k or 'xy' in k or k in ('sxy',)})


def norm(a, lo=1, hi=99.5, gamma=0.8):
    a = np.asarray(a, float); l, h = np.percentile(a, [lo, hi]); return np.clip((a - l) / (h - l + 1e-9), 0, 1) ** gamma


G = norm(hcr)
def two_p(w): w = np.asarray(w, float); return np.clip(w / (w.max() + 1e-9), 0, 1)
def rgb(g, m): return np.stack([m, g, m], -1)           # magenta = R+B, green = G  ->  overlap = white
P_r, P_b = two_p(rigid['warped']), two_p(bestw['warped'])
names = {'pwr4x4': 'piecewise rigid 4×4', 'pwr3x3': 'piecewise rigid 3×3'}
plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(1, 4, figsize=(18, 4.9))
axes[0].imshow(rgb(G, np.zeros_like(G))); axes[0].set_title('HCR 488 top slab', color='#1a9a1a', fontweight='bold', fontsize=15)
axes[1].imshow(rgb(np.zeros_like(G), P_r)); axes[1].set_title('2p top slab', color='#c000c0', fontweight='bold', fontsize=15)
axes[2].imshow(rgb(G, P_r)); axes[2].set_title(f'Overlay after rigid\nNCC = {reg["methods"]["rigid"]["ncc"]:.2f}', fontsize=15)
axes[3].imshow(rgb(G, P_b)); axes[3].set_title(f'Overlay after {names.get(best, best)}\nNCC = {reg["methods"][best]["ncc"]:.2f}', fontsize=15)
hxy = reg.get('hcr_xy_um') or getattr(s, 'hcr_xy_um', None)
for ax in axes:
    ax.set_axis_off()
if hxy:
    L = 200 / hxy; H, W = G.shape
    axes[0].plot([12, 12 + L], [H - 14] * 2, color='w', lw=4); axes[0].text(12 + L / 2, H - 22, '200 µm', color='w', ha='center', fontsize=12)
fig.subplots_adjust(wspace=0.04)
fig.savefig('/scratch/tmp/diagram/fig_05_topslab_reg.png', dpi=170, bbox_inches='tight', facecolor='white')
im = Image.open('/scratch/tmp/diagram/fig_05_topslab_reg.png').convert('RGB'); im.thumbnail((2000, 2000)); im.save(archive_env.FIGDIR / 'fig_05_topslab_reg.jpg', quality=88)
print('saved', im.size)
