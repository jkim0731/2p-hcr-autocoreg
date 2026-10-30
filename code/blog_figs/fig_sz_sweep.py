"""Axial-scale (s_z) sweep: mean slab FFT-NCC vs s_z with the std across slabs (5 slabs x 100 µm). Subject 790322."""
import json
import archive_env
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

SID = '790322'
sz = json.load(open(f'/root/capsule/code/dev_code/cached_sz/{SID}.json'))
rows = sz['slab_rows']
szs = np.array(sorted({r['sz'] for r in rows}))
slabs = sorted({r['slab_idx'] for r in rows})
M = np.array([[next(r['ncc'] for r in rows if r['sz'] == v and r['slab_idx'] == k) for v in szs] for k in slabs])   # (slab, sz)
mean, std = np.nanmean(M, 0), np.nanstd(M, 0)
best = sz['sz_best']
print('slabs', len(slabs), 'sz grid', szs.min(), szs.max(), 'best', best, 'mean@best', mean[np.argmin(abs(szs - best))], 'std@best', std[np.argmin(abs(szs - best))])

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12})
fig, ax = plt.subplots(figsize=(5.4, 3.9))
ax.fill_between(szs, mean - std, mean + std, color='#54278f', alpha=0.22, lw=0, label='± s.d. across slabs')
ax.plot(szs, mean, '-o', color='#54278f', ms=3, lw=1.6, label='mean across slabs')
ax.axvline(best, color='#d73027', lw=2, label=f'chosen s_z = {best:.2f}')
ax.set_xlabel('axial scale  s_z  (2p → HCR)'); ax.set_ylabel('slab FFT-NCC')
ax.set_title('s_z sweep (peak = chosen scale)')
for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
ax.legend(loc='upper left', fontsize=10, frameon=False)
fig.savefig('/scratch/tmp/diagram/fig_06_sz_fft.png', dpi=220, bbox_inches='tight', facecolor='white')
im = Image.open('/scratch/tmp/diagram/fig_06_sz_fft.png').convert('RGB'); im.save(archive_env.FIGDIR / 'fig_06_sz_fft.jpg', quality=88)
print('saved', im.size)
