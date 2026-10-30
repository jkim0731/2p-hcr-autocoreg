"""Blog figure: same neurons across sessions/depths -> z-stack -> HCR identity (subject 788406).

Inputs
  /scratch/tmp/diagram/{F,N,Np}_VISp_{0..7}.npz   mean FOV + ROI coords (extract.py)
  code/roi_coreg_master_260922.csv                 roi -> ucid / czstack_id / hcr_id / subclass
  coreg asset (z-stack centroids), z-stack registered TIF, HCR cell_centroids.npy
Subclass colors: SPATIAL_SUBCLASS_COLORS (project_specific_info.md section 5; ROI marks never overlap).
"""
import numpy as np, pandas as pd, tifffile, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Circle, Patch
from matplotlib.lines import Line2D
from skimage.measure import find_contours

SID = 788406
SESS = {'F': ('788406_2025-07-14', 'Session N'),       # consecutive sessions, labeled by order only
        'N': ('788406_2025-07-15', 'Session N+1'),
        'Np': ('788406_2025-07-16', 'Session N+2')}
COL = {'Sst': '#FE9901', 'Pvalb': '#D93238', 'Lamp5': '#D98691', 'Sncg': '#DE7BFB', 'Vip': '#A96BC1', 'Other': '#727272'}
ORDER = ['Lamp5', 'Sncg', 'Vip', 'Sst', 'Pvalb', 'Other']          # shallow -> deep (project convention)
D = '/root/capsule/data/'
TMP = '/scratch/tmp/diagram/'
OUT = '/scratch/tmp/site/posts/2026-10-auto-coregistration-1-pan-inhibitory/figures/fig_sessions_to_hcr.jpg'
EX_SUBCLASS = 'Vip'

m = pd.read_csv('/root/capsule/code/roi_coreg_master_260922.csv')
m = m[(m.subject_id == SID) & m.session_key.isin([v[0] for v in SESS.values()])].copy()
m['sess'] = m.session_key.map({v[0]: k for k, v in SESS.items()})
m['p'] = m.plane.str.extract(r'(\d+)$').astype(int)
# plane -> depth differs between sessions (e.g. 154 / 132 / 158 µm for the same cell), so order and label per session
depth = {s: m[m.sess == s].groupby('p').plane_depth_um.first().sort_values() for s in SESS}

# ---- choose the example cell: typed in all 3 sessions, one ucid -> one czstack_id -> one hcr_id, away from FOV edges
t = m[m.valid_roi & m.subclass.notna() & (m.ucid >= 0) & (m.czstack_id > 0) & (m.hcr_id > 0)]
g = t.groupby('ucid').agg(ns=('sess', 'nunique'), sc=('subclass', 'nunique'), cz=('czstack_id', 'nunique'),
                          hc=('hcr_id', 'nunique'), sub=('subclass', 'first'))
cand = g[(g.ns == 3) & (g.sc == 1) & (g.cz == 1) & (g.hc == 1) & (g['sub'] == EX_SUBCLASS)].index


def med(sess, p, idx):
    z = np.load(f'{TMP}{sess}_VISp_{p}.npz'); return z['med'][idx]


best = None
for u in cand:
    r = t[t.ucid == u]
    ys = [med(a.sess, a.p, a.roi_idx) for a in r.itertuples()]
    edge = min(min(y, x, 511 - y, 511 - x) for y, x in ys)
    if edge > 70 and (best is None or edge > best[0]): best = (edge, u)
ex = best[1]
exr = t[t.ucid == ex].set_index('sess')
print('example ucid', ex, exr[['plane_depth_um', 'roi_idx', 'czstack_id', 'hcr_id', 'subclass']].to_string())
ex_cz, ex_hcr = int(exr.czstack_id.iloc[0]), int(exr.hcr_id.iloc[0])

# ---- figure
# A shows sessions N and N+1 only; the "same ROI" crops (stacked at left of B) cover N, N+1 and N+2.
SHOW = ['F', 'N']
fig = plt.figure(figsize=(20, 12.4), facecolor='white')
gs = fig.add_gridspec(2, 1, height_ratios=[0.80, 1.0], hspace=0.16, left=0.015, right=0.985, top=0.945, bottom=0.03)
top = gs[0].subgridspec(1, 3, width_ratios=[1, 1, 0.42], wspace=0.06)
ex_axes = {}
ex_img = {}
for s in SESS:                                   # example ROI location + FOV for every session (crops use all three)
    p_ex = int(exr.loc[s, 'p']); z = np.load(f'{TMP}{s}_VISp_{p_ex}.npz')
    ex_img[s] = (z['mean'], z, z['med'][int(exr.loc[s, 'roi_idx'])], p_ex)
for j, s in enumerate(SHOW):
    key, lab = SESS[s]
    sub = top[j].subgridspec(2, 4, wspace=0.02, hspace=0.12)
    rows = m[m.sess == s]
    for k, p in enumerate(depth[s].index):
        ax = fig.add_subplot(sub[k // 4, k % 4])
        z = np.load(f'{TMP}{s}_VISp_{p}.npz')
        img = z['mean']; lo, hi = np.percentile(img, [1, 99.7])
        ax.imshow(np.clip((img - lo) / (hi - lo), 0, 1) ** 0.6, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        r = rows[(rows.p == p) & rows.valid_roi]
        lab_of = dict(zip(r.roi_idx, r.subclass))
        cc = z['coords']
        for idx in np.unique(cc[0]):
            if idx not in lab_of: continue
            sel = cc[0] == idx
            mk = np.zeros((512, 512), bool); mk[cc[1][sel], cc[2][sel]] = True
            sc = lab_of[idx]; typed = isinstance(sc, str)
            for c in find_contours(np.pad(mk, 1).astype(float), 0.5):
                ax.plot(c[:, 1] - 1, c[:, 0] - 1, lw=1.5 if typed else 0.5,
                        color=COL[sc] if typed else (1, 1, 1, 0.35), solid_joinstyle='round')
        if int(exr.loc[s, 'p']) == p:
            y, x = z['med'][int(exr.loc[s, 'roi_idx'])]
            ax.add_patch(Circle((x, y), 30, fill=False, ec='white', lw=2.2))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color('#444'); sp.set_linewidth(0.6)
        ax.text(6, 27, f'{40 * (k + 1)} µm', color='white', fontsize=13.5, fontweight='bold', va='center',
                bbox=dict(fc='black', ec='none', alpha=0.55, pad=1.5))
        if k == 0: ax.set_title(lab, fontsize=16, loc='left', pad=6, fontweight='bold')
fig.text(0.015, 0.968, 'A   FOV ROIs in each session', fontsize=18, fontweight='bold', va='center')

# legend (top right)
axk = fig.add_subplot(top[2]); axk.axis('off')
axk.legend(handles=[Patch(fc='none', ec=COL[s], lw=2.5, label=s) for s in ORDER],
           loc='center left', ncol=1, frameon=False, fontsize=17, title='HCR subclass', title_fontsize=17, bbox_to_anchor=(0.12, 0.5))

# ---- bottom row: stacked ROI crops | two-photon z-stack | ex vivo HCR
bot = gs[1].subgridspec(1, 5, width_ratios=[0.40, 0.20, 1.0, 0.10, 1.0], wspace=0.0)
left = bot[0].subgridspec(3, 1, hspace=0.34)
crops = {}
for j, s in enumerate(SESS):
    img, z, (y, x), p = ex_img[s]
    axc = fig.add_subplot(left[j])
    h = 45; x0, y0 = int(round(x)), int(round(y))
    lo, hi = np.percentile(img, [1, 99.7])
    axc.imshow(np.clip((img - lo) / (hi - lo), 0, 1) ** 0.6, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    sel = z['coords'][0] == int(exr.loc[s, 'roi_idx']); mk = np.zeros((512, 512), bool); mk[z['coords'][1][sel], z['coords'][2][sel]] = True
    for c in find_contours(np.pad(mk, 1).astype(float), 0.5): axc.plot(c[:, 1] - 1, c[:, 0] - 1, color=COL[EX_SUBCLASS], lw=2.2)
    axc.set_xlim(x0 - h, x0 + h); axc.set_ylim(y0 + h, y0 - h); axc.set_xticks([]); axc.set_yticks([])
    for sp in axc.spines.values(): sp.set_color('white'); sp.set_linewidth(2.2)
    axc.set_title(f'ROI X / {SESS[s][1].lower().replace("session n", "session N")}', fontsize=15, pad=5)
    crops[s] = axc
fig.text(0.015, 0.553, 'B   Identifying session ROI cell type using structural 2p z-stack as an anchor', fontsize=18, fontweight='bold', va='center')

# z-stack
cz = pd.read_csv(D + '788406_2025-07-18_ctl-czstack-hcr-coreg_2026-03-02_00-00-00/788406_2025-07-18_czstack_cell_centroids.csv')
stk = tifffile.imread(D + 'multiplane-ophys_788406_2025-07-18_12-17-32_cortical-zstack-registration_2025-11-11_20-55-55/'
                      'cortical_zstack_0/channel_0_ref_0/cortical_zstack_0_2xREG.tif')
mip = stk.max(0); lo, hi = np.percentile(mip, [1, 99.7])
axz = fig.add_subplot(bot[2])
axz.imshow(np.clip((mip - lo) / (hi - lo), 0, 1) ** 0.6, cmap='gray', interpolation='nearest')
pool = m[m.valid_roi & m.subclass.notna() & (m.czstack_id > 0)].drop_duplicates('czstack_id').set_index('czstack_id').subclass
czc = cz.set_index('czstack_cell_id')
for sc in ORDER:
    ids = [i for i in pool.index[pool == sc] if i in czc.index]
    axz.scatter(czc.loc[ids, 'czstack_x'], czc.loc[ids, 'czstack_y'], s=34, facecolors='none', edgecolors=COL[sc], linewidths=1.4)
cx, cy = czc.loc[ex_cz, ['czstack_x', 'czstack_y']]
axz.add_patch(Circle((cx, cy), 22, fill=False, ec='white', lw=2.2))
axz.set_xticks([]); axz.set_yticks([])
axz.set_title('Two-photon z-stack', fontsize=17, fontweight='bold', pad=8)

# HCR
hc = np.load(D + 'HCR_788406_2025-11-05_13-00-00_processed_2025-11-10_20-35-28/cell_body_segmentation/cell_centroids.npy')
hcs = pd.DataFrame(hc, columns=['z', 'y', 'x', 'id']).set_index('id')
hpool = m[m.valid_roi & m.subclass.notna() & (m.hcr_id > 0)].drop_duplicates('hcr_id').set_index('hcr_id').subclass
axh = fig.add_subplot(bot[4])
axh.scatter(hcs.x, hcs.y, s=1.2, c='#c8c8c8', linewidths=0, rasterized=True)
for sc in ORDER:
    ids = [i for i in hpool.index[hpool == sc] if i in hcs.index]
    axh.scatter(hcs.loc[ids, 'x'], hcs.loc[ids, 'y'], s=16, c=COL[sc], linewidths=0)
hx, hy = hcs.loc[ex_hcr, ['x', 'y']]
axh.add_patch(Circle((hx, hy), 18, fill=False, ec='black', lw=2.2))
allm = hcs.loc[[i for i in hpool.index if i in hcs.index]]
pad = 0.25 * max(np.ptp(allm.x), np.ptp(allm.y)); cxm, cym = allm.x.mean(), allm.y.mean(); half = max(np.ptp(allm.x), np.ptp(allm.y)) / 2 + pad
axh.set_xlim(cxm - half, cxm + half); axh.set_ylim(cym + half, cym - half)
axh.set_aspect('equal'); axh.set_xticks([]); axh.set_yticks([])
for sp in axh.spines.values(): sp.set_color('#888')
axh.set_title('Ex vivo HCR volume', fontsize=17, fontweight='bold', pad=8)

# ---- links: crops -> (bus) -> z-stack ring -> HCR cell
fig.canvas.draw()
inv = fig.transFigure.inverted()
ring = inv.transform(axz.transData.transform((cx, cy)))
cp = {s: a.get_position() for s, a in crops.items()}
zb = axz.get_position()
xbus = (max(p.x1 for p in cp.values()) + zb.x0) / 2
ys = [(p.y0 + p.y1) / 2 for p in cp.values()] + [ring[1]]
col = '#222'
for s, p in cp.items():
    yc = (p.y0 + p.y1) / 2
    fig.add_artist(Line2D([p.x1 + 0.002, xbus], [yc, yc], transform=fig.transFigure, color=col, lw=2.0))
fig.add_artist(Line2D([xbus, xbus], [min(ys), max(ys)], transform=fig.transFigure, color=col, lw=2.0))
a = ConnectionPatch(xyA=(xbus, ring[1]), coordsA='figure fraction', xyB=(cx, cy), coordsB=axz.transData, arrowstyle='-|>', lw=2.0, color=col,
                    mutation_scale=18, shrinkA=0, shrinkB=3, zorder=20); a.set_clip_on(False); fig.add_artist(a)
b = ConnectionPatch(xyA=(cx, cy), coordsA=axz.transData, xyB=(hx, hy), coordsB=axh.transData, arrowstyle='-|>', lw=2.6, color=col,
                    mutation_scale=18, shrinkA=3, shrinkB=3, zorder=20); b.set_clip_on(False); axh.add_artist(b)
fig.savefig(OUT, dpi=150, facecolor='white', pil_kwargs=dict(quality=88))
print('saved', OUT)
