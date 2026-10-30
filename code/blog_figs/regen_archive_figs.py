"""Regenerate session-16 archive figures with blog wording: 'CZ' -> '2p', American spellings.

Usage: python regen_archive_figs.py fig_01_gt_expansion [fig_03_surfaces ...]
The archive scripts are read-only and are executed from source with string replacements and a patched save_fig
that writes a JPG into the post's figures/ folder.  Nothing in the archive is modified.
"""
import sys, runpy, types
from pathlib import Path

ARCH = Path('/root/capsule/data/claude-data_ophys-mfish-autocoreg_soma-print_260616/sessions')
S16 = ARCH / '16_intermediate_summary_materials'
OUTDIR = Path('/scratch/tmp/site/posts/2026-10-auto-coregistration-1-pan-inhibitory/figures')
for p in (str(S16), str(ARCH / '15_geom_features')):
    sys.path.insert(0, p)

# literal string-level replacements (label text only; identifiers such as COLORS["cz"], M_CZ, cz_vecs untouched)
REP = {
    'fig_01_gt_expansion': [
        ('"expansion factor  (HCR / CZ)"', '"expansion factor  (HCR / 2p)"'),
        ('label="CZ pia tilt"', 'label="2p pia tilt"'),
        ('"D · GT relative tilt  CZ↔HCR  (arccos R₂₂)"', '"D · GT relative tilt  2p↔HCR  (arccos R₂₂)"'),
        ('"F · CZ centre vs HCR FOV centre"', '"F · 2p center vs HCR FOV center"'),
        ('"Ground-truth transform · CZ (2p) → HCR (light-sheet) · 6 benchmark subjects"', '"Ground-truth transform · 2p → HCR (light-sheet) · 6 benchmark subjects"'),
    ],
    'fig_03_surfaces': [
        ('f"CZ (2p GCaMP) — side view + pia', 'f"Two-photon z-stack — side view + pia'),
        ('ax.legend(loc="lower right", fontsize=9)', 'ax.legend(loc="lower right", fontsize=11, labelcolor="white", facecolor="black", framealpha=0.5)'),
        (r"\nbuffer ≈ {d['c']:.0f} µm", ""),
        ('f"Pia surface fits define a depth-from-surface coordinate  ·  {SID}"', '"Pia surface fits define a depth-from-surface coordinate"'),
    ],
    'fig_04_sxy_roi_area': [
        ('label=f"CZ (2p), median {med_cz:.0f} µm²  (n={len(cz_area)})"', 'label=f"2p, median {med_cz:.0f} µm²  (n={len(cz_area)})"'),
        ('"cell tight xy-bbox area (µm²)"', '"cell xy bounding-box area (µm²)"'),
        ('f"A · ROI cross-sectional area · sxy = √(med_HCR / med_CZ) = {sxy:.3f}"', 'f"ROI cross-sectional area\\ns_xy = √(median HCR / median 2p) = {sxy:.2f}"'),
        ('axc.set_title(f"CZ {cz_id}", fontsize=8)', 'axc.set_title("2p", fontsize=11)'),
        ('axh.set_title(f"HCR {hcr_id}", fontsize=8)', 'axh.set_title("HCR", fontsize=11)'),
        ('"B · matched cells, equal µm scale (40 µm half-window)"', '"B · matched cells at equal µm scale (80 µm field)"'),
    ],
    'fig_05_topslab_reg': [
        ('Top-slab 2-D registration: CZ footprint → HCR-488', 'Top-slab 2-D registration: 2p footprint → HCR-488'),
        ('best = reg["best_method"]', 'best = reg["best_method"]; PWR_NAMES = {"pwr4x4": "piecewise rigid 4×4", "pwr3x3": "piecewise rigid 3×3"}'),
        ('f"C · {best} (after)', 'f"C · {PWR_NAMES.get(best, best)} (after)'),
        ('B · rigid bootstrap (before)', 'B · rigid (before)'),
    ],
    'fig_06_sz_fft': [
        ('"axial scale  s_z  (CZ → HCR)"', '"axial scale  s_z  (2p → HCR)"'),
        ('label=f"CZ depth × s_z ({sz_best:.2f})"', 'label=f"2p depth × s_z ({sz_best:.2f})"'),
        ('label="CZ depth (raw)"', 'label="2p depth (raw)"'),
        ('"B · s_z aligns CZ & HCR depth profiles"', '"B · s_z aligns 2p & HCR depth profiles"'),
        ('"normalised density"', '"normalized density"'),
    ],
    'fig_08_somaprint_concept': [
        ('f"A · CZ cell {cz_id}\\nm={M_CZ} neighbour vectors"', 'f"A · 2p cell {cz_id}\\nm={M_CZ} neighbor vectors"'),
        ('f"B · HCR cell {hcr_id}\\nm={M_HCR} neighbour vectors"', 'f"B · HCR cell {hcr_id}\\nm={M_HCR} neighbor vectors"'),
        ('"soma-print: match the local neighbour-vector constellation (relative geometry)"', '"soma-print: match the local neighbor-vector constellation (relative geometry)"'),
    ],
    'fig_09_step2_robustness': [],
    'fig_10_final_matches': [
        ('label=f"not in GT (n={len(ng)})"', 'label=f"not in manual table (n={len(ng)})"'),
        ('label=f"GT-confirmed (n={len(gt)})"', 'label=f"in manual table (n={len(gt)})"'),
        ('ax.legend(loc="upper right", fontsize=9, markerscale=1.6, framealpha=0.7)', 'ax.legend(loc="upper right", fontsize=11, markerscale=1.6, labelcolor="white", facecolor="black", framealpha=0.85)'),
        ('f"Cell-cell matches · soma-print + local_flow · {SID} · {len(df)} pairs · "', 'f"Cell-cell matches · {SID} · {len(df)} pairs · "'),
        ('f"{n_round}\\nGT recall lower-bound: {len(gt)}/{len(df)} agree with coreg-table "', 'f"final round\\n{len(gt)}/{len(df)} agree with the manual coregistration table "'),
        ('f"(GT n={len(s.coreg_table)})"', 'f"(n={len(s.coreg_table)})"'),
        ('"B · Displacement resolved by matching\\n(locked-prior CZ → matched HCR)"', '"B · Displacement resolved by matching\\n(rough-registered 2p → matched HCR)"'),
    ],
}


# code-level edits (applied after the string replacements)
RESIDUAL_PANELS = '''
# --- G/H/I: nonrigid residual after the best global anisotropic affine (added for the blog) --------------------
from _gt import landmark_residuals_um
w2 = 0.38
ax = axes[2, 0]
rms = [d["rms_um"] for d in st]; mx = [d["max_um"] for d in st]
ax.bar(x - w2 / 2, rms, w2, label="RMS", color=COLORS["accent"])
ax.bar(x + w2 / 2, mx, w2, label="max", color="#bcbddc")
for xi, v in zip(x - w2 / 2, rms): ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
for xi, v in zip(x + w2 / 2, mx): ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
xt(ax); ax.set_ylabel("residual after global affine (µm)")
ax.set_title("G · Nonrigid residual (landmark mismatch\\nafter the best global anisotropic affine)")
ax.set_ylim(0, max(mx) * 1.2); ax.legend(loc="upper left")
hcr_l, res_l, mag_l = landmark_residuals_um(SID)
vmax = np.percentile(mag_l, 95)
for ax, k, lab, ttl in ((axes[2, 1], 1, "HCR y (µm)", f"H · {SID} residual field (xy)\\narrow = mismatch left after the affine"),
                        (axes[2, 2], 2, "HCR z / depth (µm)", f"I · {SID} residual field (depth)\\narrow = mismatch left after the affine")):
    q = ax.quiver(hcr_l[:, 0], hcr_l[:, k], res_l[:, 0], res_l[:, k], mag_l, cmap="viridis", clim=(0, vmax),
                  angles="xy", scale_units="xy", scale=0.18, width=0.004, alpha=0.9)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.grid(alpha=0.25)
    ax.set_xlabel("HCR x (µm)"); ax.set_ylabel(lab); ax.set_title(ttl)
cb = fig.colorbar(q, ax=axes[2, 2], fraction=0.046, pad=0.04); cb.set_label("|residual| (µm)")
'''


def xform_fig_01(src):
    src = src.replace('plt.subplots(2, 3, figsize=(16, 8.6))', 'plt.subplots(3, 3, figsize=(16, 12.8))')
    i = src.index('fig.suptitle(')
    return src[:i] + RESIDUAL_PANELS + '\n' + src[i:]


def xform_fig_06(src):                                # keep only panel A (the s_z sweep)
    src = src.replace('fig, axes = plt.subplots(1, 2, figsize=(13, 5))', 'fig, axes = plt.subplots(1, 1, figsize=(7.4, 5.2)); axes = [axes]')
    i = src.index('# --- B: depth-density alignment'); j = src.index('fig.suptitle(')
    src = src[:i] + src[j:]
    src = src.replace('"A · s_z sweep (peak = chosen scale)"', '"s_z sweep (peak = chosen scale)"')
    return src.replace('y=1.0, fontsize=13', 'y=1.02, fontsize=13')


def xform_fig_10(src):                                # no figure title
    i = src.index('n_round = '); j = src.index('save_fig(fig, "fig_10_final_matches")')
    return src[:i] + src[j:]


def xform_fig_04(src):                                # single panel (area distributions only); no figure title; larger text
    i = src.index('# --- B: example matched cells'); j = src.index('save_fig(fig, "fig_04_sxy_roi_area")')
    src = src[:i] + src[j:]
    src = src.replace('plt.figure(figsize=(14, 5.2))', 'plt.figure(figsize=(7.0, 4.6))').replace('fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.22)', 'fig.add_gridspec(1, 1)')
    src = src.replace('s = load_s()', 'plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12})\ns = load_s()', 1)
    return src.replace('ax.legend(loc="upper right", fontsize=9)', 'ax.legend(loc="upper left", fontsize=10.5, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95); ax.set_ylim(0, ax.get_ylim()[1] * 1.32)')


# NOTE: fig_01 and fig_06 are NOT produced here any more: see fig_gt_transform.py and fig_sz_sweep.py
XFORM = {'fig_01_gt_expansion': xform_fig_01, 'fig_06_sz_fft': xform_fig_06, 'fig_10_final_matches': xform_fig_10,
         'fig_04_sxy_roi_area': xform_fig_04}


def main(names):
    import matplotlib
    matplotlib.use('Agg')
    import _common
    import benchmark_data_loader as bdl

    def _hcr_dir(subject_id):                      # skip the *_HCR-ROI-label_* variant assets
        m = sorted(x for x in bdl.DATA_DIR.glob(f"HCR_{subject_id}_*_processed_*") if 'HCR-ROI-label' not in x.name)
        m = [x for x in m if (x / 'cell_body_segmentation' / 'cell_centroids.npy').exists()] or m     # 755252/767022 have R1 (segmentation) + R2 assets
        return m[-1]
    bdl._find_hcr_dir = _hcr_dir

    def save_fig(fig, name, pdf=True):
        import matplotlib.pyplot as plt
        from PIL import Image
        png = Path('/scratch/tmp/diagram') / f'{name}.png'
        fig.savefig(png, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
        im = Image.open(png).convert('RGB'); im.thumbnail((1800, 1800))
        out = OUTDIR / f'{name}.jpg'; im.save(out, quality=88); print('  saved', out)
    _common.save_fig = save_fig
    _common.S15 = ARCH / '15_geom_features'; _common.S11 = ARCH / 'v3_S11_roi_quality'      # session outputs live in the archive

    for n in names:
        src = (S16 / f'{n}.py').read_text()
        src = src.replace('/root/capsule/code/sessions/', str(ARCH) + '/')      # session outputs now live in the archive asset
        for a, b in REP[n]:
            assert a in src, f'{n}: pattern not found: {a[:60]}'
            src = src.replace(a, b)
        if n in XFORM: src = XFORM[n](src)
        left = [l for l in src.splitlines() if 'CZ' in l and ('"' in l or "'" in l) and not l.lstrip().startswith(('#', '"""'))]
        print(n, 'remaining CZ-in-code lines:', [l.strip()[:80] for l in left])
        g = {'__name__': '__main__', '__file__': str(S16 / f'{n}.py')}
        exec(compile(src, str(S16 / f'{n}.py'), 'exec'), g)


if __name__ == '__main__':
    main(sys.argv[1:])
