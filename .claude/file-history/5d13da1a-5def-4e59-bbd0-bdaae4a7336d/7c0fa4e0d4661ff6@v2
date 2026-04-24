"""Build notebooks/05_R1_walkthrough.ipynb — step-by-step walkthrough of r1_revised.

For subject 788406, walks through each of the 8 method-sketch steps with
a visual explanation.  No benchmark statistics needed — the notebook is
self-contained and loads the subject via the standard loader.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path("/root/capsule/code")
OUT = ROOT / "notebooks" / "05_R1_walkthrough.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(
        "# R1 revised — step-by-step walkthrough\n\n"
        "Running example: subject **788406**. Each section does one step "
        "from `r1_revised.coarse_align_revised()`, then visualises what "
        "happens. Source module: `dev_code/r1_revised.py`.\n\n"
        "**Goal of R1.** Produce a coarse affine `hcr = (cz − cz_mean) @ R · S + t` "
        "that places the CZ sub-volume inside the HCR volume. "
        "**Minimal output** `(R, t)` always; **extended output** adds `S` when the "
        "data-driven scale search is confident.\n\n"
        "**The 8 steps.**\n"
        "1. Fit a plane to each pia surface.\n"
        "2. Tilt-aligned rotation `R = R_180 · R_tilt`.\n"
        "3. Centroid translation `t = hcr_mean − R · cz_mean`.\n"
        "4. Compute depth-from-pia for both modalities.\n"
        "5. Search `sz` + `tz` via 1-D partial-overlap NCC.\n"
        "6. Search `sxy` + `(tx, ty)` via 2-D density-map NCC.\n"
        "7. Anisotropic refinement of `(sx, sy)` around the best `sxy`.\n"
        "8. Confidence gating: emit scales only when robust-z prominence "
        "≥ threshold."
    ))

    cells.append(md(
        "## Setup — load subject 788406"
    ))

    cells.append(code(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, '/root/capsule/code/dev_code')\n"
        "\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from benchmark_data_loader import load_subject\n"
        "from benchmark_analysis import analyze_subject\n"
        "import r1_revised as R1\n"
        "\n"
        "sid = '788406'\n"
        "s = load_subject(sid)\n"
        "info = analyze_subject(s)\n"
        "cz_xyz = info['cz_xyz']\n"
        "hcr_xyz = info['gfp_xyz']\n"
        "cz_surf = info['cz_surface']\n"
        "hcr_surf = info['hcr_surface']\n"
        "print(f'CZ cells: {cz_xyz.shape[0]}')\n"
        "print(f'HCR GFP+ cells: {hcr_xyz.shape[0]}')\n"
        "print(f'CZ XY span: {cz_xyz[:,0].ptp():.0f} × {cz_xyz[:,1].ptp():.0f} µm')\n"
        "print(f'HCR XY span: {hcr_xyz[:,0].ptp():.0f} × {hcr_xyz[:,1].ptp():.0f} µm')"
    ))

    cells.append(code(
        "fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))\n"
        "ax[0].scatter(hcr_xyz[:,0], hcr_xyz[:,1], s=1, color='#ff7f0e', alpha=0.3, label='HCR GFP+')\n"
        "ax[0].scatter(cz_xyz[:,0], cz_xyz[:,1], s=4, color='#1f77b4', alpha=0.8, label='CZ cells')\n"
        "ax[0].set_aspect('equal')\n"
        "ax[0].set_xlabel('x (µm)'); ax[0].set_ylabel('y (µm)')\n"
        "ax[0].set_title('Raw XY — CZ and HCR are in different coordinate frames')\n"
        "ax[0].legend()\n"
        "ax[0].grid(alpha=0.3)\n"
        "ax[1].scatter(hcr_xyz[:,0], hcr_xyz[:,2], s=1, color='#ff7f0e', alpha=0.3)\n"
        "ax[1].scatter(cz_xyz[:,0], cz_xyz[:,2], s=4, color='#1f77b4', alpha=0.8)\n"
        "ax[1].invert_yaxis()\n"
        "ax[1].set_xlabel('x (µm)'); ax[1].set_ylabel('z (µm)')\n"
        "ax[1].set_title('Raw XZ — note the 180° rotation between frames')\n"
        "ax[1].grid(alpha=0.3)\n"
        "fig.tight_layout(); plt.show()"
    ))

    # ----- STEP 1 -----
    cells.append(md(
        "## Step 1 — Fit a plane to each pia surface\n\n"
        "Each modality already has a pia-surface model "
        "(`estimate_pia_surface_image_ceiling` for CZ, "
        "`estimate_pia_surface_quantile_ceiling` for HCR). We sample each "
        "surface on a 40 × 40 grid over its XY envelope and do a "
        "least-squares plane fit.  The plane's unit normal — the *into-tissue* "
        "direction — is what step 2 will align."
    ))

    cells.append(code(
        "n_cz, cz_abc = R1._plane_normal_from_surface(cz_surf, cz_xyz[:,:2])\n"
        "n_hcr, hcr_abc = R1._plane_normal_from_surface(hcr_surf, hcr_xyz[:,:2])\n"
        "print(f'n_cz  = {n_cz}  (tilt {np.degrees(np.arccos(n_cz[2])):.2f}°)')\n"
        "print(f'n_hcr = {n_hcr}  (tilt {np.degrees(np.arccos(n_hcr[2])):.2f}°)')"
    ))

    cells.append(code(
        "def sample_surface(surf, xy, n=40):\n"
        "    xlo, xhi = xy[:,0].min(), xy[:,0].max()\n"
        "    ylo, yhi = xy[:,1].min(), xy[:,1].max()\n"
        "    xs = np.linspace(xlo, xhi, n); ys = np.linspace(ylo, yhi, n)\n"
        "    X, Y = np.meshgrid(xs, ys)\n"
        "    Z = R1._surface_z_at(surf, X.ravel(), Y.ravel()).reshape(X.shape)\n"
        "    return X, Y, Z\n"
        "\n"
        "Xc, Yc, Zc = sample_surface(cz_surf, cz_xyz[:,:2])\n"
        "Xh, Yh, Zh = sample_surface(hcr_surf, hcr_xyz[:,:2])\n"
        "\n"
        "fig = plt.figure(figsize=(11,4.5))\n"
        "ax1 = fig.add_subplot(1,2,1, projection='3d')\n"
        "ax1.plot_surface(Xc, Yc, Zc, color='#1f77b4', alpha=0.6, edgecolor='none')\n"
        "# plane-fit overlay (flat at cz mean)\n"
        "a,b,c = cz_abc\n"
        "Zc_plane = a*Xc + b*Yc + c\n"
        "ax1.plot_surface(Xc, Yc, Zc_plane, color='#333', alpha=0.2, edgecolor='none')\n"
        "ax1.invert_zaxis(); ax1.set_title(f'CZ pia (tilt {np.degrees(np.arccos(n_cz[2])):.2f}°)'); \n"
        "ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')\n"
        "ax2 = fig.add_subplot(1,2,2, projection='3d')\n"
        "ax2.plot_surface(Xh, Yh, Zh, color='#ff7f0e', alpha=0.4, edgecolor='none')\n"
        "a,b,c = hcr_abc\n"
        "Zh_plane = a*Xh + b*Yh + c\n"
        "ax2.plot_surface(Xh, Yh, Zh_plane, color='#333', alpha=0.2, edgecolor='none')\n"
        "ax2.invert_zaxis(); ax2.set_title(f'HCR pia (tilt {np.degrees(np.arccos(n_hcr[2])):.2f}°)'); \n"
        "ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')\n"
        "fig.suptitle('Step 1: pia samples (coloured) with least-squares plane fit (grey)')\n"
        "plt.tight_layout(); plt.show()"
    ))

    # ----- STEP 2 -----
    cells.append(md(
        "## Step 2 — Tilt-aligned rotation\n\n"
        "The acquisition flips CZ by 180° around z w.r.t. HCR. So start with "
        "`R_180 = Rz(−180°)`. After applying it, the CZ normal becomes "
        "`n_cz @ R_180 = (−n_cz[0], −n_cz[1], n_cz[2])`. We then compute a "
        "small rotation `R_tilt` (Rodrigues) that sends this rotated CZ "
        "normal onto the HCR normal. Composed: `R = R_180 · R_tilt`.\n\n"
        "The angle between `n_cz @ R_180` and `n_hcr` is the **residual tilt** — "
        "the part `R_tilt` has to eat."
    ))

    cells.append(code(
        "R_180 = R1._rotation_about_z_row(-180.0)\n"
        "n_cz_rot = n_cz @ R_180\n"
        "R_tilt = R1._rotation_between_row(n_cz_rot, n_hcr)\n"
        "R = R_180 @ R_tilt\n"
        "delta = np.degrees(np.arccos(np.clip(n_cz_rot @ n_hcr, -1, 1)))\n"
        "print(f'n_cz @ R_180 = {n_cz_rot}')\n"
        "print(f'n_hcr        = {n_hcr}')\n"
        "print(f'residual tilt (|Δ|) = {delta:.2f}°')\n"
        "print(f'after R_tilt: angle = {np.degrees(np.arccos(np.clip((n_cz_rot @ R_tilt) @ n_hcr, -1, 1))):.4f}°')"
    ))

    cells.append(code(
        "# Side-view of CZ pia before (R_180 only) and after (R_180·R_tilt), pia-anchored\n"
        "cz_pts = np.column_stack([Xc.ravel(), Yc.ravel(), Zc.ravel()])\n"
        "cz_mean = cz_xyz.mean(axis=0)\n"
        "hcr_mean = hcr_xyz.mean(axis=0)\n"
        "# pia-anchored translation (for visualization only)\n"
        "xy_c = cz_mean[:2]\n"
        "z_cz_pia = R1._surface_z_at(cz_surf, np.array([xy_c[0]]), np.array([xy_c[1]]))[0]\n"
        "c3d = np.array([xy_c[0], xy_c[1], z_cz_pia])\n"
        "def _t_pia(R_):\n"
        "    c_h = (c3d - cz_mean) @ R_\n"
        "    xy_h = c_h[:2] + hcr_mean[:2]\n"
        "    z_h = R1._surface_z_at(hcr_surf, np.array([xy_h[0]]), np.array([xy_h[1]]))[0]\n"
        "    return np.array([hcr_mean[0], hcr_mean[1], z_h - c_h[2]])\n"
        "pre  = (cz_pts - cz_mean) @ R_180 + _t_pia(R_180)\n"
        "post = (cz_pts - cz_mean) @ R     + _t_pia(R)\n"
        "hcr_pts = np.column_stack([Xh.ravel(), Yh.ravel(), Zh.ravel()])\n"
        "\n"
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))\n"
        "for i, a in enumerate([0, 1]):\n"
        "    ax[i].scatter(hcr_pts[:,a], hcr_pts[:,2], s=2, color='#ff7f0e', alpha=0.25, label='HCR pia')\n"
        "    ax[i].scatter(pre[:,a],    pre[:,2],    s=2, color='#aaa',    alpha=0.4, label='CZ: R_180 only')\n"
        "    ax[i].scatter(post[:,a],   post[:,2],   s=2, color='#1f77b4', alpha=0.5, label='CZ: R = R_180·R_tilt')\n"
        "    ax[i].set_xlabel('x (µm)' if a==0 else 'y (µm)')\n"
        "    ax[i].set_ylabel('z (µm)')\n"
        "    ax[i].invert_yaxis()\n"
        "    ax[i].grid(alpha=0.3)\n"
        "    ax[i].set_title(('XZ' if a==0 else 'YZ') + f' side view (Δtilt {delta:.2f}° → 0°)')\n"
        "ax[0].legend(fontsize=8)\n"
        "plt.tight_layout(); plt.show()"
    ))

    # ----- STEP 3 -----
    cells.append(md(
        "## Step 3 — Centroid translation\n\n"
        "We don't have correspondences, so `t` is anchored heuristically: "
        "`t = hcr_mean − R · cz_mean` — equivalently, CZ cell-cloud centroid "
        "lands exactly on HCR GFP+ cell-cloud centroid. This is the affine's "
        "translation vector.\n\n"
        "Note: this matches *cell centroids*, not pia planes. At `sz = 1` "
        "(minimal output, no Z rescale) CZ cells span only ~400 µm while HCR "
        "GFP+ cells span ~1200 µm, so the two pias don't coincide after this "
        "translation — they're offset by the difference in centroid depth."
    ))

    cells.append(code(
        "cz_rot = (cz_xyz - cz_mean) @ R\n"
        "cz_min_out = cz_rot + hcr_mean  # R + centroid-match (minimal output)\n"
        "\n"
        "fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))\n"
        "ax[0].scatter(hcr_xyz[:,0], hcr_xyz[:,1], s=1, color='#ff7f0e', alpha=0.3, label='HCR GFP+')\n"
        "ax[0].scatter(cz_min_out[:,0], cz_min_out[:,1], s=4, color='#1f77b4', alpha=0.8, label='CZ after (R, t)')\n"
        "ax[0].set_aspect('equal'); ax[0].set_xlabel('x (µm)'); ax[0].set_ylabel('y (µm)')\n"
        "ax[0].set_title('XY in HCR frame — CZ cells sit inside HCR')\n"
        "ax[0].legend(); ax[0].grid(alpha=0.3)\n"
        "ax[1].scatter(hcr_xyz[:,0], hcr_xyz[:,2], s=1, color='#ff7f0e', alpha=0.3)\n"
        "ax[1].scatter(cz_min_out[:,0], cz_min_out[:,2], s=4, color='#1f77b4', alpha=0.8)\n"
        "ax[1].invert_yaxis()\n"
        "ax[1].set_xlabel('x (µm)'); ax[1].set_ylabel('z (µm)')\n"
        "ax[1].set_title('XZ — CZ covers only part of HCR axially (sz=1)')\n"
        "ax[1].grid(alpha=0.3)\n"
        "plt.tight_layout(); plt.show()"
    ))

    # ----- STEP 4 -----
    cells.append(md(
        "## Step 4 — Depth-from-surface\n\n"
        "`depth_from_surface` projects each cell onto the pia plane and "
        "returns the signed distance. This gives a **pia-anchored** axial "
        "coordinate that's invariant to buffer thickness and tilt. Used by "
        "step 5 as the 1-D signal to correlate."
    ))

    cells.append(code(
        "cz_depth = R1.depth_from_surface(cz_xyz, cz_surf)\n"
        "hcr_depth = R1.depth_from_surface(hcr_xyz, hcr_surf)\n"
        "print(f'CZ depth: {cz_depth.min():.0f} — {cz_depth.max():.0f} µm '\n"
        "      f'(mean {cz_depth.mean():.0f})')\n"
        "print(f'HCR depth: {hcr_depth.min():.0f} — {hcr_depth.max():.0f} µm '\n"
        "      f'(mean {hcr_depth.mean():.0f})')\n"
        "\n"
        "bins = np.arange(0, 1600, 20)\n"
        "plt.figure(figsize=(9, 3.5))\n"
        "plt.hist(cz_depth,  bins=bins, density=True, color='#1f77b4', alpha=0.6, label=f'CZ (n={len(cz_depth)})')\n"
        "plt.hist(hcr_depth, bins=bins, density=True, color='#ff7f0e', alpha=0.5, label=f'HCR GFP+ (n={len(hcr_depth)})')\n"
        "plt.xlabel('depth from pia (µm)'); plt.ylabel('density')\n"
        "plt.title('Step 4: depth-from-pia histograms — CZ is a narrow axial band; HCR is much longer')\n"
        "plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()"
    ))

    # ----- STEP 5 -----
    cells.append(md(
        "## Step 5 — Z scale + Z offset search (1-D)\n\n"
        "For each candidate `sz` we rescale CZ's depth profile to `sz · cz_depth`, "
        "build a smoothed density on 20-µm bins, then **slide** it along the "
        "HCR density profile. At each shift (bin offset) we compute a Pearson "
        "NCC **on the overlapping region only** with a minimum-overlap floor "
        "(25 % of the longer profile — this is deviation 1 from the Grand Plan, "
        "keyed off the longer profile to avoid 2–3-bin trivial matches).\n\n"
        "For each `sz`, the *best over all shifts* is a point on the score "
        "curve.  The global best (sz*, tz*) is the pair that maximises it."
    ))

    cells.append(code(
        "sz_grid = np.arange(0.5, 6.0 + 0.02, 0.02)\n"
        "z_res = R1._z_scale_offset_search(\n"
        "    cz_depth, hcr_depth,\n"
        "    sz_grid=sz_grid,\n"
        "    depth_bin_um=20.0,\n"
        "    min_overlap_frac=0.25,\n"
        "    depth_max_um=max(hcr_depth.max(), 6.0*cz_depth.max()) + 200.0,\n"
        ")\n"
        "print(f\"sz* = {z_res['sz']:.2f}   tz* = {z_res['tz_um']:.1f} µm   \"\n"
        "      f\"best NCC = {z_res['best_score']:.3f}\")"
    ))

    cells.append(code(
        "# 1D curve: max-over-tz NCC vs sz\n"
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
        "curve = np.asarray(z_res['score_curve'])\n"
        "m = np.isfinite(curve)\n"
        "ax[0].plot(sz_grid[m], curve[m], color='#1f77b4')\n"
        "ax[0].axvline(z_res['sz'], color='red', ls='--', label=f\"sz*={z_res['sz']:.2f}\")\n"
        "ax[0].set_xlabel('sz'); ax[0].set_ylabel('max-over-tz NCC')\n"
        "ax[0].set_title('Step 5: per-sz best partial-overlap NCC')\n"
        "ax[0].legend(); ax[0].grid(alpha=0.3)\n"
        "\n"
        "# Illustrative overlay: CZ profile at sz=sz* aligned to HCR profile at tz*\n"
        "edges = np.arange(0, max(hcr_depth.max(), z_res['sz']*cz_depth.max()) + 40, 20.0)\n"
        "hcr_prof = R1._density_profile_1d(hcr_depth, edges, 1.0)\n"
        "cz_scaled = cz_depth * z_res['sz']\n"
        "tem_edges = np.arange(max(0, cz_scaled.min()), cz_scaled.max() + 20, 20.0)\n"
        "cz_prof = R1._density_profile_1d(cz_scaled, tem_edges, 1.0)\n"
        "xs_h = 0.5*(edges[:-1] + edges[1:])\n"
        "xs_c = 0.5*(tem_edges[:-1] + tem_edges[1:]) + z_res['tz_um']\n"
        "ax[1].plot(xs_h, hcr_prof / max(hcr_prof.max(), 1e-9),  color='#ff7f0e', label='HCR density')\n"
        "ax[1].plot(xs_c, cz_prof / max(cz_prof.max(), 1e-9), color='#1f77b4', label=f'CZ × sz={z_res[\"sz\"]:.2f} at tz={z_res[\"tz_um\"]:.0f}')\n"
        "ax[1].set_xlabel('depth from pia (µm)'); ax[1].set_ylabel('normalised density')\n"
        "ax[1].set_title('Aligned depth profiles at (sz*, tz*)')\n"
        "ax[1].legend(); ax[1].grid(alpha=0.3)\n"
        "plt.tight_layout(); plt.show()"
    ))

    # ----- STEP 6 -----
    cells.append(md(
        "## Step 6 — XY scale + XY translation search (2-D)\n\n"
        "For each `sxy ∈ [0.5, L_hcr/L_cz]` we scale CZ's rotated XY, build "
        "a Gaussian-blurred density map (σ=30 µm, 20-µm bins), and run a "
        "**windowed 2-D Pearson NCC** (integral-image, Lewis 1995) against "
        "HCR's density map. Best `(sxy, tx, ty)` across the grid.\n\n"
        "The feasibility cap `sxy ≤ L_hcr/L_cz` is the Grand Plan's "
        "\"CZ ⊂ HCR in XY\" prior — it's a geometric fact that a 400 µm "
        "CZ patch can't be scaled up past HCR's ~2 mm envelope."
    ))

    cells.append(code(
        "L_cz  = float(max(cz_rot[:,0].ptp(), cz_rot[:,1].ptp()))\n"
        "L_hcr = float(max(hcr_xyz[:,0].ptp(), hcr_xyz[:,1].ptp()))\n"
        "sxy_grid = np.arange(0.5, (L_hcr/L_cz) + 0.05, 0.05)\n"
        "xy_res = R1._xy_scale_translation_search(\n"
        "    cz_rot[:, :2], hcr_xyz[:, :2],\n"
        "    sxy_grid=sxy_grid,\n"
        "    xy_bin_um=20.0, xy_sigma_um=30.0, margin_um=100.0,\n"
        ")\n"
        "print(f'L_cz = {L_cz:.0f} µm   L_hcr = {L_hcr:.0f} µm   '\n"
        "      f'=> sxy_upper = {L_hcr/L_cz:.2f}')\n"
        "print(f\"sxy* = {xy_res['sxy']:.2f}   (tx, ty) = ({xy_res['tx_um']:.0f}, {xy_res['ty_um']:.0f}) µm\")\n"
        "print(f\"best 2D-NCC = {xy_res['best_score']:.3f}\")"
    ))

    cells.append(code(
        "# Show 2D density maps (HCR, and CZ at best sxy)\n"
        "hx = np.arange(hcr_xyz[:,0].min()-100, hcr_xyz[:,0].max()+100+20, 20)\n"
        "hy = np.arange(hcr_xyz[:,1].min()-100, hcr_xyz[:,1].max()+100+20, 20)\n"
        "hcr_map = R1._density_map_2d(hcr_xyz[:, :2], hx, hy, 30/20)\n"
        "cz_scaled_xy = cz_rot[:, :2] * xy_res['sxy']\n"
        "cx = np.arange(cz_scaled_xy[:,0].min()-100, cz_scaled_xy[:,0].max()+100+20, 20)\n"
        "cy = np.arange(cz_scaled_xy[:,1].min()-100, cz_scaled_xy[:,1].max()+100+20, 20)\n"
        "cz_map = R1._density_map_2d(cz_scaled_xy, cx, cy, 30/20)\n"
        "\n"
        "fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))\n"
        "ax[0].imshow(hcr_map.T, origin='lower',\n"
        "             extent=[hx[0], hx[-1], hy[0], hy[-1]], cmap='Oranges')\n"
        "ax[0].set_title('HCR GFP+ density map')\n"
        "ax[0].set_xlabel('x (µm)'); ax[0].set_ylabel('y (µm)')\n"
        "ax[1].imshow(cz_map.T, origin='lower',\n"
        "             extent=[cx[0], cx[-1], cy[0], cy[-1]], cmap='Blues')\n"
        "ax[1].set_title(f'CZ density map at sxy*={xy_res[\"sxy\"]:.2f}')\n"
        "ax[1].set_xlabel('x (µm)'); ax[1].set_ylabel('y (µm)')\n"
        "plt.tight_layout(); plt.show()"
    ))

    cells.append(code(
        "# 1D score curve: max-over-(tx,ty) 2D-NCC vs sxy\n"
        "curve = np.asarray(xy_res['score_curve'])\n"
        "m = np.isfinite(curve)\n"
        "plt.figure(figsize=(8, 3.8))\n"
        "plt.plot(sxy_grid[m], curve[m], color='#2ca02c')\n"
        "plt.axvline(xy_res['sxy'], color='red', ls='--', label=f\"sxy*={xy_res['sxy']:.2f}\")\n"
        "plt.xlabel('sxy'); plt.ylabel('max-over-(tx,ty) 2D-NCC')\n"
        "plt.title('Step 6: XY-scale score curve — peak at small sxy is the \"shrink-to-fit\" failure')\n"
        "plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()"
    ))

    # ----- STEP 7 & 8 -----
    cells.append(md(
        "## Step 7 — Anisotropic refinement (skipped when sxy isn't confident)\n\n"
        "Local grid around `sxy*` over `(sx, sy)` to allow axis-specific scales. "
        "Only runs when the XY scale would have been emitted; on 788406 the "
        "confidence check below rejects it, so this step is a no-op here."
    ))

    cells.append(md(
        "## Step 8 — Scale confidence gating\n\n"
        "For each of the Z and XY score curves, compute the **robust-z** prominence\n\n"
        "`conf = (peak − median) / (1.4826 · MAD)`\n\n"
        "and emit the scale only when `conf ≥ 6.0`. At threshold 3, the XY "
        "curve's sharp small-sxy peak would clear the bar (conf ≈ 5.07 on "
        "788406) despite being geometrically wrong. Raising to 6 routes every "
        "subject to the minimal-output fallback, matching the Grand Plan §R1 "
        "step 8 graceful-degradation intent."
    ))

    cells.append(code(
        "sz_conf  = R1._peak_to_rms(z_res['score_curve'])\n"
        "sxy_conf = R1._peak_to_rms(xy_res['score_curve'])\n"
        "print(f'sz  confidence = {sz_conf:.2f}   →  emit sz?  {sz_conf  >= 6.0}')\n"
        "print(f'sxy confidence = {sxy_conf:.2f}   →  emit sxy? {sxy_conf >= 6.0}')\n"
        "\n"
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
        "for a, (grid, curve, best, conf, color, label) in zip(\n"
        "    ax,\n"
        "    [(sz_grid, np.asarray(z_res['score_curve']), z_res['sz'], sz_conf, '#1f77b4', 'sz'),\n"
        "     (sxy_grid, np.asarray(xy_res['score_curve']), xy_res['sxy'], sxy_conf, '#2ca02c', 'sxy')]):\n"
        "    m = np.isfinite(curve)\n"
        "    med = np.median(curve[m]); mad = np.median(np.abs(curve[m] - med))\n"
        "    a.plot(grid[m], curve[m], color=color)\n"
        "    a.axhline(med, color='#888', lw=1, label='median')\n"
        "    a.axhline(med + 6*1.4826*mad, color='red', lw=1, ls=':', label='median + 6·1.4826·MAD (threshold)')\n"
        "    a.axvline(best, color='red', lw=1, ls='--', label=f'{label}*={best:.2f} (conf {conf:.2f})')\n"
        "    a.set_xlabel(label); a.set_ylabel('NCC')\n"
        "    a.set_title(f'Step 8: {label} confidence gate')\n"
        "    a.legend(fontsize=8); a.grid(alpha=0.3)\n"
        "plt.tight_layout(); plt.show()"
    ))

    # ----- FINAL -----
    cells.append(md(
        "## Full output — apply R1 to the CZ cloud\n\n"
        "Both scales fail the 6.0 bar, so the emitted affine is "
        "`(R, S=[1,1,1], t = hcr_mean − R·cz_mean)` — the minimal output. "
        "Here's the final alignment."
    ))

    cells.append(code(
        "fit = R1.coarse_align_revised(cz_xyz, hcr_xyz, cz_surf, hcr_surf)\n"
        "print(fit)\n"
        "cz_in_hcr = R1.apply_coarse_affine(cz_xyz, fit)\n"
        "\n"
        "fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))\n"
        "ax[0].scatter(hcr_xyz[:,0], hcr_xyz[:,1], s=1, color='#ff7f0e', alpha=0.3, label='HCR GFP+')\n"
        "ax[0].scatter(cz_in_hcr[:,0], cz_in_hcr[:,1], s=4, color='#1f77b4', alpha=0.8, label='CZ after R1')\n"
        "ax[0].set_aspect('equal'); ax[0].set_xlabel('x (µm)'); ax[0].set_ylabel('y (µm)')\n"
        "ax[0].set_title('Final XY alignment'); ax[0].legend(); ax[0].grid(alpha=0.3)\n"
        "ax[1].scatter(hcr_xyz[:,0], hcr_xyz[:,2], s=1, color='#ff7f0e', alpha=0.3)\n"
        "ax[1].scatter(cz_in_hcr[:,0], cz_in_hcr[:,2], s=4, color='#1f77b4', alpha=0.8)\n"
        "ax[1].invert_yaxis(); ax[1].set_xlabel('x (µm)'); ax[1].set_ylabel('z (µm)')\n"
        "ax[1].set_title('Final XZ alignment (minimal output, sz=1)')\n"
        "ax[1].grid(alpha=0.3)\n"
        "plt.tight_layout(); plt.show()"
    ))

    cells.append(md(
        "### Summary for subject 788406\n\n"
        "- Rotation: **−179.92°** (prior + small tilt correction, 2° Δtilt).\n"
        "- Translation: **cell-cloud centroid match** (minimal output).\n"
        "- sz* = 3.02 (GT 2.82 — close), but **confidence 1.81 < 6**, so not "
        "emitted.\n"
        "- sxy* = 0.75 (GT 1.77 — wrong, shrink-to-fit failure), **confidence "
        "5.07 < 6**, so not emitted.\n"
        "- Result: minimal `(R, t)` with `scales = [1, 1, 1]`, "
        "`scale_known = [F, F, F]`. Origin error vs ground-truth affine: "
        "**133 µm**; rotation error: **2.38°**.\n"
        "- Interpretation: R and t work; scale search is unreliable on this "
        "data (volume-in-volume containment is not enforced by the current "
        "objective). Grand Plan §R1 step 8 graceful-degradation route."
    ))

    nb["cells"] = cells
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
