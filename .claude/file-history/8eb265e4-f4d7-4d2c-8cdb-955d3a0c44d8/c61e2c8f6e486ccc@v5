"""Session 07e driver — sxy from CZ↔HCR ROI xy-area ratio.

All bbox / scale logic lives in the promoted module `roi_area_sxy`.
This file only contains session-specific outputs:
  - per-subject `roi_area_{sid}.png` histogram + area-vs-depth figure
  - `sessions/07e_sz_from_zvar_profile/roi_area_sxy.json` results dump
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

from roi_area_sxy import (
    D_SKIN_UM,
    SPOT_SUBJECTS,
    _prefilter_center_fov,
    analyze_subject,
    cz_cell_tight_bboxes,
    fit_anisotropic_similarity,
    hcr_cell_tight_bboxes,
    landmark_pairs_um,
    load_subject,
)
import importlib.util
_gfp_thr_spec = importlib.util.spec_from_file_location(
    "gfp_thr", THIS / "07b_gfp_intersection_threshold.py"
)
gfp_thr = importlib.util.module_from_spec(_gfp_thr_spec)
sys.modules["gfp_thr"] = gfp_thr
_gfp_thr_spec.loader.exec_module(gfp_thr)  # type: ignore

SESSION = Path("/root/capsule/code/sessions/07e_sz_from_zvar_profile")
FIG_DIR = SESSION / "figures_roi_area"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_subject(sid, cz_df, hcr_df, sxy_est, sxy_gt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    lo = min(cz_df.xy_area_um2.min(), hcr_df.xy_area_um2.min())
    hi = max(cz_df.xy_area_um2.max(), hcr_df.xy_area_um2.max())
    bins = np.logspace(np.log10(max(lo, 1)), np.log10(hi), 50)
    ax.hist(cz_df.xy_area_um2, bins=bins, alpha=0.55, color='#268bd2',
            label=f'CZ (n={len(cz_df)}, med={cz_df.xy_area_um2.median():.0f})',
            density=True)
    ax.hist(hcr_df.xy_area_um2, bins=bins, alpha=0.55, color='#cb4b16',
            label=f'HCR GFP+ (n={len(hcr_df)}, med={hcr_df.xy_area_um2.median():.0f})',
            density=True)
    ax.set_xscale('log')
    ax.set_xlabel('bbox xy area (µm²)')
    ax.set_ylabel('density')
    ax.set_title(f'{sid}  sxy_est={sxy_est:.3f}  GT={sxy_gt:.3f}  '
                 f'err={100*(sxy_est-sxy_gt)/sxy_gt:+.1f}%')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    depth_bins = np.arange(
        D_SKIN_UM, max(cz_df.depth_um.max(), hcr_df.depth_um.max()) + 50, 50
    )
    cz_d = cz_df.depth_um.to_numpy()
    hcr_d = hcr_df.depth_um.to_numpy()
    cz_med = [np.median(cz_df.xy_area_um2[(cz_d >= d) & (cz_d < d + 50)])
              for d in depth_bins]
    hcr_med = [np.median(hcr_df.xy_area_um2[(hcr_d >= d) & (hcr_d < d + 50)])
               for d in depth_bins]
    ax.plot(depth_bins, cz_med, 'o-', color='#268bd2', label='CZ median', lw=1.5)
    ax.plot(depth_bins, hcr_med, 's-', color='#cb4b16', label='HCR median', lw=1.5)
    cz_arr = np.array(cz_med, dtype=float)
    expected_hcr = cz_arr * sxy_gt ** 2
    ax.plot(depth_bins, expected_hcr, 'k--', lw=1,
            label='CZ×sxy²_GT', alpha=0.6)
    ax.set_yscale('log')
    ax.set_xlabel('pia depth (µm)')
    ax.set_ylabel('median xy bbox area (µm²)')
    ax.set_title(f'{sid} — area vs depth')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / f'roi_area_{sid}.png'
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


def analyze(sid: str) -> dict:
    if sid not in SPOT_SUBJECTS:
        raise ValueError(f"{sid}: intensity/unsupported subject")
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_surf, hcr_surf = info["cz_surface"], info["hcr_surface"]

    gi = gfp_thr.analyze_subject(sid)
    strict_cutoff = float(gi.cutoff_linear)
    strict_df = gfp_thr.strict_gfp_df(sid, strict_cutoff)
    strict_ids = set(int(x) for x in strict_df["hcr_id"].values)
    fov_ids = _prefilter_center_fov(sid, s, strict_ids)
    print(f"  [{sid}] strict BIC={gi.n_components} n_total={len(strict_ids)} "
          f"n_center_1/4_FOV={len(fov_ids)}")

    cz_df_all = cz_cell_tight_bboxes(sid, s, cz_surf)
    hcr_df_all = hcr_cell_tight_bboxes(sid, s, hcr_surf, fov_ids)

    cz_span = float(np.nanpercentile(cz_df_all.depth_um, 99))
    hcr_span = float(np.nanpercentile(hcr_df_all.depth_um, 99))

    cz_mask = (cz_df_all.depth_um >= D_SKIN_UM) & (cz_df_all.depth_um <= cz_span)
    hcr_mask = (hcr_df_all.depth_um >= D_SKIN_UM) & (hcr_df_all.depth_um <= hcr_span)
    cz_df = cz_df_all[cz_mask].copy()
    hcr_df = hcr_df_all[hcr_mask].copy()

    med_cz = float(cz_df.xy_area_um2.median())
    med_hcr = float(hcr_df.xy_area_um2.median())
    mean_cz = float(cz_df.xy_area_um2.mean())
    mean_hcr = float(hcr_df.xy_area_um2.mean())
    sxy_med = float(np.sqrt(med_hcr / med_cz))
    sxy_mean = float(np.sqrt(mean_hcr / mean_cz))

    fit = fit_anisotropic_similarity(*landmark_pairs_um(s, active_only=True))
    sxy_gt = float(np.sqrt(fit.scales[0] * fit.scales[1]))

    fig = plot_subject(sid, cz_df, hcr_df, sxy_med, sxy_gt)

    return {
        "sid": sid,
        "n_cz_total": int(len(cz_df_all)), "n_cz_used": int(len(cz_df)),
        "n_hcr_strict_total": int(len(hcr_df_all)), "n_hcr_used": int(len(hcr_df)),
        "bic_cutoff_density": strict_cutoff,
        "bic_n_components": int(gi.n_components),
        "cz_span": cz_span, "hcr_span": hcr_span,
        "cz_area_med": med_cz, "hcr_area_med": med_hcr,
        "cz_area_mean": mean_cz, "hcr_area_mean": mean_hcr,
        "sxy_from_median": sxy_med,
        "sxy_from_mean": sxy_mean,
        "sxy_gt": sxy_gt,
        "err_med_pct": 100 * (sxy_med - sxy_gt) / sxy_gt,
        "err_mean_pct": 100 * (sxy_mean - sxy_gt) / sxy_gt,
        "figure": fig,
    }


if __name__ == "__main__":
    sids = sys.argv[1:] or sorted(SPOT_SUBJECTS)
    out = {}
    for sid in sids:
        print(f"── {sid} ──")
        try:
            r = analyze(sid)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue
        print(f"  BIC K={r['bic_n_components']} cutoff={r['bic_cutoff_density']:.4g}  "
              f"n_CZ={r['n_cz_used']}/{r['n_cz_total']}  "
              f"n_HCR_strict={r['n_hcr_used']}/{r['n_hcr_strict_total']}  "
              f"CZ_span={r['cz_span']:.0f}  HCR_span={r['hcr_span']:.0f}")
        print(f"  med area: CZ={r['cz_area_med']:.0f} HCR={r['hcr_area_med']:.0f} µm²  "
              f"→ sxy={r['sxy_from_median']:.3f}  GT={r['sxy_gt']:.3f}  "
              f"err {r['err_med_pct']:+.1f}%")
        print(f"  mean area: CZ={r['cz_area_mean']:.0f} HCR={r['hcr_area_mean']:.0f} µm²  "
              f"→ sxy={r['sxy_from_mean']:.3f}  err {r['err_mean_pct']:+.1f}%")
        print(f"  fig → {r['figure']}")
        out[sid] = r

    with open(SESSION / "roi_area_sxy.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {SESSION/'roi_area_sxy.json'}")
