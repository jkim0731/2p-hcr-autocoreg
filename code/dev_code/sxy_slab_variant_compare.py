"""Compare the current center-¼-FOV sxy estimator to a new slab-restricted variant.

Current (baseline, production): ``estimate_sxy_roi_area`` logic with
``center_fov_quarter=True``, area_mode='max_xsection'.
- HCR: strict-GFP+ cells, depth ∈ [D_SKIN_UM, p99], center-¼ of HCR FOV.
- CZ : all cells, depth ∈ [D_SKIN_UM, p99_cz].

New variant — "slab + full FOV":
- HCR: same strict-GFP+ cells, depth ∈ [HCR_SLAB_LO, HCR_SLAB_HI] = [0, 100] µm
  below HCR pia, FULL FOV (no center-¼ restriction).
- CZ : all cells, depth ∈ [CZ_SLAB_LO, CZ_SLAB_HI] = [0, 50] µm below CZ pia.
- The slab bounds match surface_registration_v2's HCR_SLAB and CZ_SLAB exactly —
  the same tissue band used to build the top-slab MIP.
- Area: max_xsection (same as production default).

Two sxy estimates from the variant:
    (a) sxy_slab_median : sqrt(median(area_HCR_slab) / median(area_CZ_slab))
    (b) sxy_slab_total  : sqrt(sum(area_HCR_slab)   / sum(area_CZ_slab))

SPEED:
- HCR depths use ``s.hcr_centroids`` (level-2 parquet; no zarr I/O) so depth
  filtering is fast, and max_xsection is only fetched for slab cells.
- Baseline depths use ``hcr_cell_tight_bboxes`` for the center-¼ subset (same
  as the production path), but we pass the already-known center-¼ ids to avoid
  computing tight bboxes for cells we don't need.

Run:
    python -u sxy_slab_variant_compare.py 2>&1 | tee /tmp/sxy_slab_run.log

Writes results to sxy_slab_variant_2026-06-02.md.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
DEV = Path("/root/capsule/code/dev_code")
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

DATA_RO = Path("/root/capsule/data/claude-data_ophys-mfish-autocoreg_260503")
for _p in (
    DATA_RO / "full_automatic_execution_01/lib",
    DATA_RO / "full_automatic_execution_02/lib",
    DATA_RO / "sessions/08_surface_vascular_match",
    DATA_RO / "sessions/03c_onset_features/iterations",
):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# stub out surface_registration_v2's heavy imports — we only need its constants
for _mod in ("compare_binarization", "register_binary", "register_nonrigid_variants"):
    try:
        __import__(_mod)
    except Exception:
        sys.modules[_mod] = types.SimpleNamespace()  # type: ignore[assignment]

import numpy as np

from benchmark_analysis import depth_from_surface, fit_anisotropic_similarity
from benchmark_data_loader import (
    HCR_SEG_XY_DOWNSAMPLE,
    landmark_pairs_um,
    load_subject,
    hcr_px_to_um,
)
from roi_area_sxy import (
    D_SKIN_UM,
    cz_cell_max_xsection,
    compute_max_xsection_hcr,
    compute_tight_hcr_bboxes,
    hcr_cell_tight_bboxes,
    _prefilter_center_fov,
    _load_hcr_metrics,
)
from surface_registration_v2 import CZ_SLAB, HCR_SLAB
from surfaces_iter08 import get_cz_surface_iter08, get_hcr_top_surface_iter07

# Slab bounds from surface_registration_v2.
HCR_SLAB_LO, HCR_SLAB_HI = HCR_SLAB  # (0.0, 100.0) µm below HCR pia
CZ_SLAB_LO, CZ_SLAB_HI = CZ_SLAB    # (0.0,  50.0) µm below CZ  pia

ALL_SUBJECTS = ("755252", "767018", "767022", "782149", "788406", "790322")


def _get_strict_hcr_ids(sid: str) -> set[int]:
    """Return strict-GFP+ hcr_ids from BIC-GMM analysis."""
    spec = importlib.util.spec_from_file_location(
        "_gfp_thr", DEV / "07b_gfp_intersection_threshold.py"
    )
    _gfp_thr = importlib.util.module_from_spec(spec)
    sys.modules["_gfp_thr"] = _gfp_thr
    spec.loader.exec_module(_gfp_thr)  # type: ignore
    gi = _gfp_thr.analyze_subject(sid)
    strict_df = _gfp_thr.strict_gfp_df(sid, float(gi.cutoff_linear))
    return set(int(x) for x in strict_df["hcr_id"].values)


def _hcr_depths_from_centroids(sid: str, s, hcr_surf: dict, hcr_ids: set) -> dict[int, float]:
    """Compute pia depth for each hcr_id using ``s.hcr_centroids`` (level-2).

    Fast path: no zarr I/O — reads from the pre-loaded centroid parquet.
    Level-2 pixel frame → hcr_px_to_um → depth_from_surface.

    Returns {hcr_id: depth_um}.
    """
    cent = s.hcr_centroids  # DataFrame with hcr_id, z_px, y_px, x_px (level-2)
    mask = cent["hcr_id"].astype(int).isin(hcr_ids)
    sub = cent[mask].copy()
    if sub.empty:
        return {}
    zyx = sub[["z_px", "y_px", "x_px"]].to_numpy(float)
    xyz_um = hcr_px_to_um(zyx, s)[:, [2, 1, 0]]  # (N, 3): x, y, z µm
    depths = depth_from_surface(xyz_um, hcr_surf)
    ids = sub["hcr_id"].to_numpy(int)
    return dict(zip(ids, depths))


def compare_one(sid: str) -> dict:
    """Run both baseline and slab-variant for one subject.

    Approach for HCR slab filtering:
    1. Get depths for all strict-GFP+ ids via ``s.hcr_centroids`` (level-2,
       no zarr I/O).
    2. Filter to slab depth → slab_ids.
    3. Compute max_xsection only for slab_ids (cache-backed; typically small).
    4. For baseline, use _prefilter_center_fov on strict_ids → center_ids,
       then hcr_cell_tight_bboxes for those (same as production path).
    """
    print(f"  [{sid}] load_subject + surfaces...", flush=True)
    s = load_subject(sid)
    cz_surf  = get_cz_surface_iter08(s)
    hcr_surf = get_hcr_top_surface_iter07(s)

    print(f"  [{sid}] GFP+ threshold...", flush=True)
    strict_ids = _get_strict_hcr_ids(sid)
    n_strict = len(strict_ids)
    print(f"  [{sid}] n_strict={n_strict}", flush=True)

    # GT
    fit_gt = fit_anisotropic_similarity(*landmark_pairs_um(s, active_only=True))
    sxy_gt = float(np.sqrt(fit_gt.scales[0] * fit_gt.scales[1]))

    # ── CZ areas (shared) ────────────────────────────────────────────────────
    print(f"  [{sid}] CZ max_xsection...", flush=True)
    cz_df_all = cz_cell_max_xsection(sid, s, cz_surf)

    # ── HCR depths from centroids (fast — no zarr) ───────────────────────────
    print(f"  [{sid}] HCR depths from centroids (level-2, no zarr)...", flush=True)
    id_to_depth = _hcr_depths_from_centroids(sid, s, hcr_surf, strict_ids)
    n_found = len(id_to_depth)
    print(f"  [{sid}] depth lookup: {n_found}/{n_strict} ids found in hcr_centroids",
          flush=True)

    # ── SLAB VARIANT: full FOV, slab depth ───────────────────────────────────
    slab_ids = {i for i, d in id_to_depth.items()
                if HCR_SLAB_LO <= d <= HCR_SLAB_HI}
    n_hcr_slab = len(slab_ids)
    print(f"  [{sid}] slab ids [{HCR_SLAB_LO:.0f}–{HCR_SLAB_HI:.0f} µm]: {n_hcr_slab}",
          flush=True)

    if n_hcr_slab > 0:
        print(f"  [{sid}] max_xsection for slab ids...", flush=True)
        mx = compute_max_xsection_hcr(sid, slab_ids, verbose=True)
        area_hcr_slab = mx["max_xsection_pix"].to_numpy(float) * (s.hcr_seg_xy_um ** 2)
    else:
        area_hcr_slab = np.array([])

    cz_slab_mask = (
        (cz_df_all["depth_um"] >= CZ_SLAB_LO)
        & (cz_df_all["depth_um"] <= CZ_SLAB_HI)
    )
    cz_slab = cz_df_all[cz_slab_mask].copy()
    n_cz_slab = int(len(cz_slab))
    area_cz_slab = cz_slab["xy_area_um2"].to_numpy(float)
    print(f"  [{sid}] CZ slab n={n_cz_slab}", flush=True)

    if n_hcr_slab > 0 and n_cz_slab > 0:
        sxy_slab_median = float(np.sqrt(np.median(area_hcr_slab) / np.median(area_cz_slab)))
        sxy_slab_total  = float(np.sqrt(area_hcr_slab.sum() / area_cz_slab.sum()))
    else:
        print(f"  [{sid}] WARNING: insufficient slab cells", flush=True)
        sxy_slab_median = float("nan")
        sxy_slab_total  = float("nan")

    err_sm = (100.0 * (sxy_slab_median - sxy_gt) / sxy_gt
              if not np.isnan(sxy_slab_median) else float("nan"))
    err_st = (100.0 * (sxy_slab_total  - sxy_gt) / sxy_gt
              if not np.isnan(sxy_slab_total) else float("nan"))
    print(
        f"  [{sid}] SLAB: n_hcr={n_hcr_slab}, n_cz={n_cz_slab}, "
        f"sxy_median={sxy_slab_median:.4f} ({err_sm:+.2f}%), "
        f"sxy_total={sxy_slab_total:.4f} ({err_st:+.2f}%), GT={sxy_gt:.4f}",
        flush=True,
    )

    # ── BASELINE: center-¼ FOV, depth ∈ [D_SKIN_UM, p99] ────────────────────
    # Filter to center-¼ using fast bbox-center estimate.
    fov_ids = _prefilter_center_fov(sid, s, strict_ids)
    print(f"  [{sid}] center-¼ ids: {len(fov_ids)}", flush=True)

    # Use tight bboxes for baseline (for depth from actual zarr centroids).
    # Only compute for the center-¼ subset (same as production path).
    print(f"  [{sid}] HCR tight bboxes for center-¼ subset...", flush=True)
    hcr_base_df = hcr_cell_tight_bboxes(
        sid, s, hcr_surf, fov_ids, area_mode="max_xsection"
    )
    hcr_span = float(np.nanpercentile(hcr_base_df["depth_um"], 99))
    hcr_depth_mask = (
        (hcr_base_df["depth_um"] >= D_SKIN_UM)
        & (hcr_base_df["depth_um"] <= hcr_span)
    )
    hcr_base = hcr_base_df[hcr_depth_mask].copy()
    n_hcr_current = int(len(hcr_base))

    cz_span = float(np.nanpercentile(cz_df_all["depth_um"], 99))
    cz_base_mask = (
        (cz_df_all["depth_um"] >= D_SKIN_UM)
        & (cz_df_all["depth_um"] <= cz_span)
    )
    cz_base = cz_df_all[cz_base_mask].copy()
    n_cz_current = int(len(cz_base))

    area_hcr_base = hcr_base["xy_area_um2"].to_numpy(float)
    area_cz_base  = cz_base["xy_area_um2"].to_numpy(float)

    if len(area_hcr_base) == 0 or len(area_cz_base) == 0:
        sxy_current = float("nan")
        err_current = float("nan")
    else:
        sxy_current = float(np.sqrt(np.median(area_hcr_base) / np.median(area_cz_base)))
        err_current = 100.0 * (sxy_current - sxy_gt) / sxy_gt

    print(
        f"  [{sid}] BASELINE: n_hcr={n_hcr_current} (center-¼+depth), "
        f"n_cz={n_cz_current}, sxy={sxy_current:.4f} ({err_current:+.2f}%)",
        flush=True,
    )

    return {
        "sid": sid,
        "sxy_current": sxy_current,
        "sxy_slab_median": sxy_slab_median,
        "sxy_slab_total":  sxy_slab_total,
        "sxy_gt": sxy_gt,
        "err_current_pct":     err_current,
        "err_slab_median_pct": err_sm,
        "err_slab_total_pct":  err_st,
        "n_hcr_current": n_hcr_current,
        "n_hcr_slab":    n_hcr_slab,
        "n_cz_current":  n_cz_current,
        "n_cz_slab":     n_cz_slab,
        "n_strict": n_strict,
    }


def run_comparison() -> list:
    rows = []
    for sid in ALL_SUBJECTS:
        print(f"\n=== {sid} ===", flush=True)
        row = compare_one(sid)
        rows.append(row)
    return rows


def format_table(rows: list) -> str:
    hdr = (
        f"{'sid':<8}  {'sxy_cur':>8}  {'sxy_slab_med':>12}  {'sxy_slab_tot':>12}  "
        f"{'GT':>7}  {'err_cur%':>9}  {'err_med%':>9}  {'err_tot%':>9}  "
        f"{'n_hcr_cur':>9}  {'n_hcr_slb':>9}  {'n_cz_cur':>8}  {'n_cz_slb':>8}"
    )
    sep = "-" * len(hdr)
    lines = [hdr, sep]
    for r in rows:
        sm = r["sxy_slab_median"]
        st = r["sxy_slab_total"]
        em = r["err_slab_median_pct"]
        et = r["err_slab_total_pct"]
        lines.append(
            f"{r['sid']:<8}  {r['sxy_current']:>8.4f}  "
            f"{sm:>12.4f}  {st:>12.4f}  {r['sxy_gt']:>7.4f}  "
            f"{r['err_current_pct']:>+9.2f}  {em:>+9.2f}  {et:>+9.2f}  "
            f"{r['n_hcr_current']:>9d}  {r['n_hcr_slab']:>9d}  "
            f"{r['n_cz_current']:>8d}  {r['n_cz_slab']:>8d}"
        )
    lines.append(sep)

    def _mae(key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(np.abs(vals))) if vals else float("nan")

    mae_cur = _mae("err_current_pct")
    mae_sm  = _mae("err_slab_median_pct")
    mae_st  = _mae("err_slab_total_pct")
    lines.append(
        f"{'mean |err|':<8}  {'':>8}  {'':>12}  {'':>12}  {'':>7}  "
        f"{mae_cur:>+9.2f}  {mae_sm:>+9.2f}  {mae_st:>+9.2f}"
    )
    return "\n".join(lines)


def write_report(rows: list, path: Path):
    import json

    def _mae(key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(np.abs(vals))) if vals else float("nan")

    mae_cur = _mae("err_current_pct")
    mae_sm  = _mae("err_slab_median_pct")
    mae_st  = _mae("err_slab_total_pct")

    r782 = next(r for r in rows if r["sid"] == "782149")

    winner = "tie"
    if mae_cur < mae_sm and mae_cur < mae_st:
        winner = "baseline (center-¼ FOV)"
    elif mae_sm <= mae_cur and mae_sm <= mae_st:
        winner = "slab_median"
    elif mae_st <= mae_cur and mae_st <= mae_sm:
        winner = "slab_total"

    lines = [
        "# sxy slab-variant comparison — 2026-06-02",
        "",
        "## Setup",
        "",
        "**Baseline** (`sxy_current`): `estimate_sxy_roi_area` logic with "
        "`center_fov_quarter=True`, `area_mode='max_xsection'`  ",
        f"HCR: strict-GFP+ cells, depth ∈ [{D_SKIN_UM:.0f} µm, p99], center-¼ HCR FOV.  ",
        "CZ: all cells, depth ∈ [D_SKIN_UM, p99_cz].",
        "",
        "**New variant** (slab-restricted, full FOV):  ",
        f"- HCR: strict-GFP+ cells (same population), depth ∈ [{HCR_SLAB_LO:.0f}, {HCR_SLAB_HI:.0f}] µm below HCR pia, **full FOV** (no center-¼ restriction)  ",
        f"- CZ : all cells, depth ∈ [{CZ_SLAB_LO:.0f}, {CZ_SLAB_HI:.0f}] µm below CZ pia  ",
        "- Area: max_xsection (same as baseline)  ",
        "- Slab bounds = `surface_registration_v2.HCR_SLAB` / `CZ_SLAB`  ",
        "- HCR depths for slab filter from `s.hcr_centroids` (level-2 parquet, fast)",
        "",
        "Two sxy estimates from the variant:  ",
        "- `sxy_slab_median`: sqrt(median(area_HCR_slab) / median(area_CZ_slab))  ",
        "- `sxy_slab_total` : sqrt(sum(area_HCR_slab)   / sum(area_CZ_slab))  ",
        "",
        "## Results table",
        "",
        "```",
        format_table(rows),
        "```",
        "",
        "## Summary statistics",
        "",
        f"| Method | Mean |%err| |",
        f"|--------|------------|",
        f"| Baseline (center-¼ FOV) | **{mae_cur:.2f}%** |",
        f"| Slab-restricted median  | **{mae_sm:.2f}%** |",
        f"| Slab-restricted total   | **{mae_st:.2f}%** |",
        "",
        f"**Winner**: {winner}",
        "",
        "## 782149 detail (sparse top-slab HCR, surface-MIP NCC=0.115)",
        "",
        f"- n_strict (all strict-GFP+ before spatial/depth filter): {r782['n_strict']}  ",
        f"- n_hcr_slab (full FOV, 0–100 µm): **{r782['n_hcr_slab']}**  ",
        f"- n_hcr_current (center-¼ FOV + depth baseline): {r782['n_hcr_current']}  ",
        f"- n_cz_slab (0–50 µm): {r782['n_cz_slab']}  ",
        f"- n_cz_current: {r782['n_cz_current']}  ",
        f"- sxy_current = {r782['sxy_current']:.4f} (err {r782['err_current_pct']:+.2f}%)  ",
        f"- sxy_slab_median = {r782['sxy_slab_median']:.4f} (err {r782['err_slab_median_pct']:+.2f}%)  ",
        f"- sxy_slab_total  = {r782['sxy_slab_total']:.4f} (err {r782['err_slab_total_pct']:+.2f}%)  ",
        f"- GT = {r782['sxy_gt']:.4f}",
        "",
        "## Raw results",
        "",
        "```json",
        json.dumps(rows, indent=2, default=float),
        "```",
    ]

    path.write_text("\n".join(lines))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    rows = run_comparison()

    print("\n" + "=" * 120, flush=True)
    print(format_table(rows), flush=True)
    print("=" * 120, flush=True)

    out_path = DEV / "sxy_slab_variant_2026-06-02.md"
    write_report(rows, out_path)
