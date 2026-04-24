"""S45 — Volume-ratio smoke test for M4-augmented P1 re-ranking.

S44 confirmed the bottleneck on 755252 / 767022 is per-pair ranking at the
correct scale, NOT coarse localization. Before investing in spatial Dice/IoU
(which needs F3 cross-resolution resampling — not yet built), ask the cheaper
question first:

  *Is per-cell volume alone identity-informative, on top of F6-cosine + Euclidean distance?*

If YES, volume ratio belongs in P1's putative score (~ 1 extra feature),
and full spatial IoU may not even be needed. If NO, re-ranking on masks is
unlikely to help without spatial info, which means invest in F3 before M4.

Strategy:
  1. For each of {788406, 755252, 767022}, run P1 → pairs_df.
  2. Load F2 CZ seg-mask → per-cz_id voxel count → V_cz (µm³).
  3. Use `s.hcr_gfp_df['volume']` → V_hcr per hcr_id (µm³, already there).
  4. For each P1-emitted pair, compute log_ratio = log(V_hcr) - log(V_cz).
  5. Split pairs by:
     - GT (cz_id, hcr_id ∈ coreg_table)  → "gt"
     - Non-GT but within 30 µm of P1's TPS prediction → "near_miss"
     - P1-emitted pair that isn't GT (majority class on stress subjects) → "p1_wrong"
  6. Compare log_ratio distributions across groups.
     If GT distribution is tight around a subject-specific center AND
     `near_miss`/`p1_wrong` distributions are substantially wider,
     volume is informative.

Skip 782149 — no GT pairs are recoverable there (S42/S43/S44), so volume
signal would be meaningless.
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

import bench.candidates  # noqa: F401  (registers all)
from bench.harness import CANDIDATES  # noqa: E402
from benchmark_data_loader import load_subject  # noqa: E402
from lib.mask_loaders import load_cz_seg_mask  # noqa: E402


def cz_volumes_um3(s) -> pd.Series:
    """Return pd.Series indexed by cz_id with per-cell volume in µm³."""
    t0 = time.time()
    cz_mask, xy_um, z_um = load_cz_seg_mask(s)
    vox_um3 = float(xy_um) ** 2 * float(z_um)
    uniq, counts = np.unique(cz_mask[cz_mask > 0], return_counts=True)
    vols = pd.Series(counts * vox_um3, index=uniq.astype(int), name="cz_vol_um3")
    vols.index.name = "cz_id"
    print(f"  CZ seg-mask loaded: shape={cz_mask.shape} "
          f"xy_um={xy_um:.3f} z_um={z_um:.3f} "
          f"n_labels={len(vols)} med_vol_um3={float(vols.median()):.1f} "
          f"wall={time.time()-t0:.1f}s", flush=True)
    return vols


def hcr_volumes_um3(s) -> pd.Series:
    """Return pd.Series indexed by hcr_id with per-cell volume in µm³.

    `metrics.pickle` stores per-cell voxel counts in native HCR voxel units.
    Convert via hcr_xy_um² * hcr_z_um.
    """
    df = s.hcr_gfp_df
    if "volume" not in df.columns:
        raise RuntimeError(f"subject {s.subject_id} has no 'volume' column (no metrics.pickle)")
    vox_um3 = float(s.hcr_xy_um) ** 2 * float(s.hcr_z_um)
    vols = pd.Series(df["volume"].values * vox_um3, index=df["hcr_id"].astype(int).values,
                     name="hcr_vol_um3")
    vols.index.name = "hcr_id"
    return vols


def describe(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return dict(n=0, med=float("nan"), iqr=float("nan"),
                    p10=float("nan"), p90=float("nan"), std=float("nan"))
    return dict(
        n=len(a),
        med=float(np.median(a)),
        iqr=float(np.percentile(a, 75) - np.percentile(a, 25)),
        p10=float(np.percentile(a, 10)),
        p90=float(np.percentile(a, 90)),
        std=float(np.std(a)),
    )


def analyze_subject(subj: str):
    print(f"\n=== {subj} ===", flush=True)
    s = load_subject(subj)

    # GT pairs.
    gt = set(zip(s.coreg_table["cz_id"].astype(int),
                 s.coreg_table["hcr_id"].astype(int)))
    print(f"  n_gt={len(gt)}  n_cz={len(s.cz_centroids)}  n_hcr_gfp={len(s.hcr_gfp_df)}",
          flush=True)

    # Volumes.
    cz_vol = cz_volumes_um3(s)
    hcr_vol = hcr_volumes_um3(s)

    # GT-pair volume ratio (ground-truth).
    gt_log_ratio = []
    for c, h in gt:
        if c in cz_vol.index and h in hcr_vol.index:
            gt_log_ratio.append(np.log(hcr_vol[h]) - np.log(cz_vol[c]))
    gt_log_ratio = np.array(gt_log_ratio)
    print(f"  GT pairs with vol info: {len(gt_log_ratio)}/{len(gt)}", flush=True)

    # Run P1.
    t0 = time.time()
    res = CANDIDATES["P1"](s)
    p1_wall = time.time() - t0
    pairs = res.pairs_df
    print(f"  P1 pairs emitted: {len(pairs)}  wall={p1_wall:.1f}s", flush=True)

    # Label each P1 pair.
    is_gt = np.array([((r.cz_id, r.hcr_id) in gt) for r in pairs.itertuples()])

    # Compute log-vol-ratio for every P1 pair.
    p1_log = []
    for r in pairs.itertuples():
        c = int(r.cz_id); h = int(r.hcr_id)
        if c in cz_vol.index and h in hcr_vol.index:
            p1_log.append(np.log(hcr_vol[h]) - np.log(cz_vol[c]))
        else:
            p1_log.append(np.nan)
    p1_log = np.array(p1_log)

    # Splits.
    p1_gt_lr = p1_log[is_gt & np.isfinite(p1_log)]
    p1_wrong_lr = p1_log[(~is_gt) & np.isfinite(p1_log)]

    print(f"  log_ratio stats:")
    print(f"    GT-all (from coreg_table):  {describe(gt_log_ratio)}")
    print(f"    P1-emitted & GT:            {describe(p1_gt_lr)}")
    print(f"    P1-emitted & wrong:         {describe(p1_wrong_lr)}")

    # Identity-informative check:
    #   If the P1-GT distribution has much smaller spread (IQR) than the
    #   P1-wrong distribution around the GT center, volume is informative.
    if len(gt_log_ratio) >= 10:
        gt_center = float(np.median(gt_log_ratio))
        d_gt = np.abs(p1_gt_lr - gt_center)
        d_wrong = np.abs(p1_wrong_lr - gt_center)
        print(f"    center={gt_center:.3f}  "
              f"|Δ| medians: gt={np.median(d_gt):.3f}  wrong={np.median(d_wrong):.3f}  "
              f"ratio={np.median(d_wrong)/max(np.median(d_gt),1e-9):.2f}")

    return dict(
        subject=subj, n_gt=len(gt), n_gt_with_vol=len(gt_log_ratio),
        gt_log_med=describe(gt_log_ratio)["med"],
        gt_log_iqr=describe(gt_log_ratio)["iqr"],
        p1_n_pairs=int(len(pairs)),
        p1_n_gt=int(is_gt.sum()),
        p1_gt_log_iqr=describe(p1_gt_lr)["iqr"],
        p1_wrong_log_iqr=describe(p1_wrong_lr)["iqr"],
    )


def main():
    subjects = ["788406", "755252", "767022"]
    rows = []
    for subj in subjects:
        try:
            rows.append(analyze_subject(subj))
        except Exception as e:
            print(f"  {subj} FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()

    df = pd.DataFrame(rows)
    df.to_csv(
        "/root/capsule/code/full_automatic_execution_01/sessions/45_iou_augmented_p1/vol_ratio_summary.csv",
        index=False,
    )
    print("\n\n=== SUMMARY ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
