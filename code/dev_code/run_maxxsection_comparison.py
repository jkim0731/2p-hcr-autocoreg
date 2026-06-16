"""Compare bbox vs max_xsection sxy estimators across all 6 benchmark subjects.

Writes results to sxy_maxxsection_vs_bbox_2026-06-01.md and prints summary table.
"""
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

import numpy as np
import roi_area_sxy

ALL_SIDS = ["755252", "767018", "767022", "782149", "788406", "790322"]


def run_both(sid: str) -> dict:
    print(f"\n=== {sid} ===")
    print("  running bbox mode ...")
    rbbox = roi_area_sxy.estimate_sxy_roi_area(sid, area_mode="bbox")
    print(f"  bbox done: sxy={rbbox['sxy_median']:.4f}  "
          f"med_cz={rbbox['cz_area_median']:.1f}  med_hcr={rbbox['hcr_area_median']:.1f}")
    print("  running max_xsection mode ...")
    rmx = roi_area_sxy.estimate_sxy_roi_area(sid, area_mode="max_xsection")
    print(f"  max_xsection done: sxy={rmx['sxy_median']:.4f}  "
          f"med_cz={rmx['cz_area_median']:.1f}  med_hcr={rmx['hcr_area_median']:.1f}")
    return {"bbox": rbbox, "max_xsection": rmx}


def main():
    results = {}
    for sid in ALL_SIDS:
        try:
            results[sid] = run_both(sid)
        except Exception as e:
            print(f"  ERROR {sid}: {e}")
            import traceback; traceback.print_exc()
            results[sid] = None

    # Build comparison table
    print("\n")
    header = (f"{'sid':<8}  {'sxy_bbox':>9}  {'sxy_mxs':>9}  {'sxy_GT':>7}"
              f"  {'err_bbox':>8}  {'err_mxs':>8}"
              f"  {'medCZ_bb':>9}  {'medCZ_mx':>9}"
              f"  {'medHCR_bb':>10}  {'medHCR_mx':>10}")
    print(header)
    print("-" * len(header))

    rows_md = []
    for sid in ALL_SIDS:
        r = results[sid]
        if r is None:
            print(f"{sid:<8}  ERROR")
            rows_md.append(f"| {sid} | ERROR | ERROR | - | - | - | - | - | - | - |")
            continue
        bb = r["bbox"]
        mx = r["max_xsection"]
        gt = bb["sxy_gt"]
        err_bb = bb["err_pct_median"]
        err_mx = mx["err_pct_median"]
        med_cz_bb = bb["cz_area_median"]
        med_cz_mx = mx["cz_area_median"]
        med_hcr_bb = bb["hcr_area_median"]
        med_hcr_mx = mx["hcr_area_median"]
        line = (f"{sid:<8}  {bb['sxy_median']:>9.4f}  {mx['sxy_median']:>9.4f}  {gt:>7.4f}"
                f"  {err_bb:>+8.1f}%  {err_mx:>+8.1f}%"
                f"  {med_cz_bb:>9.1f}  {med_cz_mx:>9.1f}"
                f"  {med_hcr_bb:>10.1f}  {med_hcr_mx:>10.1f}")
        print(line)
        rows_md.append(
            f"| {sid} | {bb['sxy_median']:.4f} | {mx['sxy_median']:.4f} | {gt:.4f}"
            f" | {err_bb:+.1f}% | {err_mx:+.1f}%"
            f" | {med_cz_bb:.1f} | {med_cz_mx:.1f}"
            f" | {med_hcr_bb:.1f} | {med_hcr_mx:.1f} |"
        )

    # Compute summary stats
    valid = [(sid, results[sid]) for sid in ALL_SIDS if results[sid] is not None]
    err_bb_all = [r["bbox"]["err_pct_median"] for _, r in valid]
    err_mx_all = [r["max_xsection"]["err_pct_median"] for _, r in valid]
    mae_bb = np.mean(np.abs(err_bb_all))
    mae_mx = np.mean(np.abs(err_mx_all))
    std_bb = np.std(err_bb_all)
    std_mx = np.std(err_mx_all)

    summary_lines = [
        "",
        f"MAE bbox={mae_bb:.2f}%  max_xsection={mae_mx:.2f}%",
        f"STD bbox={std_bb:.2f}%  max_xsection={std_mx:.2f}%",
        f"Winner (lower MAE): {'bbox' if mae_bb <= mae_mx else 'max_xsection'}",
        f"More consistent (lower STD): {'bbox' if std_bb <= std_mx else 'max_xsection'}",
    ]
    for line in summary_lines:
        print(line)

    # Notable divergences: subjects where |err_mx - err_bb| > 3%
    print("\nNotable divergences (|Δerr| > 3%):")
    any_div = False
    for sid, r in valid:
        d = abs(r["max_xsection"]["err_pct_median"] - r["bbox"]["err_pct_median"])
        if d > 3.0:
            closer = "max_xsection" if abs(r["max_xsection"]["err_pct_median"]) < abs(r["bbox"]["err_pct_median"]) else "bbox"
            print(f"  {sid}: Δerr={d:.1f}% ({closer} closer to GT)")
            any_div = True
    if not any_div:
        print("  None — methods agree within 3% error for all subjects")

    # Write markdown report
    md_path = THIS / "sxy_maxxsection_vs_bbox_2026-06-01.md"
    with open(md_path, "w") as f:
        f.write("# sxy Estimator Comparison: bbox vs max_xsection\n\n")
        f.write("**Date:** 2026-06-01  \n")
        f.write("**Method:** `estimate_sxy_roi_area` with `area_mode='bbox'` (current production) "
                "vs `area_mode='max_xsection'` (max over z-slices of in-plane mask pixel count).  \n")
        f.write("**GT reference:** landmark-Procrustes fit (`fit_anisotropic_similarity` on "
                "`landmark_pairs_um(s, active_only=True)`), validation only — not used for tuning.  \n\n")

        f.write("## Per-Subject Results\n\n")
        f.write("| sid | sxy_bbox | sxy_max_xsec | sxy_GT | err_bbox | err_max_xsec"
                " | medCZ_bbox (µm²) | medCZ_mxs (µm²) | medHCR_bbox (µm²) | medHCR_mxs (µm²) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for row in rows_md:
            f.write(row + "\n")

        f.write("\n## Summary Statistics (across all 6 subjects)\n\n")
        f.write(f"| metric | bbox | max_xsection |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| MAE vs GT (%) | {mae_bb:.2f} | {mae_mx:.2f} |\n")
        f.write(f"| STD of %-error | {std_bb:.2f} | {std_mx:.2f} |\n")
        winner_mae = "bbox" if mae_bb <= mae_mx else "max_xsection"
        winner_std = "bbox" if std_bb <= std_mx else "max_xsection"
        f.write(f"| Winner (lower MAE) | {'**bbox**' if winner_mae == 'bbox' else 'bbox'} "
                f"| {'**max_xsection**' if winner_mae == 'max_xsection' else 'max_xsection'} |\n")
        f.write(f"| More consistent (lower STD) | {'**bbox**' if winner_std == 'bbox' else 'bbox'} "
                f"| {'**max_xsection**' if winner_std == 'max_xsection' else 'max_xsection'} |\n\n")

        f.write("## Notable Divergences (|Δerr| > 3%)\n\n")
        any_div2 = False
        for sid, r in valid:
            d = abs(r["max_xsection"]["err_pct_median"] - r["bbox"]["err_pct_median"])
            if d > 3.0:
                closer = "max_xsection" if abs(r["max_xsection"]["err_pct_median"]) < abs(r["bbox"]["err_pct_median"]) else "bbox"
                f.write(f"- **{sid}**: Δerr = {d:.1f}% ({closer} closer to GT)\n")
                any_div2 = True
        if not any_div2:
            f.write("None — methods agree within 3% error for all subjects.\n")

        f.write("\n## Interpretation\n\n")
        f.write(
            "The `max_xsection` area is always ≤ the bounding-box area for each cell "
            "(a box is an upper bound on the actual footprint). For the ratio "
            "sxy = sqrt(median_HCR / median_CZ) to be unbiased, both sides must "
            "be similarly affected. CZ and HCR cell shapes differ (CZ cells are more "
            "elongated due to sparse z-sampling), so the correction factor is not "
            "identical on both sides.  \n\n"
        )
        if mae_bb <= mae_mx:
            f.write(
                "**bbox is closer to GT overall** (lower MAE). "
                "This is consistent with bounding-box area being a consistent "
                "overestimate on both sides — the overestimate cancels in the ratio.  \n"
            )
        else:
            f.write(
                "**max_xsection is closer to GT overall** (lower MAE). "
                "The truer per-cell area removes bbox inflation asymmetrically, "
                "and this asymmetry happens to bring the ratio closer to the landmark GT.  \n"
            )
        f.write(
            "\n**Production recommendation:** default remains `area_mode='bbox'` "
            "(backward-compatible, slightly simpler, existing cache). "
            "`max_xsection` is available for future experiments via the `area_mode` parameter.  \n"
        )

    print(f"\nReport written to: {md_path}")
    return results


if __name__ == "__main__":
    main()
