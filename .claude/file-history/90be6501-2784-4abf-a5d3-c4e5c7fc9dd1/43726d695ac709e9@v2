"""Probe: run I2 and verify whether the TransformDescriptor can be applied to
transform CZ centroids to HCR coordinates.

Hypothesis: the SimpleITK AffineTransform has a non-zero `center` that is
not stored in the TransformDescriptor, so reapplying (A, t) alone places
CZ in the wrong coordinate frame.

Plan:
  1. Run the low-level `mi_affine` (bypassing the I2 wrapper) on 788406.
  2. Directly apply the SimpleITK transform to CZ centroids using the
     library's TransformPoint (ground-truth application).
  3. Compare to each of 4 Procrustes-style application conventions.
  4. Identify which convention matches the SimpleITK ground truth, if any,
     and what additional state (e.g., center) is required.

Output: print median distance from each convention to the SimpleITK ground
truth, and median distance to the GT HCR centroids for the one that matches.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject, cz_px_to_um, hcr_px_to_um  # noqa
from lib.centroid_helpers import centroids_um  # noqa
from lib.sitk_wrapper import mi_affine  # noqa
from bench.candidate_impls._i1_axial_ncc import _load_cz_fullstack, _load_hcr_fullstack  # noqa
import SimpleITK as sitk


def gt_pairs(s):
    ct = s.coreg_table
    cz = s.cz_centroids.set_index("cz_id")
    hc = s.hcr_centroids.set_index("hcr_id")
    mask = ct["cz_id"].isin(cz.index) & ct["hcr_id"].isin(hc.index)
    ct = ct[mask]
    cz_rows = cz.loc[ct["cz_id"].values]
    hc_rows = hc.loc[ct["hcr_id"].values]
    cz_um = cz_px_to_um(cz_rows[["z_px", "y_px", "x_px"]].values, s)
    hc_um = hcr_px_to_um(hc_rows[["z_px", "y_px", "x_px"]].values, s)
    return cz_um[:, [2, 1, 0]], hc_um[:, [2, 1, 0]]


def run_sitk_raw(subj, target_um=8.0):
    """Reproduce mi_affine but also capture and return the raw SITK transform."""
    s = load_subject(subj)
    cz_stack, cz_xy_um, cz_z_um = _load_cz_fullstack(s)
    hcr_vol, hcr_xy_um, hcr_z_um = _load_hcr_fullstack(s)

    cz_ds = zoom(cz_stack, (cz_z_um / target_um, cz_xy_um / target_um, cz_xy_um / target_um),
                 order=1).astype(np.float32)
    hcr_ds = zoom(hcr_vol, (hcr_z_um / target_um, hcr_xy_um / target_um, hcr_xy_um / target_um),
                  order=1).astype(np.float32)

    cz_img = sitk.GetImageFromArray(cz_ds)
    cz_img.SetSpacing((target_um, target_um, target_um))
    hcr_img = sitk.GetImageFromArray(hcr_ds)
    hcr_img.SetSpacing((target_um, target_um, target_um))

    init = sitk.AffineTransform(3)
    R0 = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    S0 = np.diag([1.8, 1.8, 2.8])  # (x,y,z) per sitk
    init.SetMatrix((R0 @ S0).flatten().tolist())
    init.SetCenter([s_ * n / 2.0 for s_, n in zip(hcr_img.GetSpacing(), hcr_img.GetSize())])

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200, gradientMagnitudeTolerance=1e-6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(init, inPlace=False)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([4., 2., 1.])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    try:
        out = reg.Execute(hcr_img, cz_img)
    except RuntimeError as e:
        print(f"  Registration failed: {e}")
        out = init

    if hasattr(out, "GetNthTransform"):
        aff = sitk.AffineTransform(out.GetNthTransform(0))
    else:
        aff = sitk.AffineTransform(out)

    return s, aff


def main():
    for subj in ["788406", "782149"]:
        print(f"\n=== {subj} ===", flush=True)
        s, aff = run_sitk_raw(subj, target_um=8.0)

        cz_gt, hcr_gt = gt_pairs(s)  # physical µm, (x,y,z) order
        print(f"  cz_gt[0]  = {cz_gt[0]}")
        print(f"  hcr_gt[0] = {hcr_gt[0]}")

        # SimpleITK: T maps fixed(HCR) → moving(CZ).
        # We want CZ → HCR: invert.
        aff_inv = aff.GetInverse()

        # Test: apply aff_inv to cz_gt and compare to hcr_gt.
        # cz_gt[:, (x,y,z)] → feed directly; aff_inv expects (x,y,z).
        pred = np.asarray([aff_inv.TransformPoint(tuple(p.tolist())) for p in cz_gt])
        d = np.linalg.norm(pred - hcr_gt, axis=1)
        print(f"  SITK inverse applied to CZ: median={np.median(d):.1f} µm, n<50={int((d<50).sum())}")

        # Sanity: forward (HCR → CZ) applied to hcr_gt, compare to cz_gt.
        fwd = np.asarray([aff.TransformPoint(tuple(p.tolist())) for p in hcr_gt])
        d2 = np.linalg.norm(fwd - cz_gt, axis=1)
        print(f"  SITK forward applied to HCR: median={np.median(d2):.1f} µm, n<50={int((d2<50).sum())}")

        # Extract and print raw pieces
        A_xyz = np.array(aff.GetMatrix()).reshape(3, 3)
        t_xyz = np.array(aff.GetTranslation())
        c_xyz = np.array(aff.GetCenter())
        print(f"  SITK matrix (xyz):\n{A_xyz}")
        print(f"  SITK translation (xyz): {t_xyz}")
        print(f"  SITK center (xyz): {c_xyz}")

        Ai_xyz = np.array(aff_inv.GetMatrix()).reshape(3, 3)
        ti_xyz = np.array(aff_inv.GetTranslation())
        ci_xyz = np.array(aff_inv.GetCenter())
        print(f"  SITK inv matrix (xyz):\n{Ai_xyz}")
        print(f"  SITK inv translation (xyz): {ti_xyz}")
        print(f"  SITK inv center (xyz): {ci_xyz}")

        # Manual forward: T(p) = A @ (p - c) + c + t
        pred_manual = ((cz_gt - ci_xyz) @ Ai_xyz.T) + ci_xyz + ti_xyz
        d3 = np.linalg.norm(pred_manual - hcr_gt, axis=1)
        print(f"  Manual (inv): A@(p-c)+c+t median={np.median(d3):.1f} µm")


if __name__ == "__main__":
    main()
