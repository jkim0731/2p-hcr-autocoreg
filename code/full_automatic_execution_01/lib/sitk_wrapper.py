"""F5 — SimpleITK mutual-information / NCC wrappers.

Thin wrapper that:
  - takes np arrays + anisotropic voxel spacing,
  - wraps them in `SimpleITK.Image`,
  - runs an affine (rotation + anisotropic scale + translation) MI fit,
  - optional B-spline deformable refinement.

The caller is responsible for a sensible initial affine — for CZ↔HCR we
typically seed with 180° XY rotation + centroid translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import SimpleITK as sitk
except Exception as e:  # pragma: no cover
    sitk = None
    _SITK_ERR = e


@dataclass
class MIFitResult:
    affine_matrix: np.ndarray       # (3,3) linear, (z,y,x) order
    translation: np.ndarray         # (3,) µm, (z,y,x) order
    center: np.ndarray              # (3,) µm, (z,y,x) order — SITK transform center
    metric: float
    iterations: int
    converged: bool
    bspline_grid_spacing_um: Optional[tuple] = None
    bspline_params: Optional[np.ndarray] = None
    _sitk_composite: object = None  # sitk.CompositeTransform for forward-applying to points

    def apply_inverse(self, pts_zyx: np.ndarray) -> np.ndarray:
        """Apply the inverse (moving→fixed, i.e., CZ→HCR) to an (N,3) zyx array.

        SITK forward semantics: ``p_moving = A @ (p_fixed - c) + c + t``.
        Inverse (moving→fixed): ``p_fixed = A^-1 @ (p_moving - c - t) + c``.

        When a B-spline refinement is present (`_sitk_composite is not None`),
        this method still returns the affine-only inverse — use
        `apply_forward_hcr_to_cz` and rely on nearest-neighbor matching in CZ
        space, because B-spline transforms don't admit a closed-form inverse.
        """
        A = np.asarray(self.affine_matrix)
        t = np.asarray(self.translation)
        c = np.asarray(self.center)
        A_inv = np.linalg.inv(A)
        return ((np.asarray(pts_zyx) - c - t) @ A_inv.T) + c

    def apply_forward_hcr_to_cz(self, pts_zyx: np.ndarray) -> np.ndarray:
        """Apply the composed forward (fixed→moving, i.e., HCR→CZ) including
        the B-spline refinement if present.

        Returns the same (N, 3) shape in (z, y, x) order.
        """
        pts_zyx = np.asarray(pts_zyx, float)
        if self._sitk_composite is None:
            # Affine-only forward
            A = np.asarray(self.affine_matrix)
            t = np.asarray(self.translation)
            c = np.asarray(self.center)
            return ((pts_zyx - c) @ A.T) + c + t
        # SITK: TransformPoint takes (x, y, z)
        P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
        pts_xyz = pts_zyx @ P.T
        out_xyz = np.empty_like(pts_xyz)
        tx = self._sitk_composite
        for i, p in enumerate(pts_xyz):
            out_xyz[i] = tx.TransformPoint(p.tolist())
        return out_xyz @ P.T


def _ensure_sitk():
    if sitk is None:
        raise RuntimeError(f"SimpleITK not available: {_SITK_ERR}")


def _to_sitk(arr: np.ndarray, spacing_xyz_um: "tuple[float, float, float]") -> "sitk.Image":
    _ensure_sitk()
    arr = np.ascontiguousarray(arr.astype(np.float32))
    img = sitk.GetImageFromArray(arr)  # SimpleITK is X-major in space
    # spacing_xyz is (x, y, z) in SimpleITK's world order
    img.SetSpacing(spacing_xyz_um)
    return img


def mi_affine(
    cz_arr: np.ndarray,
    hcr_arr: np.ndarray,
    cz_xy_um: float,
    cz_z_um: float,
    hcr_xy_um: float,
    hcr_z_um: float,
    *,
    init_rotation_deg_z: float = 180.0,
    init_scale: "tuple[float, float, float]" = (1.8, 1.8, 2.8),
    init_translation_um: "tuple[float, float, float]" = (0.0, 0.0, 0.0),
    learning_rate: float = 1.0,
    n_iterations: int = 200,
    pyramid_levels: "tuple[int, ...]" = (4, 2, 1),
    histogram_bins: int = 50,
    sampling_fraction: float = 0.1,
) -> MIFitResult:
    """Register ``cz`` (moving) onto ``hcr`` (fixed) with Mattes MI and an affine
    transform (rotation + anisotropic scale + translation).

    Returns the linear (3,3) and translation in (z, y, x) order matching our
    downstream centroid conventions.
    """
    _ensure_sitk()

    cz_img = _to_sitk(cz_arr, (cz_xy_um, cz_xy_um, cz_z_um))
    hcr_img = _to_sitk(hcr_arr, (hcr_xy_um, hcr_xy_um, hcr_z_um))

    # Initial transform: centered on HCR with 180° XY rotation + anisotropic scale.
    # SITK forward: p_moving = A @ (p_fixed - center) + center + translation.
    # For 'place CZ centred inside HCR', at p_fixed = HCR_center the forward should
    # return p_moving = CZ_center; translation therefore = CZ_center - HCR_center.
    init = sitk.AffineTransform(3)
    R = np.array([
        [np.cos(np.deg2rad(init_rotation_deg_z)), -np.sin(np.deg2rad(init_rotation_deg_z)), 0.0],
        [np.sin(np.deg2rad(init_rotation_deg_z)),  np.cos(np.deg2rad(init_rotation_deg_z)), 0.0],
        [0.0, 0.0, 1.0],
    ])
    S = np.diag(init_scale)
    RS = R @ S
    init.SetMatrix(RS.flatten().tolist())
    hcr_center_xyz = np.array([s_ * n / 2.0 for s_, n in zip(hcr_img.GetSpacing(), hcr_img.GetSize())])
    cz_center_xyz = np.array([s_ * n / 2.0 for s_, n in zip(cz_img.GetSpacing(), cz_img.GetSize())])
    init.SetCenter(hcr_center_xyz.tolist())
    init.SetTranslation((cz_center_xyz - hcr_center_xyz + np.asarray(init_translation_um, float)).tolist())

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(histogram_bins)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(sampling_fraction))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=1e-4,
        numberOfIterations=n_iterations,
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(init, inPlace=False)
    if pyramid_levels:
        reg.SetShrinkFactorsPerLevel(list(pyramid_levels))
        reg.SetSmoothingSigmasPerLevel([float(l) for l in pyramid_levels])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    try:
        out_tx = reg.Execute(hcr_img, cz_img)
        converged = True
    except RuntimeError as e:
        out_tx = init
        converged = False

    # `out_tx` may be a CompositeTransform wrapping an AffineTransform.
    if hasattr(out_tx, "GetNthTransform"):
        aff = out_tx.GetNthTransform(0)
    else:
        aff = out_tx
    aff = sitk.AffineTransform(aff)
    A = np.array(aff.GetMatrix()).reshape(3, 3)
    t_xyz = np.array(aff.GetTranslation())
    c_xyz = np.array(aff.GetCenter())
    # Permute (x,y,z) → (z,y,x) for our conventions.
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    A_zyx = P @ A @ P.T
    t_zyx = P @ t_xyz
    c_zyx = P @ c_xyz

    return MIFitResult(
        affine_matrix=A_zyx,
        translation=t_zyx,
        center=c_zyx,
        metric=float(reg.GetMetricValue()),
        iterations=int(reg.GetOptimizerIteration()),
        converged=converged,
    )


def mi_bspline(
    cz_arr: np.ndarray,
    hcr_arr: np.ndarray,
    cz_xy_um: float,
    cz_z_um: float,
    hcr_xy_um: float,
    hcr_z_um: float,
    *,
    initial_affine: Optional[np.ndarray] = None,        # (3,3) in (z,y,x)
    initial_translation: Optional[np.ndarray] = None,   # (3,) in (z,y,x)
    initial_center: Optional[np.ndarray] = None,        # (3,) in (z,y,x)
    grid_spacing_um: "tuple[float, float, float]" = (30.0, 30.0, 60.0),
    n_iterations: int = 100,
    sampling_fraction: float = 0.2,
) -> MIFitResult:
    """MI + B-spline deformable refinement composed on top of an affine initial.

    When ``initial_affine`` / ``initial_translation`` / ``initial_center`` are
    supplied (in (z, y, x) order to match our centroid conventions), they are
    pre-composed via ``SetMovingInitialTransform`` so the B-spline only fits
    the residual nonrigid deformation. Without them the B-spline fits a raw
    CZ↔HCR volume pair, which does not converge when scale/rotation differ.
    """
    _ensure_sitk()

    cz_img = _to_sitk(cz_arr, (cz_xy_um, cz_xy_um, cz_z_um))
    hcr_img = _to_sitk(hcr_arr, (hcr_xy_um, hcr_xy_um, hcr_z_um))

    # Build the moving-initial affine from caller-supplied (z, y, x) matrices.
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)  # z,y,x ↔ x,y,z swap
    aff = sitk.AffineTransform(3)
    if initial_affine is not None:
        A_zyx = np.asarray(initial_affine, float)
        t_zyx = (np.asarray(initial_translation, float)
                 if initial_translation is not None else np.zeros(3))
        c_zyx = (np.asarray(initial_center, float)
                 if initial_center is not None else np.zeros(3))
        A_xyz = P @ A_zyx @ P.T
        t_xyz = P @ t_zyx
        c_xyz = P @ c_zyx
        aff.SetMatrix(A_xyz.flatten().tolist())
        aff.SetTranslation(t_xyz.tolist())
        aff.SetCenter(c_xyz.tolist())
    else:
        A_zyx = np.eye(3)
        t_zyx = np.zeros(3)
        c_zyx = np.zeros(3)

    mesh_size = [
        max(1, int(round(hcr_img.GetSize()[i] * hcr_img.GetSpacing()[i] / grid_spacing_um[i])))
        for i in range(3)
    ]
    bspline = sitk.BSplineTransformInitializer(hcr_img, mesh_size, order=3)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(sampling_fraction))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=n_iterations,
    )
    # Pre-compose the affine; only bspline optimizes.
    reg.SetMovingInitialTransform(aff)
    reg.SetInitialTransform(bspline, inPlace=False)

    _verbose = bool(globals().get("MI_BSPLINE_VERBOSE", False))
    if _verbose:
        import time as _time
        _state = {"last": _time.time(), "it": 0}

        def _cb():
            now = _time.time()
            dt = now - _state["last"]
            _state["last"] = now
            it = reg.GetOptimizerIteration()
            print(f"    bspline iter={it} metric={reg.GetMetricValue():.5f} "
                  f"dt={dt:.2f}s", flush=True)
        reg.AddCommand(sitk.sitkIterationEvent, _cb)

    try:
        out_tx = reg.Execute(hcr_img, cz_img)
        converged = True
    except RuntimeError:
        out_tx = bspline
        converged = False

    # Recover the fitted B-spline transform from the registration output.
    if hasattr(out_tx, "GetNthTransform"):
        fitted_bspline = sitk.BSplineTransform(out_tx.GetNthTransform(0))
    else:
        fitted_bspline = sitk.BSplineTransform(out_tx)

    # Build a CompositeTransform so downstream code can apply (bspline ∘ affine)
    # forward to HCR-space points and get CZ-space locations.
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(aff)
    composite.AddTransform(fitted_bspline)

    return MIFitResult(
        affine_matrix=A_zyx,
        translation=t_zyx,
        center=c_zyx,
        metric=float(reg.GetMetricValue()),
        iterations=int(reg.GetOptimizerIteration()),
        converged=converged,
        bspline_grid_spacing_um=tuple(grid_spacing_um),
        bspline_params=np.asarray(fitted_bspline.GetParameters()),
        _sitk_composite=composite,
    )


def _selftest():
    rng = np.random.default_rng(0)
    vol = rng.random((20, 30, 30)).astype(np.float32)
    r = mi_affine(vol, vol, 1.0, 1.0, 1.0, 1.0,
                   init_rotation_deg_z=0.0, init_scale=(1.0, 1.0, 1.0),
                   n_iterations=20, pyramid_levels=(2, 1))
    print("F5 selftest metric:", r.metric, "converged:", r.converged)


if __name__ == "__main__":
    _selftest()
