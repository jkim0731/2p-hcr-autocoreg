"""F1 / F2 — mask loaders.

F1 — HCR: read `cell_body_segmentation/segmentation_mask.zarr`, restrict to
  GFP+ cell IDs, downsample to a requested level matching the HCR zarr
  pyramid convention (level 2 ~= centroid grid, level 4 ~= surface-fit
  image grid).

F2 — CZ: read `*_seg-mask-outline.tif` as a labelled 3D volume in CZ
  native orientation; filter by `keep_ids` if requested.

Both loaders return `(mask_zyx, xy_um, z_um)` matching the upstream
centroid convention.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import zarr

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent
if str(_ROOT / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "dev_code"))

from benchmark_data_loader import SubjectData  # noqa: E402
from benchmark_analysis import hcr_level_resolution  # noqa: E402


# ----------------------------------------------------------------------
# F1 — HCR segmentation-mask loader
# ----------------------------------------------------------------------
def load_hcr_seg_mask(
    s: SubjectData,
    level: int = 4,
    gfp_ids: Optional[Iterable[int]] = None,
    *,
    chunk_size_um: float = 200.0,
) -> tuple[np.ndarray, float, float]:
    """Return a downsampled HCR cell-body segmentation mask.

    The native zarr is single-level, full-res (~0.246 µm XY).  For
    ``level=k`` we stride-downsample by ``sxy = 2**k`` in XY and
    ``sz = 2**max(0, k-2)`` in Z (matching the HCR image pyramid
    convention in `benchmark_analysis.hcr_level_resolution`).

    ``gfp_ids`` restricts the mask to those labels; everything else becomes 0.
    """
    zpath = s.hcr_dir / "cell_body_segmentation" / "segmentation_mask.zarr"
    if not zpath.exists():
        raise FileNotFoundError(f"No HCR seg mask zarr at {zpath}")

    cache_dir = _THIS_DIR.parent / ".cache" / "hcr_seg_mask"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = "gfpall" if gfp_ids is None else _gfp_digest(gfp_ids)
    cache_path = cache_dir / f"{s.subject_id}_level{level}_{cache_tag}.npy"
    if cache_path.exists():
        out = np.load(cache_path)
        xy_um, z_um = hcr_level_resolution(s, level)
        return out, float(xy_um), float(z_um)

    z = zarr.open(str(zpath), mode="r")
    arr = z["0"]  # (1, 1, Z, Y, X), uint32, chunks 128 per spatial axis
    Z, Y, X = arr.shape[-3:]

    sxy = 2 ** level
    sz = 2 ** max(0, level - 2)
    oz, oy, ox = (Z + sz - 1) // sz, (Y + sxy - 1) // sxy, (X + sxy - 1) // sxy

    out = np.zeros((oz, oy, ox), dtype=np.uint32)

    # 128×512×512 tiles ~134 MB, ~0.14 s decompress → 10 min full-volume read.
    tz, ty, tx = 128, 512, 512
    for z0 in range(0, Z, tz):
        z1 = min(Z, z0 + tz)
        for y0 in range(0, Y, ty):
            y1 = min(Y, y0 + ty)
            for x0 in range(0, X, tx):
                x1 = min(X, x0 + tx)
                tile = np.asarray(arr[0, 0, z0:z1, y0:y1, x0:x1])
                # Compute native offsets that land on the target-grid.
                # Output index for z,y,x: i = (z0 + k*sz), so k starts
                # at ceil((first output z) * sz - z0)/sz. Simpler:
                # compute first native index in [z0,z1) divisible by sz.
                def _stride_start(lo: int, step: int) -> int:
                    # smallest m >= lo with m % step == 0
                    return ((lo + step - 1) // step) * step
                nz0 = _stride_start(z0, sz) - z0
                ny0 = _stride_start(y0, sxy) - y0
                nx0 = _stride_start(x0, sxy) - x0
                sub = tile[nz0::sz, ny0::sxy, nx0::sxy]
                oz0 = (z0 + nz0) // sz
                oy0 = (y0 + ny0) // sxy
                ox0 = (x0 + nx0) // sxy
                oz1 = min(oz0 + sub.shape[0], out.shape[0])
                oy1 = min(oy0 + sub.shape[1], out.shape[1])
                ox1 = min(ox0 + sub.shape[2], out.shape[2])
                sub = sub[: oz1 - oz0, : oy1 - oy0, : ox1 - ox0]
                out[oz0:oz1, oy0:oy1, ox0:ox1] = sub

    if gfp_ids is not None:
        gfp_set = np.asarray(sorted({int(i) for i in gfp_ids}), dtype=np.uint32)
        mask = np.isin(out, gfp_set)
        out = np.where(mask, out, 0).astype(np.uint32)

    np.save(cache_path, out)
    xy_um, z_um = hcr_level_resolution(s, level)
    return out, float(xy_um), float(z_um)


def _gfp_digest(gfp_ids: Iterable[int]) -> str:
    import hashlib
    ids = np.asarray(sorted({int(i) for i in gfp_ids}), dtype=np.int64)
    h = hashlib.md5(ids.tobytes()).hexdigest()[:10]
    return f"gfp{len(ids)}_{h}"


# ----------------------------------------------------------------------
# F2 — CZ segmentation-mask loader
# ----------------------------------------------------------------------
def load_cz_seg_mask(
    s: SubjectData,
    keep_ids: Optional[Iterable[int]] = None,
) -> tuple[np.ndarray, float, float]:
    """Load ``*_seg-mask-outline.tif`` as ``(Z, Y, X)`` label volume.

    If the file contains only outlines (non-filled), we flood-fill with
    the interior of each closed shape per-z-slice.  If it's already
    filled, we return as-is.
    """
    import tifffile
    from scipy import ndimage as ndi

    paths = list(s.coreg_dir.glob("*_seg-mask-outline.tif"))
    if not paths:
        paths = list(s.coreg_dir.glob("*seg-mask-outline*.tif"))
    if not paths:
        raise FileNotFoundError(f"No CZ seg mask tiff in {s.coreg_dir}")
    arr = tifffile.imread(str(paths[0]))
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected CZ seg-mask shape: {arr.shape}")

    # The file may store either labelled interiors or outlines.  Detect:
    # if every cell's voxels form a thin band (thin in local bbox), it's
    # an outline.  Cheap heuristic: total nonzero voxel count vs. number
    # of unique labels * typical cell radius^3.
    nz = (arr > 0).sum()
    uniq = np.unique(arr)
    uniq = uniq[uniq > 0]
    # If mean voxels-per-cell is under ~100, these are outlines; fill them.
    filled = arr
    if len(uniq) > 0 and nz / max(1, len(uniq)) < 100:
        filled = _fill_outlines(arr)

    if keep_ids is not None:
        keep_set = np.asarray(sorted({int(i) for i in keep_ids}), dtype=filled.dtype)
        mask = np.isin(filled, keep_set)
        filled = np.where(mask, filled, 0).astype(filled.dtype)

    return filled.astype(np.uint32), float(s.cz_xy_um), float(s.cz_z_um)


def _fill_outlines(arr: np.ndarray) -> np.ndarray:
    """Per-z fill outline holes.  Mass-conservative for closed contours."""
    from scipy import ndimage as ndi

    out = np.zeros_like(arr, dtype=arr.dtype)
    Z = arr.shape[0]
    for zi in range(Z):
        sl = arr[zi]
        if sl.max() == 0:
            continue
        # For each cell ID on this slice, fill the interior enclosed by its
        # outline.  Outline ids are stored in `sl`; inside is 0.
        ids_here = np.unique(sl)
        ids_here = ids_here[ids_here > 0]
        for label in ids_here:
            contour = (sl == label)
            filled = ndi.binary_fill_holes(contour)
            out[zi][filled] = label
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _summary(subject_id: str = "788406", level: int = 4):
    from benchmark_data_loader import load_subject
    s = load_subject(subject_id)
    gfp_ids = s.hcr_gfp_df["hcr_id"].astype(int).tolist()
    m, xy_um, z_um = load_hcr_seg_mask(s, level=level, gfp_ids=gfp_ids)
    nz = int((m > 0).sum())
    uniq = np.unique(m)
    uniq = uniq[uniq > 0]
    print(f"[F1] {subject_id} level={level} shape={m.shape} xy_um={xy_um:.3f} "
          f"z_um={z_um:.3f} nonzero_voxels={nz} unique_gfp={len(uniq)}")
    try:
        cz, cxy, cz_ = load_cz_seg_mask(s)
        nz2 = int((cz > 0).sum())
        uniq2 = np.unique(cz)
        uniq2 = uniq2[uniq2 > 0]
        print(f"[F2] {subject_id} shape={cz.shape} xy_um={cxy:.3f} z_um={cz_:.3f} "
              f"nonzero={nz2} unique_ids={len(uniq2)}")
    except FileNotFoundError as e:
        print(f"[F2] {subject_id} SKIP — {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("subject_id", nargs="?", default="788406")
    p.add_argument("--level", type=int, default=4)
    args = p.parse_args()
    _summary(args.subject_id, args.level)
