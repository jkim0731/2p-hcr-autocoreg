"""F9 — Benchmark harness for Grand Plan candidates.

Thin dispatcher. Each candidate registers a callable with

    @register_candidate("P1")
    def run(s: SubjectData, **kwargs) -> CoregResult:
        ...

and `run_candidate(candidate_id, subject_id)` loads the subject, invokes the
callable, scores the returned :class:`CoregResult` against the subject's
``coreg_table.csv``, and writes a row to ``bench_results.csv`` plus a JSON
dump of the raw diagnostics.

Scoring metrics:
  * ``recall``: fraction of ground-truth CZ cells whose predicted HCR match
    (if any) equals the GT ``hcr_id``.
  * ``precision``: fraction of predictions over GT-covered CZ cells that are
    correct.  Predictions on CZ cells missing from GT are ignored (the GT is
    known-incomplete).
  * ``recall_at_D_um``: fraction of GT CZ cells whose predicted HCR centroid
    is within D µm of the true HCR centroid.  D in {5, 10, 20}.
  * ``precision_at_D_um``: over all predicted pairs, fraction with centroid
    error ≤ D µm, using GT pairs as the reference.
  * ``runtime_s``.

A transform (when emitted) is also compared to the landmark set: origin
error, rotation-around-z, and per-axis scale (from ``fit_anisotropic_similarity``
on the predicted correspondences when available).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent  # /root/capsule/code
if str(_ROOT / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "dev_code"))
if str(_THIS_DIR.parent / "lib") not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent / "lib"))

from benchmark_data_loader import (  # noqa: E402
    BENCHMARK_SUBJECTS,
    SubjectData,
    cz_px_to_um,
    hcr_px_to_um,
    load_subject,
)

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TransformDescriptor:
    """Compact summary of a candidate's coarse-alignment transform.

    Any field may be None/NaN — not every candidate emits every term.
    """

    R: Optional[np.ndarray] = None             # (3, 3), row-vec convention (same as R1)
    scales: Optional[np.ndarray] = None        # (3,) per-axis scale; 1.0 = unknown
    translation: Optional[np.ndarray] = None   # (3,)
    src_mean: Optional[np.ndarray] = None      # (3,) CZ centroid subtracted first
    rotation_deg_z: Optional[float] = None
    kind: str = "affine"                        # affine | rigid_scale | tps | none


@dataclass
class CoregResult:
    """Uniform candidate output.

    ``pairs_df`` columns (strict):
      * ``cz_id`` int
      * ``hcr_id`` int
      * ``confidence`` float in [0, 1] (intrinsic score, candidate-defined)
      * ``cz_x_um``, ``cz_y_um``, ``cz_z_um`` — CZ centroid in microns
      * ``hcr_x_um``, ``hcr_y_um``, ``hcr_z_um`` — predicted HCR centroid in microns
        (= the matched HCR cell's centroid if one-to-one; or the warped CZ position if
        the candidate emitted a transform but no discrete match).
    """

    pairs_df: pd.DataFrame
    confidence: float                       # overall confidence of the whole solution, in [0, 1]
    transform: Optional[TransformDescriptor] = None
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
CANDIDATES: Dict[str, Callable[[SubjectData], CoregResult]] = {}


def register_candidate(candidate_id: str):
    """Decorator: register a candidate function under ``candidate_id``."""

    def _wrap(fn: Callable[[SubjectData], CoregResult]):
        if candidate_id in CANDIDATES:
            raise RuntimeError(f"Candidate {candidate_id} already registered")
        CANDIDATES[candidate_id] = fn
        return fn

    return _wrap


# ---------------------------------------------------------------------------
# Scoring against ground truth
# ---------------------------------------------------------------------------
def _cz_centroids_um(s: SubjectData) -> pd.DataFrame:
    arr = s.cz_centroids[["z_px", "y_px", "x_px"]].values.astype(float)
    um = cz_px_to_um(arr, s)
    df = pd.DataFrame({
        "cz_id": s.cz_centroids["cz_id"].values,
        "cz_x_um": um[:, 2],
        "cz_y_um": um[:, 1],
        "cz_z_um": um[:, 0],
    })
    return df


def _hcr_centroids_um(s: SubjectData) -> pd.DataFrame:
    arr = s.hcr_centroids[["z_px", "y_px", "x_px"]].values.astype(float)
    um = hcr_px_to_um(arr, s)
    df = pd.DataFrame({
        "hcr_id": s.hcr_centroids["hcr_id"].values,
        "hcr_x_um": um[:, 2],
        "hcr_y_um": um[:, 1],
        "hcr_z_um": um[:, 0],
    })
    return df


def compare_to_gt(pairs_df: pd.DataFrame, s: SubjectData) -> dict:
    """Score `pairs_df` against the subject's `coreg_table.csv`.

    Distance metrics compare predicted HCR centroid (``hcr_x_um`` etc.) to
    the GT HCR cell's centroid (from ``hcr_centroids``) — so predictions
    from a candidate that snaps to the *wrong* HCR ID still get a distance
    credit if the predicted centroid is near the true HCR centroid.
    """
    gt = s.coreg_table
    if gt.empty or pairs_df.empty:
        return {
            "n_gt": int(len(gt)),
            "n_pred": int(len(pairs_df)),
            "recall": 0.0,
            "precision": 0.0,
            "recall_at_5um": 0.0,
            "recall_at_10um": 0.0,
            "recall_at_20um": 0.0,
            "median_error_um": float("nan"),
            "p95_error_um": float("nan"),
        }

    cz_cent = _cz_centroids_um(s).set_index("cz_id")
    hcr_cent = _hcr_centroids_um(s).set_index("hcr_id")

    pred = pairs_df.copy()
    pred = pred.drop_duplicates(subset=["cz_id"], keep="first")
    pred = pred.set_index("cz_id")

    gt_rows = gt.set_index("cz_id")

    # Join on CZ id — rename pred cols first so column names never clash
    pred_r = pred.rename(columns={
        "hcr_id": "pred_hcr_id",
        "confidence": "pred_conf",
        "hcr_x_um": "pred_hcr_x_um",
        "hcr_y_um": "pred_hcr_y_um",
        "hcr_z_um": "pred_hcr_z_um",
    })[[
        "pred_hcr_id", "pred_conf",
        "pred_hcr_x_um", "pred_hcr_y_um", "pred_hcr_z_um",
    ]]
    joined = gt_rows.join(pred_r, how="left")

    n_gt = len(gt_rows)
    # ID-level match (exact)
    id_match = (joined["pred_hcr_id"] == joined["hcr_id"]).fillna(False)
    id_pred_present = joined["pred_hcr_id"].notna()
    recall_id = float(id_match.sum() / max(1, n_gt))
    precision_id = float(id_match.sum() / max(1, int(id_pred_present.sum())))

    # Distance-based recall: distance from predicted HCR centroid to GT HCR centroid
    gt_hcr = hcr_cent.reindex(joined["hcr_id"])
    pred_xyz = joined[["pred_hcr_x_um", "pred_hcr_y_um", "pred_hcr_z_um"]].values.astype(float)
    gt_xyz = gt_hcr[["hcr_x_um", "hcr_y_um", "hcr_z_um"]].values.astype(float)
    valid = np.isfinite(pred_xyz).all(1) & np.isfinite(gt_xyz).all(1)
    dist = np.full(len(joined), np.nan)
    if valid.any():
        dist[valid] = np.linalg.norm(pred_xyz[valid] - gt_xyz[valid], axis=1)

    d_valid = dist[~np.isnan(dist)]
    def _at(d):
        if len(d_valid) == 0:
            return 0.0
        return float((d_valid <= d).sum() / n_gt)  # denominator = all GT rows
    recall5 = _at(5.0)
    recall10 = _at(10.0)
    recall20 = _at(20.0)

    median_err = float(np.median(d_valid)) if len(d_valid) else float("nan")
    p95_err = float(np.percentile(d_valid, 95)) if len(d_valid) else float("nan")

    return {
        "n_gt": int(n_gt),
        "n_pred": int(id_pred_present.sum()),
        "recall": recall_id,
        "precision": precision_id,
        "recall_at_5um": recall5,
        "recall_at_10um": recall10,
        "recall_at_20um": recall20,
        "median_error_um": median_err,
        "p95_error_um": p95_err,
    }


# ---------------------------------------------------------------------------
# Landmark-based transform evaluation
# ---------------------------------------------------------------------------
def transform_error_vs_landmarks(transform: Optional[TransformDescriptor],
                                  s: SubjectData) -> dict:
    """Evaluate an emitted transform against the manual landmark set.

    Returns ``{}`` if no transform or no landmarks.
    """
    if transform is None:
        return {}

    from benchmark_data_loader import landmark_pairs_um
    cz_um_xyz, hcr_um_xyz = landmark_pairs_um(s, active_only=True)
    if len(cz_um_xyz) < 4:
        return {}

    # landmark_pairs_um returns (x, y, z); candidates use (z, y, x) from
    # centroids_um().  Permute to align with candidate convention.
    cz_um = cz_um_xyz[:, [2, 1, 0]]
    hcr_um = hcr_um_xyz[:, [2, 1, 0]]

    R = transform.R if transform.R is not None else np.eye(3)
    sc = transform.scales if transform.scales is not None else np.ones(3)
    t = transform.translation if transform.translation is not None else np.zeros(3)
    src_mean = transform.src_mean if transform.src_mean is not None else np.zeros(3)

    pred = ((cz_um - src_mean) * sc) @ R.T + t  # canonical (scale, rotate, shift)
    resid = pred - hcr_um
    origin_err = float(np.linalg.norm(pred.mean(0) - hcr_um.mean(0)))
    rms = float(np.sqrt((resid ** 2).sum(1).mean()))
    rot_z = transform.rotation_deg_z
    return {
        "landmark_n": int(len(cz_um)),
        "landmark_rms_um": rms,
        "landmark_origin_err_um": origin_err,
        "landmark_rotation_deg_z": rot_z,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _out_dir(candidate_id: str) -> Path:
    d = _THIS_DIR.parent / "bench_out" / candidate_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bench_csv() -> Path:
    return _THIS_DIR.parent / "bench_out" / "bench_results.csv"


def run_candidate(
    candidate_id: str,
    subject_id: str,
    *,
    write_csv: bool = True,
    extra_kwargs: Optional[dict] = None,
) -> dict:
    """Execute one candidate on one subject and score the result.

    Returns a flat dict of metrics.  Writes:
      * ``bench_out/{candidate_id}/{subject_id}_pairs.csv`` — predicted pairs.
      * ``bench_out/{candidate_id}/{subject_id}_diagnostics.json`` — raw diagnostics.
      * appends a row to ``bench_out/bench_results.csv``.
    """
    if candidate_id not in CANDIDATES:
        raise KeyError(
            f"Candidate {candidate_id!r} not registered. Known: {sorted(CANDIDATES)}"
        )
    fn = CANDIDATES[candidate_id]
    extra_kwargs = extra_kwargs or {}

    t0 = time.time()
    s = load_subject(subject_id)
    load_t = time.time() - t0

    t1 = time.time()
    try:
        result = fn(s, **extra_kwargs)
        err_str = None
    except Exception as exc:  # noqa: BLE001
        err_str = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        result = CoregResult(pairs_df=pd.DataFrame(columns=[
            "cz_id", "hcr_id", "confidence",
            "cz_x_um", "cz_y_um", "cz_z_um",
            "hcr_x_um", "hcr_y_um", "hcr_z_um",
        ]), confidence=0.0)
    run_t = time.time() - t1

    scores = compare_to_gt(result.pairs_df, s)
    t_err = transform_error_vs_landmarks(result.transform, s)

    row = {
        "candidate_id": candidate_id,
        "subject_id": subject_id,
        "runtime_s": round(run_t, 3),
        "load_s": round(load_t, 3),
        "confidence": float(result.confidence),
        **scores,
        **t_err,
        "error": err_str,
    }

    if write_csv:
        out = _out_dir(candidate_id)
        pairs_path = out / f"{subject_id}_pairs.csv"
        result.pairs_df.to_csv(pairs_path, index=False)
        diag_path = out / f"{subject_id}_diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(_json_safe(result.diagnostics), f, indent=2)

        # Append to aggregated CSV (header once).
        bench_csv = _bench_csv()
        header = not bench_csv.exists()
        pd.DataFrame([row]).to_csv(bench_csv, mode="a", header=header, index=False)

    return row


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("candidate_id")
    p.add_argument("subject_id", nargs="?", default="all")
    args = p.parse_args(argv)

    subjects = BENCHMARK_SUBJECTS if args.subject_id == "all" else [args.subject_id]
    # Make parent importable so `bench.candidates` loads via absolute import
    _PARENT = str(_THIS_DIR.parent)
    if _PARENT not in sys.path:
        sys.path.insert(0, _PARENT)
    # When harness is invoked as a script, __main__ is distinct from
    # bench.harness. Submodules' decorator calls end up on bench.harness's
    # CANDIDATES dict. Re-bind CANDIDATES here so both modules share the
    # same registry.
    import bench.harness as _bh
    from bench import candidates as _cand  # noqa: F401
    global CANDIDATES
    CANDIDATES = _bh.CANDIDATES

    rows = []
    for sid in subjects:
        row = run_candidate(args.candidate_id, sid)
        rows.append(row)
        print(f"[{args.candidate_id}:{sid}] recall={row['recall']:.2f} "
              f"recall@10um={row['recall_at_10um']:.2f} "
              f"n_pred={row['n_pred']} runtime={row['runtime_s']}s")
    return rows


if __name__ == "__main__":
    main()
