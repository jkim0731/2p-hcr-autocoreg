"""
coreg_data_loading.py
Standardized data loaders for the co-registration pipeline.

All loaders return DataFrames or dicts in canonical column names so downstream
code never has to know the on-disk format.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
CZ_COLS = ["czstack_cell_id", "czstack_z", "czstack_y", "czstack_x"]
HCR_COLS = ["hcr_cell_id", "hcr_z", "hcr_y", "hcr_x"]
LM_COLS = ["ids", "active", "czstack_x", "czstack_y", "czstack_z",
           "hcr_x", "hcr_y", "hcr_z"]


# ---------------------------------------------------------------------------
# Filepaths
# ---------------------------------------------------------------------------

def load_filepaths(coreg_or_save_dir: str | Path,
                   subject_id: str,
                   czstack_date: str,
                   iter: bool = False) -> dict:
    """Load the filepaths JSON for a given subject.

    Parameters
    ----------
    coreg_or_save_dir : path to the subject's coreg directory (contains the JSON)
    subject_id        : e.g. '790322'
    czstack_date      : e.g. '2025-07-10'
    iter              : if True, load filepaths_iter.json (has spot_488_counts_path)

    Returns
    -------
    dict with string → Path values
    """
    d = Path(coreg_or_save_dir)
    if iter:
        json_name = f"{subject_id}_{czstack_date}_filepaths_iter.json"
    else:
        json_name = f"{subject_id}_{czstack_date}_filepaths.json"

    # Fallback: some old dirs don't have the date prefix; also look in /scratch/
    scratch = Path("/root/capsule/scratch")
    candidates = [
        d / json_name,
        d / f"{subject_id}_filepaths_iter.json",
        d / f"{subject_id}_filepaths.json",
        scratch / f"{subject_id}_coreg_filepaths.json",  # manual override in scratch
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break

    if path is None:
        raise FileNotFoundError(
            f"Could not find filepaths JSON in {d}. Tried: {candidates}"
        )

    with open(path) as f:
        raw = json.load(f)

    return {k: Path(v) for k, v in raw.items()}


def save_filepaths(filepaths: dict, coreg_or_save_dir: str | Path,
                   subject_id: str, czstack_date: str, iter: bool = False):
    """Persist a filepaths dict to JSON."""
    d = Path(coreg_or_save_dir)
    d.mkdir(parents=True, exist_ok=True)
    suffix = "filepaths_iter" if iter else "filepaths"
    out = d / f"{subject_id}_{czstack_date}_{suffix}.json"
    with open(out, "w") as f:
        json.dump({k: str(v) for k, v in filepaths.items()}, f, indent=4)
    return out


# ---------------------------------------------------------------------------
# Czstack centroids
# ---------------------------------------------------------------------------

def load_czstack_centroids(path: str | Path) -> pd.DataFrame:
    """Load czstack cell centroids CSV.

    Returns DataFrame with columns: czstack_cell_id, czstack_z, czstack_y, czstack_x
    """
    df = pd.read_csv(path)
    # Normalise column names
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "id" in lc:
            col_map[c] = "czstack_cell_id"
        elif lc == "z":
            col_map[c] = "czstack_z"
        elif lc == "y":
            col_map[c] = "czstack_y"
        elif lc == "x":
            col_map[c] = "czstack_x"
    df = df.rename(columns=col_map)
    # Ensure integer cell id
    df["czstack_cell_id"] = df["czstack_cell_id"].astype(int)
    return df[CZ_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# HCR centroids
# ---------------------------------------------------------------------------

def load_hcr_centroids(npy_path: str | Path) -> pd.DataFrame:
    """Load HCR cell centroids from an N×4 .npy file.

    The .npy file has columns [z, y, x, cell_id] (unit: pixels in HCR space).

    Returns DataFrame with columns: hcr_cell_id, hcr_z, hcr_y, hcr_x
    """
    arr = np.load(npy_path)  # shape (N, 4)
    if arr.shape[1] == 4:
        df = pd.DataFrame(arr, columns=["hcr_z", "hcr_y", "hcr_x", "hcr_cell_id"])
    else:
        raise ValueError(f"Expected N×4 array, got {arr.shape}")
    df["hcr_cell_id"] = df["hcr_cell_id"].astype(int)
    return df[HCR_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# HCR resolution (µm/pixel)
# ---------------------------------------------------------------------------

def load_hcr_scales(fused_json_path: str | Path) -> dict:
    """Parse fused_ng.json to extract voxel resolution in µm/pixel.

    Returns dict with keys: scale_x, scale_y, scale_z  (all in µm/pixel)
    """
    with open(fused_json_path) as f:
        ng = json.load(f)

    dims = ng.get("dimensions", {})

    def _to_um(entry):
        """Convert [value, unit] → µm."""
        val, unit = entry
        unit = unit.strip()
        if unit == "m":
            return float(val) * 1e6
        elif unit == "mm":
            return float(val) * 1e3
        elif unit == "um" or unit == "µm":
            return float(val)
        elif unit == "nm":
            return float(val) * 1e-3
        else:
            raise ValueError(f"Unknown unit: {unit}")

    return {
        "scale_x": _to_um(dims["x"]),
        "scale_y": _to_um(dims["y"]),
        "scale_z": _to_um(dims["z"]),
    }


# ---------------------------------------------------------------------------
# HCR segmentation metrics
# ---------------------------------------------------------------------------

def load_hcr_metrics(metrics_pickle_path: str | Path) -> pd.DataFrame:
    """Load HCR segmentation metrics pickle.

    The pickle is a dict {cell_id: {'volume': float, 'global_bbox': ndarray}}.

    Returns DataFrame with columns: hcr_cell_id, volume, global_bbox
    """
    with open(metrics_pickle_path, "rb") as f:
        raw = pickle.load(f)

    rows = []
    for cell_id, info in raw.items():
        # info can be a plain dict or a list of chunk-dicts (chunked segmentation)
        if isinstance(info, list):
            # Aggregate volume across chunks; take global_bbox from last chunk
            vol = sum(float(chunk.get("volume", 0)) for chunk in info)
            bbox = info[-1].get("global_bbox", None) if info else None
        else:
            vol = float(info.get("volume", np.nan))
            bbox = info.get("global_bbox", None)
        rows.append({
            "hcr_cell_id": int(cell_id),
            "volume": vol,
            "global_bbox": bbox,
        })
    df = pd.DataFrame(rows)
    df["hcr_cell_id"] = df["hcr_cell_id"].astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Spot counts
# ---------------------------------------------------------------------------

def load_spot_counts(
    hcr_processed_dir: str | Path,
    hcr_metrics_df: pd.DataFrame,
    fallback_coreg_dir: Optional[str | Path] = None,
    subject_id: Optional[str] = None,
    czstack_date: Optional[str] = None,
    gfp_threshold: int = 5,
) -> pd.DataFrame:
    """Load or compute spot counts for the 488 (GFP) channel.

    Search order:
    1. Pre-computed CSV in the coreg dir:  {subject}_{date}_spot_488_counts.csv
    2. Pre-computed CSV without date:      {subject}_spot_488_counts.csv
    3. Compute from HCR image_spot_detection output

    Returns DataFrame with columns: hcr_cell_id, counts, volume, density
    with a boolean column `is_gfp` = counts >= gfp_threshold.
    """
    # -- Try coreg dir CSV -------------------------------------------------
    if fallback_coreg_dir is not None:
        d = Path(fallback_coreg_dir)
        candidates = []
        if subject_id and czstack_date:
            candidates.append(d / f"{subject_id}_{czstack_date}_spot_488_counts.csv")
        if subject_id:
            candidates.append(d / f"{subject_id}_spot_488_counts.csv")
        for p in candidates:
            if p.exists():
                df = pd.read_csv(p)
                # Normalise id column name
                for col in df.columns:
                    if col.lower() in ("hcr_id", "id", "cell_id"):
                        df = df.rename(columns={col: "hcr_cell_id"})
                        break
                df["hcr_cell_id"] = df["hcr_cell_id"].astype(int)
                if "density" not in df.columns:
                    df["density"] = df["counts"] / df["volume"]
                if "is_gfp" not in df.columns:
                    df["is_gfp"] = df["counts"] >= gfp_threshold
                return df

    # -- Compute from image_spot_detection output (spots.csv per detected point) ---
    hcr_dir = Path(hcr_processed_dir)
    spots_csv = hcr_dir / "image_spot_detection" / "channel_488_spots" / "spots.csv"
    if spots_csv.exists():
        spots = pd.read_csv(spots_csv)
        # SEG_ID column holds the segmented cell ID each spot belongs to
        counts_series = spots["SEG_ID"].value_counts()
        df = counts_series.reset_index()
        df.columns = ["hcr_cell_id", "counts"]
        df["hcr_cell_id"] = df["hcr_cell_id"].astype(int)
        if not hcr_metrics_df.empty and "volume" in hcr_metrics_df.columns:
            df = df.merge(hcr_metrics_df[["hcr_cell_id", "volume"]],
                          on="hcr_cell_id", how="left")
            df["density"] = df["counts"] / df["volume"].clip(lower=1)
        else:
            df["volume"] = np.nan
            df["density"] = np.nan
        df["is_gfp"] = df["counts"] >= gfp_threshold
        return df.reset_index(drop=True)

    # -- Fallback: cell_data_mean_{subject_id}_R1.csv (older subjects without spot detection) --
    if subject_id:
        data_root = hcr_dir.parent  # capsule data root
        cell_data_csv = data_root / f"cell_data_mean_{subject_id}_R1.csv"
        if cell_data_csv.exists():
            raw = pd.read_csv(cell_data_csv)
            ch488 = raw[raw["channel"] == 488].copy()
            ch488 = ch488.rename(columns={"cell_id": "hcr_cell_id", "mean": "density"})
            ch488["hcr_cell_id"] = ch488["hcr_cell_id"].astype(int)
            ch488["counts"] = np.nan   # no discrete spot counts available
            ch488["volume"] = np.nan
            # GFP threshold is on density (mean fluorescence); use a sensible cutoff
            gfp_density_threshold = 5.0
            ch488["is_gfp"] = ch488["density"] >= gfp_density_threshold
            return ch488[["hcr_cell_id", "counts", "volume", "density", "is_gfp"]].reset_index(drop=True)

    raise FileNotFoundError(
        f"No spot counts found for subject {subject_id!r} in {hcr_dir}"
    )


# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------

def load_landmarks(csv_path: str | Path) -> pd.DataFrame:
    """Load a BigWarp-style landmark CSV (no header).

    Expected column order: ids, active, czstack_x, czstack_y, czstack_z, hcr_x, hcr_y, hcr_z
    The `active` column may be stored as string 'true'/'false' or bool.

    Returns DataFrame with the 8 canonical columns, active as bool.
    """
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] == 8:
        df.columns = LM_COLS
    elif df.shape[1] == 7:
        # Some old files lack the ids column
        df.columns = LM_COLS[1:]
        df.insert(0, "ids", [f"Pt-{i}" for i in range(len(df))])
    else:
        raise ValueError(
            f"Unexpected landmark CSV width {df.shape[1]} in {csv_path}"
        )

    # Normalise active column
    if df["active"].dtype == object:
        df["active"] = df["active"].str.strip().str.lower().map(
            {"true": True, "false": False}
        ).fillna(False)
    df["active"] = df["active"].astype(bool)

    # Numeric columns
    for c in LM_COLS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)


def save_landmarks(df: pd.DataFrame, path: str | Path):
    """Save a landmarks DataFrame to BigWarp-style CSV (no header)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure correct column order
    out = df[LM_COLS].copy()
    out.to_csv(path, index=False, header=False)


# ---------------------------------------------------------------------------
# Coreg directories discovery
# ---------------------------------------------------------------------------

def find_coreg_dirs(data_root: str | Path) -> list[Path]:
    """Return all subject coreg directories under data_root."""
    data_root = Path(data_root)
    return sorted([p for p in data_root.iterdir()
                   if p.is_dir() and "ctl-czstack-hcr-coreg" in p.name])


def parse_coreg_dir_name(coreg_dir: Path) -> tuple[str, str]:
    """Parse subject_id and czstack_date from a coreg dir name.

    Handles both:
      755252_2024-12-19_ctl-czstack-hcr-coreg_2025-11-18_00-00-00
      767018_ctl-czstack-hcr-coreg_2025-10-16_00-00-00
    """
    name = coreg_dir.name
    parts = name.split("_ctl-czstack-hcr-coreg")[0].split("_")
    subject_id = parts[0]
    czstack_date = parts[1] if len(parts) > 1 else ""
    return subject_id, czstack_date


# ---------------------------------------------------------------------------
# Convenience: load everything for one subject
# ---------------------------------------------------------------------------

def _resolve_path(p: Path, fallback_dir: Path) -> Path:
    """Return p if it exists, else look for p.name in fallback_dir."""
    if p.exists():
        return p
    candidate = fallback_dir / p.name
    if candidate.exists():
        return candidate
    # Try matching by pattern (e.g. centroid CSV without date prefix)
    stem_parts = p.stem.split("_")
    for f in fallback_dir.iterdir():
        if f.suffix == p.suffix and all(part in f.name for part in stem_parts[-2:]):
            return f
    return p  # return original so caller gets a useful error


def load_subject_data(
    coreg_dir: str | Path,
    subject_id: str,
    czstack_date: str,
    gfp_threshold: int = 5,
    load_iter_paths: bool = False,
) -> dict:
    """Load all data needed for one subject into a single dict.

    Keys:
        filepaths, czstack_df, hcr_df, hcr_scales, hcr_metrics, spot_counts

    Paths in the filepaths JSON may point to scratch locations that don't
    exist on a fresh capsule run.  ``_resolve_path`` falls back to the
    coreg dir when a path is missing.
    """
    coreg_dir = Path(coreg_dir)

    try:
        fps = load_filepaths(coreg_dir, subject_id, czstack_date, iter=load_iter_paths)
    except FileNotFoundError:
        fps = load_filepaths(coreg_dir, subject_id, czstack_date, iter=False)

    czstack_df = load_czstack_centroids(
        _resolve_path(fps["czstack_centroid_path"], coreg_dir))
    hcr_df = load_hcr_centroids(fps["hcr_centroid_path"])
    hcr_scales = load_hcr_scales(fps["fused_json_file"])

    metrics_key = "hcr_segmentation_metrics_path"
    if metrics_key in fps and fps[metrics_key].exists():
        hcr_metrics = load_hcr_metrics(fps[metrics_key])
    else:
        hcr_metrics = pd.DataFrame(columns=["hcr_cell_id", "volume", "global_bbox"])

    try:
        spot_counts = load_spot_counts(
            hcr_processed_dir=fps["hcr_centroid_path"].parent.parent,
            hcr_metrics_df=hcr_metrics,
            fallback_coreg_dir=coreg_dir,
            subject_id=subject_id,
            czstack_date=czstack_date,
            gfp_threshold=gfp_threshold,
        )
    except FileNotFoundError:
        # Spot counts unavailable — return stub; gfp-dependent features will be NaN
        spot_counts = pd.DataFrame(
            columns=["hcr_cell_id", "counts", "volume", "density", "is_gfp"]
        )

    return {
        "filepaths": fps,
        "czstack_df": czstack_df,
        "hcr_df": hcr_df,
        "hcr_scales": hcr_scales,
        "hcr_metrics": hcr_metrics,
        "spot_counts": spot_counts,
    }
