"""
Standalone HCR ROI labelling GUI (no Jupyter).

A pure matplotlib-window application.  Run from a shell with a display::

    python /root/capsule/code/dev_code/roi_label_gui.py \
        --sid 788406 --reviewer alice

Layout (3-D orthoview)::

    +-----------+ +-----------+ +-----------+ +----------+
    | xy axial  | | xz coronal| | yz sagit. | | feature  |
    +-----------+ +-----------+ +-----------+ | panel    |
    | header (sid / hcr_id / score / pos z,y,x / i / N)  |
    | (405)(488)(overlay)  [MIP]                          |
    | [Good][Bad][Bad-OK][Merged][Unsure][Skip][Undo][Quit]
    | last: good(14328) bad(14329) ...        |          |
    +-----------------------------------------------+----+

Keyboard:
    g / b / o / e / u    label good / bad / bad_ok / merged / unsure (and advance)
    s                    skip          z   undo            q  quit
    j / down-arrow       z - 1
    k / up-arrow         z + 1
    mouse wheel          z +/- 1 by default; over xz panel scrolls y;
                         over yz panel scrolls x
    m                    toggle MIP / single-slice
    1 / 2 / 3            channel = 405 / 488 / overlay
    n / p                next / previous subject (when --sid lists multiple)

Labels are appended (by default) to::

    code/sessions/v3_S11_roi_quality/outputs/roi_qc_actions.jsonl

Override with --label-log PATH.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

# Pick a backend that supports interactive windows.  TkAgg is in the
# Python stdlib so it is the safest default; fall back to QtAgg if Tk
# is not available.
for _bk in ("TkAgg", "QtAgg", "Qt5Agg"):
    try:
        matplotlib.use(_bk, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox

ROOT = Path("/root/capsule")
CACHE_DIR = ROOT / "code/dev_code/cached_roi_quality"
TIGHT_BBOX_DIR = ROOT / "code/dev_code/cached_hcr_cell_tight_bbox"

# Default JSONL label log — kept in the session folder so history accumulates
# in the same place the stage-2 trainer reads from.  Override via --label-log.
_DEFAULT_LABEL_LOG = (
    ROOT / "code/sessions/v3_S11_roi_quality/outputs/roi_qc_actions.jsonl"
)

# Populated lazily by _hcr_dir(); reset does not clear this cache — it is
# process-global and harmless to keep across subjects.
_HCR_DATA_ROOTS: dict[str, Path] = {}

_BENCHMARK_SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]

# Module-level mutable so the label-write path can be overridden from main().
LABEL_LOG: Path = _DEFAULT_LABEL_LOG

# ──────────────────────────────────────────────────────────────────────────────
# data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _ROI:
    sid: str
    hcr_id: int
    score: float                                   # v5d binary positive probability
    bbox: tuple[int, int, int, int, int, int]      # zmin,zmax,ymin,ymax,xmin,xmax (level-2 vox)
    centroid: tuple[float, float, float]           # zc, yc, xc (level-2 vox)
    features: dict[str, Any]
    proba_4class: dict[str, float] = field(default_factory=dict)  # bad, bad_ok, good, merged


@dataclass
class _RoiCrops:
    """Cached per-ROI volumetric crops, normalised once for fast scrubbing."""

    img_405: np.ndarray       # (Z, Y, X) float32 in [0, 1]
    img_488: np.ndarray       # (Z, Y, X) float32 in [0, 1]
    label_mask: np.ndarray    # (Z, Y, X) bool -- only the active hcr_id
    crop_origin: tuple[int, int, int]   # (z0, y0, x0) global level-2 vox
    z_anchor: int             # local z index of centroid (clamped into bbox)
    y_anchor: int
    x_anchor: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.img_405.shape


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _hcr_dir(sid: str) -> Path:
    if sid not in _HCR_DATA_ROOTS:
        matches = sorted((ROOT / "data").glob(f"HCR_{sid}_*"))
        if not matches:
            raise FileNotFoundError(f"No HCR data dir for sid={sid}")
        _HCR_DATA_ROOTS[sid] = matches[0]
    return _HCR_DATA_ROOTS[sid]


def _open_zarr(path: Path):
    return zarr.open(str(path), mode="r")


def _load_label_log() -> pd.DataFrame:
    if not LABEL_LOG.exists():
        return pd.DataFrame(columns=["ts", "sid", "hcr_id", "label"])
    rows = []
    with open(LABEL_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _append_label(record: dict[str, Any]) -> None:
    LABEL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_LOG, "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# v5d feature + prediction loading  (delegates to roi_quality_v5d)
# ──────────────────────────────────────────────────────────────────────────────

def _load_v5d_for_subject(sid: str) -> pd.DataFrame:
    """Return a DataFrame with hcr_id, score, p_bad, p_bad_ok, p_good,
    p_merged, and all FEATURE_COLUMNS for `sid`.

    Uses roi_quality_v5d.extract_features + predict so inference always
    reflects the current on-disk model, not stale cached parquets.
    """
    import roi_quality_v5d as rqv

    feat = rqv.extract_features(sid)               # hcr_id + FEATURE_COLUMNS
    binary_score, four_class_proba = rqv.predict(feat)

    out = feat.copy()
    # binary_score is a Series indexed by hcr_id
    out["score"] = binary_score.values
    for c in rqv.CLASS_NAMES:
        out[f"p_{c}"] = four_class_proba[c].values
    return out


def _load_all_data_for_subject(sid: str) -> pd.DataFrame:
    """Merge v5d features+predictions with the tight-bbox parquet.

    Returns a DataFrame with all columns needed to build _ROI objects
    and populate the sampler.
    """
    v5d = _load_v5d_for_subject(sid)
    bbox = pd.read_parquet(TIGHT_BBOX_DIR / f"{sid}_hcr_cell_tight_bbox_v1.parquet")
    merged = v5d.merge(bbox, on="hcr_id", how="inner")
    print(
        f"  [{sid}] loaded {len(merged)} ROIs; "
        f"score range [{merged['score'].min():.3f}, {merged['score'].max():.3f}]"
    )
    return merged


# Columns we strip when copying a merged-row into _ROI.features so the dict
# does not carry duplicated bbox / score / proba values.
_RESERVED_ROI_KEYS = frozenset({
    "hcr_id", "sid", "y", "label", "human_label",
    "zmin_vox", "zmax_vox", "ymin_vox", "ymax_vox", "xmin_vox", "xmax_vox",
    "zc_vox", "yc_vox", "xc_vox", "volume_vox",
    "score", "p_bad", "p_bad_ok", "p_good", "p_merged",
})


def _make_roi_from_row(sid: str, r: pd.Series) -> _ROI:
    features: dict[str, Any] = {}
    for k in r.index:
        if k in _RESERVED_ROI_KEYS:
            continue
        v = r[k]
        if isinstance(v, (str, bytes)):
            continue
        features[k] = v
    return _ROI(
        sid=sid,
        hcr_id=int(r["hcr_id"]),
        score=float(r["score"]),
        bbox=(
            int(r["zmin_vox"]), int(r["zmax_vox"]),
            int(r["ymin_vox"]), int(r["ymax_vox"]),
            int(r["xmin_vox"]), int(r["xmax_vox"]),
        ),
        centroid=(float(r["zc_vox"]), float(r["yc_vox"]), float(r["xc_vox"])),
        features=features,
        proba_4class={
            "bad": float(r["p_bad"]),
            "bad_ok": float(r["p_bad_ok"]),
            "good": float(r["p_good"]),
            "merged": float(r["p_merged"]),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# ROI sampling
# ──────────────────────────────────────────────────────────────────────────────

def _sample_uncertain(
    sid: str,
    n_rois: int,
    score_band: tuple[float, float],
    rng: np.random.Generator,
    skip_hcr_ids: set[int],
) -> list[_ROI]:
    df = _load_all_data_for_subject(sid)
    lo, hi = score_band
    df = df[(df["score"] >= lo) & (df["score"] <= hi)]
    df = df[~df["hcr_id"].isin(skip_hcr_ids)]
    if df.empty:
        return []
    n = min(n_rois, len(df))
    idx = rng.choice(len(df), size=n, replace=False)
    sub = df.iloc[idx]
    return [_make_roi_from_row(sid, r) for _, r in sub.iterrows()]


def _sample_from_candidates(
    sid: str,
    hcr_ids: list[int],
    n_rois: int,
    rng: np.random.Generator,
    skip_hcr_ids: set[int],
) -> list[_ROI]:
    """Walk a fixed candidate list (e.g. top-K by margin or p_merged).

    Preserves the order in `hcr_ids` (highest-priority first), drops anything
    already labelled, then takes the first `n_rois`.
    """
    if not hcr_ids:
        return []
    df = _load_all_data_for_subject(sid)
    keep = [h for h in hcr_ids if h not in skip_hcr_ids]
    df = df.set_index("hcr_id")
    keep = [h for h in keep if h in df.index]
    if not keep:
        return []
    sub = df.loc[keep[:n_rois]].reset_index()
    return [_make_roi_from_row(sid, r) for _, r in sub.iterrows()]


def _active_labels(label_log: pd.DataFrame, sid: str) -> dict[int, str]:
    """Return {hcr_id: label} for `sid`, replaying _undone_ tombstones."""
    if label_log.empty:
        return {}
    sub = label_log[label_log["sid"] == sid].copy()
    if sub.empty:
        return {}
    tomb_keys: set[int] = set()
    if "label" in sub.columns and (sub["label"] == "_undone_").any():
        for _, r in sub[sub["label"] == "_undone_"].iterrows():
            ub = r.get("undoes") or {}
            try:
                tomb_keys.add(int(ub.get("hcr_id", -1)))
            except (TypeError, ValueError):
                pass
    sub = sub[sub["label"].isin(["good", "bad", "bad_ok", "merged", "unsure"])]
    sub = sub[~sub["hcr_id"].astype(int).isin(tomb_keys)]
    if "ts" in sub.columns:
        sub = sub.sort_values("ts")
    sub = sub.drop_duplicates(subset=["hcr_id"], keep="last")
    return dict(zip(sub["hcr_id"].astype(int), sub["label"]))


def _sample_labelled(
    sid: str,
    n_rois: int,
    rng: np.random.Generator,
    label_log: pd.DataFrame,
) -> tuple[list[_ROI], dict[int, str]]:
    active = _active_labels(label_log, sid)
    if not active:
        return [], {}
    df = _load_all_data_for_subject(sid)
    df = df[df["hcr_id"].astype(int).isin(active.keys())]
    if df.empty:
        return [], {}
    n = min(n_rois, len(df))
    idx = rng.choice(len(df), size=n, replace=False)
    sub = df.iloc[idx]
    rois = [_make_roi_from_row(sid, r) for _, r in sub.iterrows()]
    prior_labels = {roi.hcr_id: active[roi.hcr_id] for roi in rois}
    return rois, prior_labels


# ──────────────────────────────────────────────────────────────────────────────
# image crops
# ──────────────────────────────────────────────────────────────────────────────

def _crop_with_margin(
    arr_zarr,
    bbox: tuple[int, int, int, int, int, int],
    margin_xy: int = 8,
    margin_z: int = 2,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Read a 3-D crop from a zarr shaped (1,1,Z,Y,X).

    Returns (crop, (z0, y0, x0)) where the offset gives global coords of
    crop[0,0,0] in level-2 voxels.
    """
    Z, Y, X = arr_zarr.shape[-3:]
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    z0 = max(0, zmin - margin_z)
    z1 = min(Z, zmax + margin_z)
    y0 = max(0, ymin - margin_xy)
    y1 = min(Y, ymax + margin_xy)
    x0 = max(0, xmin - margin_xy)
    x1 = min(X, xmax + margin_xy)
    crop = np.asarray(arr_zarr[0, 0, z0:z1, y0:y1, x0:x1])
    return crop, (z0, y0, x0)


def _normalize_volume(vol: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    """Robust per-volume min-max scaling to [0, 1].  Same contrast across all
    slices so scrubbing does not flicker between z planes."""
    flat = vol.ravel().astype(np.float32)
    if flat.size == 0:
        return vol.astype(np.float32)
    lo = float(np.percentile(flat, p_lo))
    hi = float(np.percentile(flat, p_hi))
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def _load_roi_crops(roi: _ROI) -> _RoiCrops:
    """Load the three volumetric crops for a single ROI.

    All bbox coordinates are in level-2 voxels (xy = 0.988 µm/vox).
    The segmentation mask zarr is also opened at level-2 (the file named
    'orig_res' is actually level-2 — see reference_hcr_seg_zarr_levels.md).
    """
    hd = _hcr_dir(roi.sid)
    seg = _open_zarr(hd / "cell_body_segmentation/segmentation_mask_orig_res.zarr")
    fz_405 = _open_zarr(hd / "image_tile_fusing/fused/channel_405.zarr")["2"]
    fz_488 = _open_zarr(hd / "image_tile_fusing/fused/channel_488.zarr")["2"]

    mask_crop, (z0, y0, x0) = _crop_with_margin(seg, roi.bbox)
    img_405, _ = _crop_with_margin(fz_405, roi.bbox)
    img_488, _ = _crop_with_margin(fz_488, roi.bbox)
    label_mask = (mask_crop == roi.hcr_id)

    img_405 = _normalize_volume(img_405)
    img_488 = _normalize_volume(img_488)

    # Anchor indices are centroid coords minus crop origin, clamped.
    zc = int(round(roi.centroid[0])) - z0
    yc = int(round(roi.centroid[1])) - y0
    xc = int(round(roi.centroid[2])) - x0
    Z, Y, X = label_mask.shape
    zc = int(np.clip(zc, 0, Z - 1))
    yc = int(np.clip(yc, 0, Y - 1))
    xc = int(np.clip(xc, 0, X - 1))

    return _RoiCrops(
        img_405=img_405,
        img_488=img_488,
        label_mask=label_mask,
        crop_origin=(z0, y0, x0),
        z_anchor=zc,
        y_anchor=yc,
        x_anchor=xc,
    )


def _planes_for_channel(
    crops: _RoiCrops,
    channel: str,
    mip: bool,
    z_idx: int,
    y_idx: int | None = None,
    x_idx: int | None = None,
):
    """Return ((xy_img, xy_mask), (xz_img, xz_mask), (yz_img, yz_mask))."""
    if channel == "405":
        vol = crops.img_405
    elif channel == "488":
        vol = crops.img_488
    elif channel == "overlay":
        vol = np.maximum(crops.img_405, crops.img_488 * 0.7)
    else:
        raise ValueError(f"unknown channel {channel!r}")

    yc = crops.y_anchor if y_idx is None else int(y_idx)
    xc = crops.x_anchor if x_idx is None else int(x_idx)

    if mip:
        return (
            (vol.max(axis=0), crops.label_mask.any(axis=0)),
            (vol.max(axis=1), crops.label_mask.any(axis=1)),
            (vol.max(axis=2), crops.label_mask.any(axis=2)),
        )
    return (
        (vol[z_idx],     crops.label_mask[z_idx]),
        (vol[:, yc, :],  crops.label_mask[:, yc, :]),
        (vol[:, :, xc],  crops.label_mask[:, :, xc]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# feature-panel helpers
# ──────────────────────────────────────────────────────────────────────────────

_PROBA_ORDER = ("bad", "bad_ok", "good", "merged")
_TOP_FEATURE_N = 10
_V5D_4CLASS_MODEL = CACHE_DIR / "roi_quality_stage2_4class_v5d.txt"


def _predicted_class(proba_4class: dict) -> tuple[str, float]:
    if not proba_4class:
        return "?", 0.0
    items = [(c, float(proba_4class.get(c, 0.0))) for c in _PROBA_ORDER]
    cls, p = max(items, key=lambda kv: kv[1])
    return cls, p


class _TopFeaturesCache:
    """Cache top-N (name, gain) from a LightGBM model file.

    Re-reads whenever the file's mtime changes so the GUI can surface
    fresh importances after a retrain without restarting.
    """

    def __init__(self, model_path: Path, n: int = _TOP_FEATURE_N) -> None:
        self.path = model_path
        self.n = n
        self._mtime: float | None = None
        self._top: list[tuple[str, float]] = []
        self._error: str | None = None

    def get(self) -> tuple[list[tuple[str, float]], bool]:
        """Return (top_features, changed_since_last_get)."""
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            changed = bool(self._top)
            if changed:
                self._top = []
                self._mtime = None
            self._error = f"model file missing: {self.path.name}"
            return self._top, changed
        if mtime == self._mtime:
            return self._top, False
        try:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=str(self.path))
            names = booster.feature_name()
            gains = booster.feature_importance(importance_type="gain")
            top = sorted(zip(names, gains), key=lambda kv: -float(kv[1]))[: self.n]
            self._top = [(nm, float(g)) for nm, g in top]
            self._mtime = mtime
            self._error = None
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
        return self._top, True

    @property
    def error(self) -> str | None:
        return self._error


def _format_feature_value(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    try:
        return f"{float(v):.3g}"
    except (TypeError, ValueError):
        return str(v)


# ──────────────────────────────────────────────────────────────────────────────
# visual constants
# ──────────────────────────────────────────────────────────────────────────────

_LABEL_BLINK_COLORS = {
    "good":   "#2ecc71",
    "bad":    "#e74c3c",
    "bad_ok": "#e67e22",
    "merged": "#9b59b6",
    "unsure": "#f1c40f",
}
_BLINK_SECONDS = 0.15


def _now_iso() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ──────────────────────────────────────────────────────────────────────────────
# main GUI class
# ──────────────────────────────────────────────────────────────────────────────

class StandaloneLabeller:
    """Single-window 3-D orthoview labelling app."""

    def __init__(
        self,
        sids: list[str],
        n_rois: int,
        score_band: tuple[float, float],
        reviewer: str,
        seed: int,
        candidates: dict[str, list[int]] | None = None,
    ) -> None:
        if not sids:
            raise ValueError("at least one sid is required")
        self.sids = list(sids)
        self.subject_idx = 0
        self.n_rois = n_rois
        self.score_band = score_band
        self.reviewer = reviewer
        self.seed = seed
        self.token = uuid.uuid4().hex[:8]
        self.candidates = candidates or {}

        self.rois: list[_ROI] = []
        self.idx = 0
        self.history: list[dict] = []
        self.crops: _RoiCrops | None = None
        self._suppress_subject_radio = False

        self._z_idx = 0
        self._y_idx = 0
        self._x_idx = 0

        self._channel = "405"
        self._mip = False
        self._mode = "new"
        self._prior_labels: dict[int, str] = {}

        self._top_feats = _TopFeaturesCache(_V5D_4CLASS_MODEL, n=_TOP_FEATURE_N)

        self._load_subject()
        for _ in range(len(self.sids)):
            if self.rois:
                break
            self.subject_idx = (self.subject_idx + 1) % len(self.sids)
            self._load_subject()
        if not self.rois:
            raise RuntimeError(
                f"No ROIs to label in any subject ({self.sids}) "
                f"within score band {self.score_band}."
            )

        # ---- figure layout ----
        title = (
            f"ROI labeller -- sid={self.sids[0]}"
            if len(self.sids) == 1
            else f"ROI labeller -- {len(self.sids)} subjects"
        )
        self.fig = plt.figure(f"{title}  reviewer={reviewer}", figsize=(15.5, 9.0))

        self.ax_xy = self.fig.add_axes([0.030, 0.45, 0.225, 0.50])
        self.ax_xz = self.fig.add_axes([0.265, 0.45, 0.225, 0.50])
        self.ax_yz = self.fig.add_axes([0.500, 0.45, 0.225, 0.50])
        for ax, ttl in [
            (self.ax_xy, "xy (axial)"),
            (self.ax_xz, "xz (coronal)"),
            (self.ax_yz, "yz (sagittal)"),
        ]:
            ax.set_title(ttl, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        self._im_xy = self.ax_xy.imshow(np.zeros((1, 1)), cmap="gray", vmin=0, vmax=1)
        self._im_xz = self.ax_xz.imshow(
            np.zeros((1, 1)), cmap="gray", vmin=0, vmax=1, aspect="auto"
        )
        self._im_yz = self.ax_yz.imshow(
            np.zeros((1, 1)), cmap="gray", vmin=0, vmax=1, aspect="auto"
        )
        self._contour_artists: list = []
        self._crosshair_artists: list = []

        self.ax_header = self.fig.add_axes([0.030, 0.395, 0.69, 0.04])
        self.ax_header.axis("off")
        self.text_header = self.ax_header.text(
            0, 0.5, "", va="center", ha="left", fontsize=11
        )

        n_sub = max(2, len(self.sids))
        sub_h = max(0.08, min(0.16, 0.026 * n_sub))
        self.ax_subject = self.fig.add_axes([0.030, 0.345 - sub_h, 0.105, sub_h])
        self.ax_subject.set_title("subject (n/p)", fontsize=8.5, loc="left", pad=2)
        self.subject_radio = RadioButtons(
            self.ax_subject, tuple(self.sids), active=self.subject_idx
        )
        self.subject_radio.on_clicked(self._on_subject_radio)
        if len(self.sids) <= 1:
            self.ax_subject.set_visible(False)

        self.ax_radio = self.fig.add_axes([0.150, 0.235, 0.085, 0.10])
        self.ax_radio.set_title("channel (1/2/3)", fontsize=8.5, loc="left", pad=2)
        self.radio = RadioButtons(self.ax_radio, ("405", "488", "overlay"), active=0)
        self.radio.on_clicked(self._on_channel)

        self.ax_mode = self.fig.add_axes([0.245, 0.260, 0.085, 0.075])
        self.ax_mode.set_title("mode", fontsize=8.5, loc="left", pad=2)
        self.mode_radio = RadioButtons(self.ax_mode, ("new", "review"), active=0)
        self.mode_radio.on_clicked(self._on_mode)

        self.ax_check = self.fig.add_axes([0.340, 0.275, 0.075, 0.055])
        self.check = CheckButtons(self.ax_check, ("MIP (m)",), actives=[False])
        self.check.on_clicked(self._on_mip)

        self.ax_reviewer = self.fig.add_axes([0.495, 0.285, 0.20, 0.040])
        self.tb_reviewer = TextBox(
            self.ax_reviewer, "reviewer ", initial=self.reviewer
        )
        self.tb_reviewer.on_submit(self._on_reviewer)

        btn_w, btn_h, btn_y = 0.082, 0.055, 0.10
        btn_xs = [0.030, 0.117, 0.204, 0.291, 0.378, 0.465, 0.552, 0.640]
        btn_labels = [
            "Good (g)", "Bad (b)", "Bad-OK (o)", "Merged (e)", "Unsure (u)",
            "Skip (s)", "Undo (z)", "Quit (q)",
        ]
        btn_colors = [
            "#a8e6a8", "#f5b5b5", "#f5d0a3", "#d5b5e5", "#ffe4a3",
            "#dcdcdc", "#dcdcdc", "#dcdcdc",
        ]
        callbacks = [
            lambda _e: self._record("good"),
            lambda _e: self._record("bad"),
            lambda _e: self._record("bad_ok"),
            lambda _e: self._record("merged"),
            lambda _e: self._record("unsure"),
            lambda _e: self._advance(),
            lambda _e: self._undo(),
            lambda _e: self._quit(),
        ]
        self.buttons: list[Button] = []
        for x, lbl, color, cb in zip(btn_xs, btn_labels, btn_colors, callbacks):
            ax = self.fig.add_axes([x, btn_y, btn_w, btn_h])
            b = Button(ax, lbl, color=color, hovercolor="#bcd")
            b.on_clicked(cb)
            self.buttons.append(b)

        self.ax_hist = self.fig.add_axes([0.030, 0.030, 0.69, 0.04])
        self.ax_hist.axis("off")
        self.text_hist = self.ax_hist.text(
            0, 0.5, "", va="center", ha="left", fontsize=9, color="#444"
        )

        self.ax_feat = self.fig.add_axes([0.760, 0.030, 0.230, 0.93])
        self.ax_feat.axis("off")
        self.ax_feat.set_title(
            f"v5d top-{_TOP_FEATURE_N} (4-class gain)", fontsize=10, loc="left"
        )
        self.text_feat = self.ax_feat.text(
            0.0, 0.985, "", va="top", ha="left",
            fontsize=9.5, family="monospace", clip_on=True,
            transform=self.ax_feat.transAxes,
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._render()

    # ---------------------------------------------------------------- events

    def _on_key(self, event) -> None:
        # Don't steal keys while the reviewer TextBox is being edited.
        if getattr(self.tb_reviewer, "capturekeystrokes", False):
            return
        k = (event.key or "").lower()
        if k == "g":
            self._record("good")
        elif k == "b":
            self._record("bad")
        elif k == "o":
            self._record("bad_ok")
        elif k == "e":
            self._record("merged")
        elif k == "u":
            self._record("unsure")
        elif k == "s":
            self._advance()
        elif k == "z":
            self._undo()
        elif k == "q":
            self._quit()
        elif k in ("j", "down"):
            self._step_z(-1)
        elif k in ("k", "up"):
            self._step_z(+1)
        elif k == "m":
            self.check.set_active(0)
        elif k in ("1", "2", "3"):
            self.radio.set_active({"405": 0, "488": 1, "overlay": 2}[
                ["405", "488", "overlay"][int(k) - 1]
            ])
        elif k == "n":
            self._switch_subject(+1)
        elif k == "p":
            self._switch_subject(-1)

    def _on_scroll(self, event) -> None:
        delta = 0
        if getattr(event, "step", None):
            delta = int(np.sign(event.step))
        elif getattr(event, "button", None) == "up":
            delta = +1
        elif getattr(event, "button", None) == "down":
            delta = -1
        if not delta:
            return
        if event.inaxes is self.ax_xz:
            self._step_y(delta)
        elif event.inaxes is self.ax_yz:
            self._step_x(delta)
        else:
            self._step_z(delta)

    def _on_channel(self, label) -> None:
        self._channel = label
        self._redraw_orthoview()

    def _on_mip(self, _label) -> None:
        self._mip = bool(self.check.get_status()[0])
        self._redraw_orthoview()

    def _on_subject_radio(self, label) -> None:
        if self._suppress_subject_radio:
            return
        try:
            target = self.sids.index(label)
        except ValueError:
            return
        self._select_subject(target)

    def _on_mode(self, label) -> None:
        if self._mode == label:
            return
        self._mode = label
        self._load_subject()
        self._render()

    def _on_reviewer(self, text) -> None:
        new = (text or "").strip() or "anonymous"
        if new == self.reviewer:
            return
        self.reviewer = new
        self._update_header_only()

    # ---------------------------------------------------------------- subject

    def _load_subject(self) -> None:
        sid = self.sids[self.subject_idx]
        log = _load_label_log()
        rng = np.random.default_rng(self.seed + self.subject_idx)
        if self._mode == "review":
            rois, prior = _sample_labelled(sid, self.n_rois, rng, log)
            self.rois = rois
            self._prior_labels = prior
        else:
            skip = (
                set(log.loc[log["sid"] == sid, "hcr_id"].astype(int))
                if not log.empty
                else set()
            )
            if sid in self.candidates:
                self.rois = _sample_from_candidates(
                    sid, self.candidates[sid], self.n_rois, rng, skip
                )
            else:
                self.rois = _sample_uncertain(
                    sid, self.n_rois, self.score_band, rng, skip
                )
            self._prior_labels = {}
        self.idx = 0
        self.crops = None

    def _select_subject(self, target_idx: int) -> None:
        if target_idx == self.subject_idx or not (0 <= target_idx < len(self.sids)):
            return
        self.subject_idx = target_idx
        self._load_subject()
        self._suppress_subject_radio = True
        try:
            self.subject_radio.set_active(target_idx)
        finally:
            self._suppress_subject_radio = False
        self._render()

    def _switch_subject(self, delta: int) -> None:
        if len(self.sids) <= 1:
            return
        self._select_subject((self.subject_idx + delta) % len(self.sids))

    # ---------------------------------------------------------------- navigation

    def _step_z(self, delta: int) -> None:
        if self.crops is None:
            return
        Z = self.crops.shape[0]
        new = int(np.clip(self._z_idx + delta, 0, Z - 1))
        if new == self._z_idx:
            return
        self._z_idx = new
        self._redraw_orthoview()

    def _step_y(self, delta: int) -> None:
        if self.crops is None:
            return
        Y = self.crops.shape[1]
        new = int(np.clip(self._y_idx + delta, 0, Y - 1))
        if new == self._y_idx:
            return
        self._y_idx = new
        self._redraw_orthoview()

    def _step_x(self, delta: int) -> None:
        if self.crops is None:
            return
        X = self.crops.shape[2]
        new = int(np.clip(self._x_idx + delta, 0, X - 1))
        if new == self._x_idx:
            return
        self._x_idx = new
        self._redraw_orthoview()

    def _current(self) -> _ROI | None:
        if self.idx >= len(self.rois):
            return None
        return self.rois[self.idx]

    # ---------------------------------------------------------------- labelling

    def _record(self, label: str) -> None:
        roi = self._current()
        if roi is None:
            return
        pred_cls, pred_p = _predicted_class(roi.proba_4class)
        rec = {
            "ts": _now_iso(),
            "sid": roi.sid,
            "hcr_id": roi.hcr_id,
            "label": label,
            "binary_score_v5d": roi.score,
            "proba_v5d": dict(roi.proba_4class),
            "predicted_class_v5d": pred_cls,
            "predicted_p_v5d": pred_p,
            "reviewer": self.reviewer,
            "session_token": self.token,
        }
        _append_label(rec)
        self.history.append(rec)
        self._blink_label_color(label)
        self._advance()

    def _blink_label_color(self, label: str) -> None:
        color = _LABEL_BLINK_COLORS.get(label)
        if color is None:
            return
        axes = (self.ax_xy, self.ax_xz, self.ax_yz)
        try:
            for ax in axes:
                for sp in ax.spines.values():
                    sp.set_edgecolor(color)
                    sp.set_linewidth(4.0)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(_BLINK_SECONDS)
        finally:
            for ax in axes:
                for sp in ax.spines.values():
                    sp.set_edgecolor("black")
                    sp.set_linewidth(1.0)
            self.fig.canvas.draw_idle()

    def _advance(self) -> None:
        self.idx += 1
        self._render()

    def _undo(self) -> None:
        if not self.history or self.idx == 0:
            return
        last = self.history.pop()
        tomb = {
            "ts": _now_iso(),
            "sid": last["sid"],
            "hcr_id": last["hcr_id"],
            "label": "_undone_",
            "undoes": last,
            "reviewer": self.reviewer,
            "session_token": self.token,
        }
        _append_label(tomb)
        self.idx = max(0, self.idx - 1)
        self._render()

    def _quit(self) -> None:
        self.text_header.set_text(
            f"Session ended -- {len(self.history)} labels written to "
            f"{LABEL_LOG}.  You can close this window."
        )
        self.fig.canvas.draw_idle()
        plt.close(self.fig)

    # ---------------------------------------------------------------- features panel

    def _refresh_feature_panel(self, roi: _ROI | None) -> None:
        top, _ = self._top_feats.get()
        n_used = len(top)
        title = (
            f"v5d top-{n_used} (4-class gain)" if n_used > 0 else "v5d top features (no model)"
        )
        self.ax_feat.set_title(title, fontsize=10, loc="left")
        if roi is None:
            self.text_feat.set_text("")
            return
        lines: list[str] = ["v5d (stage-2):"]
        pred_cls, pred_p = _predicted_class(roi.proba_4class)
        lines.append(f"  binary score (good|bad_ok)  {roi.score:.3f}")
        lines.append(f"  predicted class             {pred_cls} ({pred_p:.3f})")
        for c in _PROBA_ORDER:
            if c in roi.proba_4class:
                lines.append(f"  p_{c:<24s}  {roi.proba_4class[c]:.3f}")
        lines.append("")
        if not top:
            err = self._top_feats.error or "no v5d 4-class model file"
            lines.append(f"top-{_TOP_FEATURE_N} features: ({err})")
        else:
            lines.append(f"top-{n_used} features (gain rank -> value):")
            name_w = min(max(len(nm) for nm, _ in top), 32)
            for i, (nm, gain) in enumerate(top, 1):
                v = roi.features.get(nm)
                txt = _format_feature_value(v)
                short = nm if len(nm) <= name_w else nm[: name_w - 1] + "~"
                lines.append(f"  {i:>2}. {short:<{name_w}s}  {txt}  (g={gain:.0f})")
        self.text_feat.set_text("\n".join(lines))

    # ---------------------------------------------------------------- render

    def _format_header(self) -> str:
        roi = self._current()
        if roi is None or self.crops is None:
            return ""
        Z, Y, X = self.crops.shape
        prior = self._prior_label_text(roi)
        pred_cls, pred_p = _predicted_class(roi.proba_4class)
        return (
            f"sid={roi.sid}   hcr_id={roi.hcr_id}   "
            f"v5d bin={roi.score:.3f}  pred={pred_cls}({pred_p:.2f})   "
            f"shape (Z x Y x X)={Z} x {Y} x {X}   "
            f"pos (z,y,x)=({self._z_idx},{self._y_idx},{self._x_idx})   "
            f"[{self.idx + 1} / {len(self.rois)}]"
            f"{self._subject_tag()}   reviewer={self.reviewer}{prior}"
        )

    def _update_header_only(self) -> None:
        if self._current() is None or self.crops is None:
            return
        self.text_header.set_text(self._format_header())
        self.fig.canvas.draw_idle()

    def _prior_label_text(self, roi: _ROI) -> str:
        if self._mode != "review":
            return ""
        prev = self._prior_labels.get(roi.hcr_id)
        if prev is None:
            return ""
        color_label = {
            "good": "GOOD", "bad": "BAD", "bad_ok": "BAD-OK",
            "merged": "MERGED", "unsure": "UNSURE",
        }.get(prev, prev.upper())
        return f"   prior={color_label}"

    def _subject_tag(self) -> str:
        if len(self.sids) <= 1:
            return ""
        return f"  subj {self.subject_idx + 1}/{len(self.sids)} (n,p)"

    def _clear_overlay(self) -> None:
        for art in self._contour_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._contour_artists = []
        for art in self._crosshair_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._crosshair_artists = []

    def _draw_overlay(self, xy_mask, xz_mask, yz_mask) -> None:
        for ax, m in [
            (self.ax_xy, xy_mask),
            (self.ax_xz, xz_mask),
            (self.ax_yz, yz_mask),
        ]:
            if m.any():
                cs = ax.contour(m.astype(np.uint8), levels=[0.5], colors="red", linewidths=1.0)
                self._contour_artists.append(cs)
        if self._mip or self.crops is None:
            return
        self._crosshair_artists.extend([
            self.ax_xy.axvline(self._x_idx, color="cyan", lw=0.5, alpha=0.7),
            self.ax_xy.axhline(self._y_idx, color="cyan", lw=0.5, alpha=0.7),
            self.ax_xz.axvline(self._x_idx, color="cyan", lw=0.5, alpha=0.7),
            self.ax_xz.axhline(self._z_idx, color="cyan", lw=0.5, alpha=0.7),
            self.ax_yz.axvline(self._y_idx, color="cyan", lw=0.5, alpha=0.7),
            self.ax_yz.axhline(self._z_idx, color="cyan", lw=0.5, alpha=0.7),
        ])

    def _redraw_orthoview(self) -> None:
        if self.crops is None:
            return
        try:
            (xy_img, xy_mask), (xz_img, xz_mask), (yz_img, yz_mask) = _planes_for_channel(
                self.crops,
                channel=self._channel,
                mip=self._mip,
                z_idx=self._z_idx,
                y_idx=self._y_idx,
                x_idx=self._x_idx,
            )
            self._im_xy.set_data(xy_img)
            self._im_xz.set_data(xz_img)
            self._im_yz.set_data(yz_img)
            self._clear_overlay()
            self._draw_overlay(xy_mask, xz_mask, yz_mask)
            self.text_header.set_text(self._format_header())
            _, changed = self._top_feats.get()
            if changed:
                self._refresh_feature_panel(self._current())
            self.fig.canvas.draw_idle()
        except Exception as exc:
            import traceback as _tb
            print(
                f"\n[redraw_orthoview] error on idx={self.idx} "
                f"sid={self.sids[self.subject_idx]} "
                f"channel={self._channel} mip={self._mip} "
                f"pos (z,y,x)=({self._z_idx},{self._y_idx},{self._x_idx}) "
                f"shape={getattr(self.crops, 'shape', '?')}",
                file=sys.stderr,
            )
            _tb.print_exc()
            self.text_header.set_text(
                f"redraw error: {type(exc).__name__}: {exc}.  "
                f"Press s to skip, z to undo, or change channel."
            )
            self.fig.canvas.draw_idle()

    def _resize_axes_for_crop(self) -> None:
        Z, Y, X = self.crops.shape
        self._im_xy.set_extent((-0.5, X - 0.5, Y - 0.5, -0.5))
        self.ax_xy.set_xlim(-0.5, X - 0.5)
        self.ax_xy.set_ylim(Y - 0.5, -0.5)
        self._im_xz.set_extent((-0.5, X - 0.5, Z - 0.5, -0.5))
        self.ax_xz.set_xlim(-0.5, X - 0.5)
        self.ax_xz.set_ylim(Z - 0.5, -0.5)
        self._im_yz.set_extent((-0.5, Y - 0.5, Z - 0.5, -0.5))
        self.ax_yz.set_xlim(-0.5, Y - 0.5)
        self.ax_yz.set_ylim(Z - 0.5, -0.5)

    def _init_indices_for_crop(self) -> None:
        if self.crops is None:
            return
        Z, Y, X = self.crops.shape
        self._z_idx = int(np.clip(self.crops.z_anchor, 0, max(0, Z - 1)))
        self._y_idx = int(np.clip(self.crops.y_anchor, 0, max(0, Y - 1)))
        self._x_idx = int(np.clip(self.crops.x_anchor, 0, max(0, X - 1)))

    def _update_history_strip(self) -> None:
        items = []
        for h in self.history[-6:]:
            items.append(f"{h['label']}(#{h['hcr_id']})")
        self.text_hist.set_text("last: " + ", ".join(items) if items else "")

    def _render(self) -> None:
        roi = self._current()
        if roi is None:
            n_g = sum(1 for h in self.history if h["label"] == "good")
            n_b = sum(1 for h in self.history if h["label"] == "bad")
            n_o = sum(1 for h in self.history if h["label"] == "bad_ok")
            n_m = sum(1 for h in self.history if h["label"] == "merged")
            n_u = sum(1 for h in self.history if h["label"] == "unsure")
            cur_sid = self.sids[self.subject_idx]
            tail = "  press n for next subject" if len(self.sids) > 1 else ""
            self.text_header.set_text(
                f"sid={cur_sid} DONE -- session totals: {len(self.history)} labels "
                f"({n_g} good / {n_b} bad / {n_o} bad-OK / {n_m} merged / {n_u} unsure)."
                f"{self._subject_tag()}{tail}"
            )
            self.text_feat.set_text("")
            self._clear_overlay()
            self._im_xy.set_data(np.zeros((1, 1)))
            self._im_xz.set_data(np.zeros((1, 1)))
            self._im_yz.set_data(np.zeros((1, 1)))
            self._update_history_strip()
            self.fig.canvas.draw_idle()
            return

        try:
            self.crops = _load_roi_crops(roi)
            Z, Y, X = self.crops.shape
            if Z == 0 or Y == 0 or X == 0:
                raise ValueError(f"degenerate crop shape (Z x Y x X)={Z} x {Y} x {X}")
            self._init_indices_for_crop()
            self._refresh_feature_panel(roi)
            self._resize_axes_for_crop()
            self._redraw_orthoview()
            self._update_history_strip()
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            self.text_header.set_text(
                f"sid={roi.sid}  hcr_id={roi.hcr_id} #{self.idx + 1}/{len(self.rois)} "
                f"-- render error: {type(exc).__name__}: {exc}.  "
                f"Press s to skip, z to undo."
            )
            self.text_feat.set_text("")
            self._clear_overlay()
            self._im_xy.set_data(np.zeros((1, 1)))
            self._im_xz.set_data(np.zeros((1, 1)))
            self._im_yz.set_data(np.zeros((1, 1)))
            self.fig.canvas.draw_idle()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_sids(spec: str) -> list[str]:
    spec = spec.strip()
    if spec.lower() == "all":
        return list(_BENCHMARK_SUBJECTS)
    out = [s.strip() for s in spec.split(",") if s.strip()]
    if not out:
        raise argparse.ArgumentTypeError(f"invalid --sid value: {spec!r}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone HCR ROI labelling GUI (no Jupyter).",
    )
    p.add_argument(
        "--sid", default=None, type=_parse_sids,
        help=(
            "Subject ID, e.g. 788406.  Comma-separated list or 'all' cycles "
            "through subjects via n / p keys.  "
            f"Default: all 6 benchmark subjects ({', '.join(_BENCHMARK_SUBJECTS)})."
        ),
    )
    p.add_argument("--n-rois", type=int, default=80,
                   help="ROIs to sample per subject (default 80)")
    p.add_argument("--score-min", type=float, default=0.3,
                   help="Lower bound of v5d binary-score uncertain band (default 0.3)")
    p.add_argument("--score-max", type=float, default=0.7,
                   help="Upper bound of v5d binary-score uncertain band (default 0.7)")
    p.add_argument("--reviewer", default="anonymous",
                   help="Reviewer name recorded in every label")
    p.add_argument("--seed", type=int, default=20260429,
                   help="RNG seed for reproducible sampling")
    p.add_argument(
        "--candidates", default=None, type=Path,
        help=(
            "CSV with 'sid,hcr_id' columns.  When set, ROIs are walked in "
            "CSV order (top-priority first) instead of random uncertain-band "
            "sampling.  Already-labelled rows are still skipped."
        ),
    )
    p.add_argument(
        "--label-log", default=None, type=Path,
        help=(
            f"Path to the JSONL label log.  "
            f"Default: {_DEFAULT_LABEL_LOG}"
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global LABEL_LOG

    args = parse_args(argv)

    if args.label_log is not None:
        LABEL_LOG = args.label_log.resolve()
    # else LABEL_LOG keeps its module-level default

    sids: list[str] = args.sid if args.sid else list(_BENCHMARK_SUBJECTS)
    candidates: dict[str, list[int]] | None = None
    if args.candidates is not None:
        df = pd.read_csv(args.candidates)
        df["sid"] = df["sid"].astype(str)
        candidates = {
            sid: g["hcr_id"].astype(int).tolist()
            for sid, g in df.groupby("sid", sort=False)
        }
        print(
            f"  candidates loaded from {args.candidates}: "
            + ", ".join(f"{s}={len(v)}" for s, v in candidates.items())
        )

    print(f"ROI labeller  backend={matplotlib.get_backend()}")
    print(
        f"  sids={sids}  n_rois={args.n_rois}  "
        f"band=({args.score_min}, {args.score_max})  reviewer={args.reviewer}"
        + ("  [CANDIDATES MODE]" if candidates else "")
    )
    print(f"  label log -> {LABEL_LOG}")

    app = StandaloneLabeller(
        sids=sids,
        n_rois=args.n_rois,
        score_band=(args.score_min, args.score_max),
        reviewer=args.reviewer,
        seed=args.seed,
        candidates=candidates,
    )
    print(
        f"  loaded {len(app.rois)} ROIs for sid={sids[app.subject_idx]}"
    )
    print(
        "  shortcuts: g/b/o/e/u label (good/bad/bad_ok/merged/unsure)  "
        "s skip  z undo  q quit  "
        "j/k or mouse-wheel scroll z  m MIP  1/2/3 channel  n/p next/prev subject"
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
