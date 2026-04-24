"""P5 — Fused / Partial Gromov-Wasserstein.

Solve an FGW coupling between CZ and HCR centroid sets using pairwise
intra-distances for the GW side and F6 features for the Wasserstein side.
Read off the matching by argmax per row of the transport plan with a
coupling-mass confidence.

Requires the `POT` package (`ot`).
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench.harness import register_candidate, CoregResult, TransformDescriptor
from lib.cell_features import extract_cell_features, invariant_feature_mask
from lib.centroid_helpers import centroids_um

try:
    import ot  # POT
    _HAS_OT = True
except Exception:
    _HAS_OT = False


@register_candidate("P5")
def run_p5(s, *, alpha=0.5, eps=0.05, n_hcr_max=3000, rng_seed=0) -> CoregResult:
    if not _HAS_OT:
        return CoregResult(pd.DataFrame(), 0.0,
                           diagnostics={"error": "POT (ot) not available"})

    rng = np.random.default_rng(rng_seed)

    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")

    from lib.centroid_helpers import default_warmstart_zyx
    cz_init, _ws = default_warmstart_zyx(cz_um, hcr_um)
    cz_c = cz_um.mean(0); hcr_c = hcr_um.mean(0)
    R0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)

    # Crop HCR cells around the warped CZ cloud centroid (post-warm-start).
    cz_init_c = cz_init.mean(0)
    cz_init_range = cz_init.max(0) - cz_init.min(0)
    bbox_half = cz_init_range * np.array([2.0, 2.0, 2.0]) / 2 + 100.0  # generous
    mask = np.all(np.abs(hcr_um - cz_init_c) < bbox_half, axis=1)
    hcr_sel = np.where(mask)[0]
    if len(hcr_sel) > n_hcr_max:
        hcr_sel = rng.choice(hcr_sel, size=n_hcr_max, replace=False)
    hcr_um_sub = hcr_um[hcr_sel]
    hcr_ids_sub = hcr_ids[hcr_sel]

    Fc, names, _ = extract_cell_features(s, "cz")
    Fg, _, _ = extract_cell_features(s, "hcr_gfp")
    inv = invariant_feature_mask(names)
    keep = inv & ~np.isnan(Fc).any(0) & ~np.isnan(Fg).any(0)
    mu = np.nanmean(Fg[:, keep], 0); sd = np.nanstd(Fg[:, keep], 0) + 1e-6
    Fcn = (Fc[:, keep] - mu) / sd
    Fgn = (Fg[hcr_sel][:, keep] - mu) / sd

    # Intra-distance matrices (µm).  Normalise by median spacing so geometric
    # cost is scale-insensitive.
    from scipy.spatial.distance import pdist, squareform, cdist
    Ccz = squareform(pdist(cz_init))
    Chr = squareform(pdist(hcr_um_sub))
    Ccz /= (np.median(Ccz[Ccz > 0]) + 1e-9)
    Chr /= (np.median(Chr[Chr > 0]) + 1e-9)

    # Feature distance
    M = cdist(Fcn, Fgn)

    n = len(cz_init); m = len(hcr_um_sub)
    a = np.ones(n) / n
    b = np.ones(m) / m

    try:
        T = ot.gromov.fused_gromov_wasserstein(
            M, Ccz, Chr, a, b, loss_fun="square_loss",
            alpha=alpha, verbose=False,
        )
    except Exception as e:
        return CoregResult(pd.DataFrame(), 0.0,
                           diagnostics={"error": f"fgw: {e}"})

    # Read off per CZ: argmax over HCR, use mass as confidence
    rows = []
    row_max = T.max(axis=1) + 1e-12
    for i in range(n):
        j = int(np.argmax(T[i]))
        mass = float(T[i, j] / row_max[i])
        # Accept only if the mass is not trivially tiny
        if T[i, j] < 1e-6:
            continue
        rows.append(dict(
            cz_id=int(cz_ids[i]), hcr_id=int(hcr_ids_sub[j]),
            confidence=float(mass),
            cz_x_um=float(cz_um[i, 2]), cz_y_um=float(cz_um[i, 1]), cz_z_um=float(cz_um[i, 0]),
            hcr_x_um=float(hcr_um_sub[j, 2]), hcr_y_um=float(hcr_um_sub[j, 1]),
            hcr_z_um=float(hcr_um_sub[j, 0]),
        ))

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("confidence", ascending=False)
        df = df.drop_duplicates("hcr_id", keep="first").sort_values("cz_id").reset_index(drop=True)

    transform = TransformDescriptor(
        R=R0, scales=np.ones(3), translation=hcr_c,
        src_mean=cz_c, rotation_deg_z=180.0, kind="fgw",
    )
    return CoregResult(
        pairs_df=df,
        confidence=float(df["confidence"].median()) if len(df) else 0.0,
        transform=transform,
        diagnostics=dict(n_hcr_used=m, alpha=alpha, eps=eps),
    )
