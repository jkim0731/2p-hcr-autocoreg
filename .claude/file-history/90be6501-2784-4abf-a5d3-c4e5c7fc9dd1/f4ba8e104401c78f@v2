"""Fit a landmark-perfect anisotropic similarity on each subject's GT pairs and
report the per-GT distance distribution. This is the ICP ceiling we should
compare against.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01/lib")

from benchmark_data_loader import load_subject, cz_px_to_um, hcr_px_to_um  # noqa
from benchmark_analysis import fit_anisotropic_similarity  # noqa

SUBJECTS = ["788406", "755252", "767018", "767022", "782149", "790322"]


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


rows = []
for subj in SUBJECTS:
    s = load_subject(subj)
    cz_gt, hc_gt = gt_pairs(s)
    fit = fit_anisotropic_similarity(cz_gt, hc_gt)
    pred = ((cz_gt * fit.scales) @ fit.R.T) + fit.translation
    d = np.linalg.norm(pred - hc_gt, axis=1)
    rows.append(dict(
        subject=subj, n_gt=len(cz_gt),
        sxy=float(np.mean([fit.scales[0], fit.scales[1]])),
        sz=float(fit.scales[2]),
        rms=float(fit.rms_um),
        median=float(np.median(d)),
        n_lt5=int((d < 5).sum()),
        n_lt10=int((d < 10).sum()),
        n_lt50=int((d < 50).sum()),
    ))

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("ceiling.csv", index=False)
