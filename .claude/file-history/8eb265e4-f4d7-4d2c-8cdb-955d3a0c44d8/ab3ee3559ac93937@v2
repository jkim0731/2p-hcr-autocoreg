"""Figure: M1/M3 scale estimates vs GT (session 07c)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SESSION = Path("/root/capsule/code/sessions/07c_gfp_bic_cz_density")
with open(SESSION / "results.json") as f:
    data = json.load(f)

rows = [r for r in data["scales"] if "sxy_gt" in r]
sids = [r["subject"] for r in rows]
sxy_gt = np.array([r["sxy_gt"] for r in rows])
sz_gt = np.array([r["sz_gt"] for r in rows])
sxy_m1 = np.array([r["sxy_m1"] for r in rows])
sz_m1 = np.array([r["sz_m1"] for r in rows])
sxy_m3 = np.array([r["sxy_m3"] for r in rows])
sz_m3 = np.array([r["sz_m3"] for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
xs = np.arange(len(sids))
w = 0.25
ax.bar(xs - w, sxy_gt, w, color="black", label="GT")
ax.bar(xs, sxy_m1, w, color="#268bd2", label="M1 k-NN")
ax.bar(xs + w, sxy_m3, w, color="#cb4b16", label="M3 span ratio")
ax.set_xticks(xs)
ax.set_xticklabels(sids, rotation=20)
ax.set_ylabel("sxy")
ax.set_title("sxy: M1/M3 vs GT  (±5 % band)")
for i, g in enumerate(sxy_gt):
    ax.fill_between([i - w * 1.7, i + w * 1.7], g * 0.95, g * 1.05,
                    color="black", alpha=0.08, lw=0)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(xs - w, sz_gt, w, color="black", label="GT")
ax.bar(xs, sz_m1, w, color="#268bd2", label="M1 k-NN")
ax.bar(xs + w, sz_m3, w, color="#cb4b16", label="M3 span ratio")
ax.set_xticks(xs)
ax.set_xticklabels(sids, rotation=20)
ax.set_ylabel("sz")
ax.set_title("sz: M1/M3 vs GT  (±5 % band)")
for i, g in enumerate(sz_gt):
    ax.fill_between([i - w * 1.7, i + w * 1.7], g * 0.95, g * 1.05,
                    color="black", alpha=0.08, lw=0)
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = SESSION / "figures" / "scales_comparison.png"
fig.savefig(out, dpi=140)
print(f"Wrote {out}")
