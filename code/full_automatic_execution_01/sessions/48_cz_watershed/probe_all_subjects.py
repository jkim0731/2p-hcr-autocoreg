"""S48/S46-c — run voronoi labels on all 4 subjects, measure stats."""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject  # noqa: E402
from lib.cz_labels import cz_voronoi_labels  # noqa: E402


def main():
    subjects = sys.argv[1:] or ["788406", "755252", "767022", "782149"]
    for subj in subjects:
        s = load_subject(subj)
        n_cells = len(s.cz_centroids)
        t0 = time.time()
        labels = cz_voronoi_labels(s, R_um=8.0)
        wall = time.time() - t0
        sizes = np.bincount(labels.ravel())
        per = sizes[1:][sizes[1:] > 0]
        um3 = float(s.cz_xy_um) ** 2 * float(s.cz_z_um)
        print(f"  {subj}: n_cz={n_cells} labeled={len(per)} "
              f"med={np.median(per)*um3:.0f}µm³ "
              f"p10={np.percentile(per, 10)*um3:.0f} "
              f"p90={np.percentile(per, 90)*um3:.0f} "
              f"wall={wall:.1f}s")


if __name__ == "__main__":
    main()
