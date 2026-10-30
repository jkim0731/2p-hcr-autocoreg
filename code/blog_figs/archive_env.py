"""Import for side effects: puts the read-only session-16 archive helpers on sys.path and applies the loader fixes
(skip *_HCR-ROI-label_* assets, prefer the HCR round that has centroids, session outputs from the archive)."""
import sys
from pathlib import Path
ARCH = Path('/root/capsule/data/claude-data_ophys-mfish-autocoreg_soma-print_260616/sessions')
S16 = ARCH / '16_intermediate_summary_materials'
for p in (str(S16), str(ARCH / '15_geom_features')):
    if p not in sys.path: sys.path.insert(0, p)
import matplotlib
matplotlib.use('Agg')
import _common
import benchmark_data_loader as bdl


def _hcr_dir(sid):
    m = sorted(x for x in bdl.DATA_DIR.glob(f"HCR_{sid}_*_processed_*") if 'HCR-ROI-label' not in x.name)
    m = [x for x in m if (x / 'cell_body_segmentation' / 'cell_centroids.npy').exists()] or m
    return m[-1]
bdl._find_hcr_dir = _hcr_dir
_common.S15 = ARCH / '15_geom_features'; _common.S11 = ARCH / 'v3_S11_roi_quality'
FIGDIR = Path('/scratch/tmp/site/posts/2026-10-auto-coregistration-1-pan-inhibitory/figures')
