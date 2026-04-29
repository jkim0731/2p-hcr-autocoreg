"""v2-S02 iter 6 — frozen t_z (no search, no per-sz coupling).

Per user direction (2026-04-27): t_z is fully derived from
`surface_registration_v2` + `surfaces_iter08` (the LP's pia-anchor
formula evaluated once at sz_init).  Sweep sz only.

Implementation: pass ``tz_search_half_um=0, couple_tz=False`` to
`estimate_sz_image_ncc`; t_z stays at `lp.translation[0]` for the whole
sweep; the smoothed_voxel NCC is computed only inside the per-sz warped
CZ FOV (b1 — slab actually filled by the warped CZ at that sz).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_02/lib")
sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/sessions/03c_onset_features/iterations")

import numpy as np
import pandas as pd
from benchmark_data_loader import load_subject, BENCHMARK_SUBJECTS, landmark_pairs_um
from benchmark_analysis import fit_anisotropic_similarity
from sz_estimator import estimate_sz_image_ncc

OUT = Path('/root/capsule/code/full_automatic_execution_02/sessions/v2_S02_sz_image_sweep')

rows = []
for sid in BENCHMARK_SUBJECTS:
    print(f'\n=== {sid} ===', flush=True)
    s = load_subject(sid)
    cz_xyz, hcr_xyz = landmark_pairs_um(s)
    fit = fit_anisotropic_similarity(cz_xyz[:, [2, 1, 0]], hcr_xyz[:, [2, 1, 0]])
    sz_gt = float(fit.scales[0])
    sxy_gt = float(fit.scales[1])

    t0 = time.time()
    res = estimate_sz_image_ncc(
        s,
        sz_grid=np.arange(1.5, 4.01, 0.10),
        scoring='smoothed_voxel',
        couple_tz=False,        # do not recompute tz per sz
        tz_search_half_um=0.0,  # do not search tz at all
        verbose=False,
    )
    elapsed = time.time() - t0

    err = res.sz_peak - sz_gt if res.sz_peak is not None else float('nan')
    print(f'  sz_lp={res.sz_lp:.3f}  sz_peak={res.sz_peak}  sz_gt={sz_gt:.3f}  '
          f'err={err:+.3f}  passed={res.passed}  reason={res.fail_reason}',
          flush=True)
    print(f'  ratio={res.peak_ratio:.3f}  half_width={res.half_width:.3f}  '
          f'tz_off={res.tz_offset_um}  ({elapsed:.0f}s)', flush=True)

    rows.append({
        'subject_id': sid,
        'sz_gt': sz_gt,
        'sxy_gt': sxy_gt,
        'sz_lp': res.sz_lp,
        'sz_peak': res.sz_peak,
        'ncc_peak': res.ncc_peak,
        'ncc_median': res.ncc_median,
        'peak_ratio': res.peak_ratio,
        'half_width': res.half_width,
        'passed': res.passed,
        'fail_reason': res.fail_reason,
        'tz_offset_um': res.tz_offset_um,
        'sz_err_vs_gt': err,
        'runtime_s': round(elapsed, 1),
    })

    sweep = pd.DataFrame({
        'sz': res.sz_grid,
        'ncc': res.ncc_grid,
        'tz_offset_um': [d.get('tz_offset_um', np.nan) for d in res.diagnostics['sweep_rows']],
    })
    sweep.to_csv(OUT / f'sweep_iter6_{sid}.csv', index=False)

df = pd.DataFrame(rows)
df.to_csv(OUT / 'results_iter6.csv', index=False)
print('\n=== summary (iter 6 — frozen tz) ===')
print(df[['subject_id', 'sz_gt', 'sz_peak', 'sz_err_vs_gt', 'peak_ratio',
         'half_width', 'passed', 'fail_reason', 'runtime_s']].to_string(index=False))
print(f'\npassed: {df["passed"].sum()}/{len(df)}')
print(f'mean abs err: {df["sz_err_vs_gt"].abs().mean():.3f}')
print(f'subjects within ±0.30 of GT: {(df["sz_err_vs_gt"].abs() <= 0.30).sum()}/{len(df)}')
