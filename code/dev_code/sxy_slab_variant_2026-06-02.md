# sxy slab-variant comparison — 2026-06-02

## Setup

**Baseline** (`sxy_current`): `estimate_sxy_roi_area` logic with `center_fov_quarter=True`, `area_mode='max_xsection'`  
HCR: strict-GFP+ cells, depth ∈ [100 µm, p99], center-¼ HCR FOV.  
CZ: all cells, depth ∈ [D_SKIN_UM, p99_cz].

**New variant** (slab-restricted, full FOV):  
- HCR: strict-GFP+ cells (same population), depth ∈ [0, 100] µm below HCR pia, **full FOV** (no center-¼ restriction)  
- CZ : all cells, depth ∈ [0, 50] µm below CZ pia  
- Area: max_xsection (same as baseline)  
- Slab bounds = `surface_registration_v2.HCR_SLAB` / `CZ_SLAB`  
- HCR depths for slab filter from `s.hcr_centroids` (level-2 parquet, fast)

Two sxy estimates from the variant:  
- `sxy_slab_median`: sqrt(median(area_HCR_slab) / median(area_CZ_slab))  
- `sxy_slab_total` : sqrt(sum(area_HCR_slab)   / sum(area_CZ_slab))  

## Results table

```
sid        sxy_cur  sxy_slab_med  sxy_slab_tot       GT   err_cur%   err_med%   err_tot%  n_hcr_cur  n_hcr_slb  n_cz_cur  n_cz_slb
----------------------------------------------------------------------------------------------------------------------------------
755252      1.6264        1.5350        2.6561   1.6397      -0.81      -6.38     +61.98       1317        252       623        85
767018      1.7740        1.5246        2.4121   1.7016      +4.25     -10.40     +41.76       2176        272       475       109
767022      1.8327        1.6396        2.7133   1.8090      +1.31      -9.37     +49.99       1494        280       707        99
782149      1.6785        1.6768        2.4827   1.9240     -12.76     -12.84     +29.04        801        312       561       138
788406      1.7902        1.7291        3.4722   1.7776      +0.70      -2.73     +95.33       3031        443       650       108
790322      1.8275        1.5471        3.3155   1.7633      +3.64     -12.26     +88.03       2368        489       760       100
----------------------------------------------------------------------------------------------------------------------------------
mean |err|                                                     +3.91      +9.00     +61.02
```

## Summary statistics

| Method | Mean |%err| |
|--------|------------|
| Baseline (center-¼ FOV) | **3.91%** |
| Slab-restricted median  | **9.00%** |
| Slab-restricted total   | **61.02%** |

**Winner**: baseline (center-¼ FOV)

## 782149 detail (sparse top-slab HCR, surface-MIP NCC=0.115)

- n_strict (all strict-GFP+ before spatial/depth filter): 3450  
- n_hcr_slab (full FOV, 0–100 µm): **312**  
- n_hcr_current (center-¼ FOV + depth baseline): 801  
- n_cz_slab (0–50 µm): 138  
- n_cz_current: 561  
- sxy_current = 1.6785 (err -12.76%)  
- sxy_slab_median = 1.6768 (err -12.84%)  
- sxy_slab_total  = 2.4827 (err +29.04%)  
- GT = 1.9240

## Raw results

```json
[
  {
    "sid": "755252",
    "sxy_current": 1.6263923252585757,
    "sxy_slab_median": 1.5350484393795916,
    "sxy_slab_total": 2.656097613826947,
    "sxy_gt": 1.6397222783849568,
    "err_current_pct": -0.8129396850977962,
    "err_slab_median_pct": -6.383632178765273,
    "err_slab_total_pct": 61.98460244396193,
    "n_hcr_current": 1317,
    "n_hcr_slab": 252,
    "n_cz_current": 623,
    "n_cz_slab": 85,
    "n_strict": 5402
  },
  {
    "sid": "767018",
    "sxy_current": 1.7739575979260405,
    "sxy_slab_median": 1.5245843598033457,
    "sxy_slab_total": 2.412059391307418,
    "sxy_gt": 1.7015579265976006,
    "err_current_pct": 4.254904884326136,
    "err_slab_median_pct": -10.400678344705403,
    "err_slab_total_pct": 41.75593752077082,
    "n_hcr_current": 2176,
    "n_hcr_slab": 272,
    "n_cz_current": 475,
    "n_cz_slab": 109,
    "n_strict": 8114
  },
  {
    "sid": "767022",
    "sxy_current": 1.8326699822615868,
    "sxy_slab_median": 1.6395639492702825,
    "sxy_slab_total": 2.71330621085745,
    "sxy_gt": 1.809010481595337,
    "err_current_pct": 1.30786973911754,
    "err_slab_median_pct": -9.366807658053053,
    "err_slab_total_pct": 49.98841844546026,
    "n_hcr_current": 1494,
    "n_hcr_slab": 280,
    "n_cz_current": 707,
    "n_cz_slab": 99,
    "n_strict": 6341
  },
  {
    "sid": "782149",
    "sxy_current": 1.6785411036697673,
    "sxy_slab_median": 1.676845381542174,
    "sxy_slab_total": 2.482729992857233,
    "sxy_gt": 1.9239616962835315,
    "err_current_pct": -12.756002008139614,
    "err_slab_median_pct": -12.844139008521111,
    "err_slab_total_pct": 29.042589447235876,
    "n_hcr_current": 801,
    "n_hcr_slab": 312,
    "n_cz_current": 561,
    "n_cz_slab": 138,
    "n_strict": 3450
  },
  {
    "sid": "788406",
    "sxy_current": 1.7901596893259457,
    "sxy_slab_median": 1.7291082932871236,
    "sxy_slab_total": 3.4722276517484105,
    "sxy_gt": 1.7776323333286204,
    "err_current_pct": 0.7047214298734008,
    "err_slab_median_pct": -2.7297005759698028,
    "err_slab_total_pct": 95.3287857476499,
    "n_hcr_current": 3031,
    "n_hcr_slab": 443,
    "n_cz_current": 650,
    "n_cz_slab": 108,
    "n_strict": 10729
  },
  {
    "sid": "790322",
    "sxy_current": 1.8274655539774165,
    "sxy_slab_median": 1.5470540067625107,
    "sxy_slab_total": 3.3155282960041266,
    "sxy_gt": 1.7633249436750058,
    "err_current_pct": 3.6374810288076036,
    "err_slab_median_pct": -12.264950807181203,
    "err_slab_total_pct": 88.02707396029462,
    "n_hcr_current": 2368,
    "n_hcr_slab": 489,
    "n_cz_current": 760,
    "n_cz_slab": 100,
    "n_strict": 9675
  }
]
```