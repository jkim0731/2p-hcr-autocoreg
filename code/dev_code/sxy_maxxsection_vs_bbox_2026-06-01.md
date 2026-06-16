# sxy: max cross-section vs tight bbox (2026-06-01)

GT = landmark-Procrustes (validation only).

| sid | sxy_bbox | %err | sxy_maxx | %err | GT |
|---|---|---|---|---|---|
| 755252 | 1.5935 | -2.8% | 1.6245 | -0.9% | 1.6397 |
| 767018 | 1.8102 | +6.4% | 1.7849 | +4.9% | 1.7016 |
| 767022 | 1.7662 | -2.4% | 1.8338 | +1.4% | 1.8090 |
| 782149 | 1.6399 | -14.8% | 1.6847 | -12.4% | 1.9240 |
| 788406 | 1.7539 | -1.3% | 1.7978 | +1.1% | 1.7776 |
| 790322 | 1.7668 | +0.2% | 1.8258 | +3.5% | 1.7633 |

**mean |err|:** bbox 4.6% · maxx 4.1%  |  **std(err):** bbox 6.3 · maxx 5.7

## per-side medians
```
755252 bbox {'sxy_median': 1.6, 'err_pct_median': -2.8, 'cz_area_median': 230.0, 'hcr_area_median': 583.9}
        maxx {'sxy_median': 1.6, 'err_pct_median': -0.9, 'cz_area_median': 154.5, 'hcr_area_median': 407.8}
767018 bbox {'sxy_median': 1.8, 'err_pct_median': 6.4, 'cz_area_median': 200.8, 'hcr_area_median': 657.9}
        maxx {'sxy_median': 1.8, 'err_pct_median': 4.9, 'cz_area_median': 146.0, 'hcr_area_median': 465.2}
767022 bbox {'sxy_median': 1.8, 'err_pct_median': -2.4, 'cz_area_median': 219.0, 'hcr_area_median': 683.2}
        maxx {'sxy_median': 1.8, 'err_pct_median': 1.4, 'cz_area_median': 146.0, 'hcr_area_median': 491.0}
782149 bbox {'sxy_median': 1.6, 'err_pct_median': -14.8, 'cz_area_median': 196.5, 'hcr_area_median': 528.5}
        maxx {'sxy_median': 1.7, 'err_pct_median': -12.4, 'cz_area_median': 129.0, 'hcr_area_median': 366.1}
788406 bbox {'sxy_median': 1.8, 'err_pct_median': -1.3, 'cz_area_median': 206.9, 'hcr_area_median': 636.3}
        maxx {'sxy_median': 1.8, 'err_pct_median': 1.1, 'cz_area_median': 143.0, 'hcr_area_median': 462.1}
790322 bbox {'sxy_median': 1.8, 'err_pct_median': 0.2, 'cz_area_median': 208.1, 'hcr_area_median': 649.5}
        maxx {'sxy_median': 1.8, 'err_pct_median': 3.5, 'cz_area_median': 142.7, 'hcr_area_median': 475.6}
```
