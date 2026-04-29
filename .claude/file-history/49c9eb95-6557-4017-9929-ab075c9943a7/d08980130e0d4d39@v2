---
name: sxy must come from surface registration, not ROI bbox ratio
description: For locked-prior / Stage A and any final sxy decision, use the surface_registration_v2 PWR-affine sxy. ROI-area xy-bbox ratio is only an initial guess for rigid bootstrap.
type: feedback
originSessionId: 40fd9680-417d-48de-ac1a-71e8a68a7966
---
For all 6 subjects, the final per-subject sxy must come from
`surface_registration_v2`'s PWR fit (image-NCC-driven, e.g. the affine
stage `base_sxy * exp(d_log_scale)`). The per-cell xy bbox area ratio
(`roi_area_sxy.estimate_sxy_roi_area`) is only acceptable as an
**initial guess** to bootstrap rigid registration — never as the final
locked-prior sxy.

**Why:** image-NCC against the actual modality intensities is the
ground-truth physical sxy. ROI-area ratios assume cells are
detected uniformly between modalities, which has known biases (e.g.
782149 −15 % from surface span); using it as the final sxy bakes
those biases into every downstream stage.

**How to apply:**
- In any module that pins sxy for warm-start or hard-locks it
  (Stage A, locked-prior, candidate constructors), pull from
  `surface_registration_v2` directly.
- `roi_area_sxy.estimate_sxy_roi_area` is acceptable only as a starting
  scale for an iterative image-based fit (i.e. the PWR rigid bootstrap),
  never as a load-bearing output.
- Do NOT special-case 788406/790322/767018 to use roi_area sxy
  "because it agrees" — uniformity matters more than tiny per-subject
  agreement.
