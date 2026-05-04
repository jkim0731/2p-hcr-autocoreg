---
name: S12 FPFH descriptor feasibility (P15 deferral confirmed)
description: 2026-05-02 — under GT-TPS + strict-GFP+ + ROI-q≥0.5, mean AUC@R50µm: PPF 0.74, kNN-dist 0.71, FPFH 0.66, centroid-LP 0.63; recall@5 dominated by centroid-LP (mean 0.795); FPFH+RANSAC not earned, pose-only matcher family closed.
type: project
originSessionId: a6b80cf6-8ea2-4333-9758-4422dc184b49
---
S12 ran the FPFH descriptor feasibility test the user proposed — before
investing in P15 (FPFH+RANSAC global registration), check whether FPFH or
similar local 3D descriptors discriminate GT-paired HCR cells from
close-but-wrong neighbours under a clean alignment.

**Setup.** GT-TPS warp of CZ centroids into HCR µm zyx (so descriptor
quality is isolated from pose error); per-cell pia-surface normals
(analytic gradient of cached `surfaces_iter08` polynomial); HCR pool =
strict-GFP+ (07b GMM-intersection cutoff) ∩ `roi_quality_v5d ≥ 0.5` ∩
bbox-of-warped-CZ + 30 µm.

**Descriptors.** FPFH (Open3D, 33d, r=120 µm), PPF marginal hist (Drost
4-feat × 5 bins = 20d), kNN-distance hist (k=12, 16d on [0,200] µm),
centroid distance under LP warm-start, centroid distance under GT-TPS
(degenerate sanity floor).

**Results** (mean AUC vs hard near-miss within R=50µm, recall@5 of true
partner among full filtered HCR pool):

| descriptor | mean AUC R=50µm | mean recall@5 |
|------------|---------------:|---------------:|
| PPF        | 0.738          | 0.199          |
| kNN-dist   | 0.711          | 0.115          |
| FPFH       | 0.662          | 0.189          |
| centroid_LP | 0.631          | **0.795**      |
| centroid_GT_TPS | 1.000     | 1.000          |

**Verdict.** FPFH+RANSAC promotion is **not earned** — no feature
descriptor clears AUC ≥ 0.85, and centroid-LP dominates recall@K by
0.5–0.9 on every subject.  The LP-warmed CZ centroid already places the
true HCR partner in the **top-5 LP-distance shortlist 80 % of the time**
(recall@20 ~0.95).  Pose-only matcher family (Hungarian, TEASER,
FPFH+RANSAC) closed.

**How to apply.** When user asks "should we try FPFH/RANSAC/local
geometric descriptors?" — refer them here.  The actionable hand-off is
into §4.2/4.3: feed the top-K LP-distance shortlist into a per-pair
classifier; the classifier supplies tie-breaking signal that the
descriptors cannot.  For subjects with long-tail LP residual
(755252/782149), §4.5 TPS expansion seeded by classifier-confirmed
anchors is the right next step, not a different geometric matcher.

**Caveat.** Pia-projected normals carry only mean cortical curvature,
not per-cell shape — chosen deliberately at user request over noisy
PCA-of-neighbours normals.  PCA-FPFH was not tested.  Strict-GFP+ on
755252 keeps only 70 % of GT cells (coreg_coverage 0.701); diagnostic
on that subject is correspondingly noisier.

Files: `code/sessions/v3_S12_fpfh_feasibility/{run_fpfh_feasibility.py,
fpfh_feasibility_results.csv, subject_summary.csv, run_log.txt, log.md}`.
