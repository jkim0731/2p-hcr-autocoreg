# 260429
- HCR ROI segmentation errors.
- Initial landmark search not great or difficult to verify.
- Cell-cell matching classifier would be great to automate the procedure (how can I deal with orientation variation?)
- Next steps:
    - Build HCR ROI classifier (compare with Matt's classifier). How long does it take to get HCR ROI segmentation? Within and nearby the overlapping volume + margin only?
    - Redo automatic registration algorithm with the initial registration constraint.
        - Centroid-based: Both registration and cell matching. Requires good segmentation and GFP+ filtering. cell matching can be improved after initial registration. How to assess registration improvement?
        - Image-based: registration only. Apply cell-cell matching later (maybe similar to manual workflow after the fine registration - but requires cell-cell matching classifier to be fully automatic)

# 260514
- HCR ROI classifier is built and quite reliable. Use good + bad_ok cells when filtering is necessary.
- GFP+ cells can be defined from BIC GMM from 488 spot density (raw "unmixed" spots - R2 for 755252 and 767022)
- CP SAM test is running but not promising. Current segmentation may be the best option.
- 3 potential options left: 
    - 0. All after initial surface registration 
    - 1. Matching using geometric features (shape contexts - Belongie et al., 2002; soma-print - Wang et al., 2026 https://www.biorxiv.org/content/10.64898/2026.04.28.719500v1.full.pdf)
    - 2. Piecewise volumetric nonrigid registration - pyramid. 
    - 3. Mimicking manual workflow (requires evaluating matches - maybe the same features in #1, or using 3D image networks)
    