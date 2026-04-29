# 260429
- HCR ROI segmentation errors.
- Initial landmark search not great or difficult to verify.
- Cell-cell matching classifier would be great to automate the procedure (how can I deal with orientation variation?)
- Next steps:
    - Build HCR ROI classifier (compare with Matt's classifier). How long does it take to get HCR ROI segmentation? Within and nearby the overlapping volume + margin only?
    - Redo automatic registration algorithm with the initial registration constraint.
        - Centroid-based: Both registration and cell matching. Requires good segmentation and GFP+ filtering. cell matching can be improved after initial registration. How to assess registration improvement?
        - Image-based: registration only. Apply cell-cell matching later (maybe similar to manual workflow after the fine registration - but requires cell-cell matching classifier to be fully automatic)