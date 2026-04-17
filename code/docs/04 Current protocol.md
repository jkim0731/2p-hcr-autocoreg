# Current protocol
- Manual coregistration and cell by cell QC
- Using BigWarp

## Protocol files
/root/capsule/code/step_*_*.ipynb
1. Process input data files
2. Manual coregistration using BigWarp
  1) Find 4-6 initial constellation by eyes in 2p data near the surface (all ROIs near each other, unique pattern)
  2) Find GFP+ cell patterns in HCR data matched to the the 2p initial constellation
  3) Repeat 1->2 until confident match found
  4) Set the constellation and match as landmarks and apply thin plate spline transformation.
  5) Find matched patterns near the current landmarks, gradually covering more volume. Do not exhaust all cells. Total about 50 to 100 landmarks covering almost the entire volume. Activate each landmark whenever found, applying thin plate spline transformation.
3. Automatic mapping for qc: using current landmarks, find matched ROIs between 2p and GFP+ HCR ROIs based on distance.
4. Manually go through each candidate match, and add them to landmarks if matched by eyes. Look for context and cell morphology, and if activating the pair increases correlation within the context volume.
5. Repeat 3 and 4 until no more new matched pairs appear.
6. generate coregistration table.