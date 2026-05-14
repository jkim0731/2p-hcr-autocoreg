# Benchmarking datasets
6 subjects: 755252, 767018, 767022, 782149, 788406, 790322

## Available data
- f'{subject_id}_*ctl-czstack-hcr-coreg_*' directories contain
  1) 2p zstack tif file (*_zstack.tif or *_reg-dim-swapped.ome.tif or similar name)
  2) 2p zstack ROI segmentaion tif file (*_seg-matsk-outline.tif or similar) and centroids csv (czstack_cell_centroids.csv)
  3) HCR cell centroids.csv
  4) landmarks_*.csv files contain landmarks, starting with 'Pt-{n}' format rows for manual landmarks, then cz*-hcr* for candidate landmarks.
  4) coreg_table.csv contains final mapping between 2p czstack and hcr volume.
- f'HCR_{subject_id}_*_processed_*' directories contain matched HCR data
  - Look step_1_process_files.ipynb for contents and locations

## Notes
1. 782149 has thinner HCR section
2. 755252 and 767022 do not have R1 spot data (R1 GFP probe failed). Spot
   data IS available in R2: `HCR_{sid}_pairwise-unmixing_*/pairwise_unmixing/{sid}_R2/mixed_cell_by_gene.csv`
   filtered to `gene == "GFP"` (columns `spot_count`, `volume`) — equivalent
   to `spot_488_counts.csv` for the other subjects (bit-identical on
   782149/788406/790322 R1 mixed vs direct). Intensity-only fallback
   (`/root/capsule/data/cell_data_mean_*_R1.csv` channel 488) remains
   available.

## Priorities
- Test with 788406 and 790322 first.
- Then 767018 and 782149.
