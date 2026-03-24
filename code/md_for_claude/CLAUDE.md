# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Code Ocean capsule** for automated co-registration of ophys (optical physiology) and mFISH/HCR (multiplexed fluorescence in situ hybridization) imaging data at the Allen Institute for Neural Dynamics. The goal is to map cell identities across two imaging modalities by matching landmarks (cell centroids).

## Environment

All dependencies are managed via Docker (`environment/Dockerfile`). The runtime environment uses:
- **Python 3.11** via `/opt/conda/bin/python`
- Key packages: `numpy==1.26.4`, `scipy`, `zarr>=3.0.0`, `opencv-python-headless`, `scikit-learn`, `pandas`, `suite2p`, `pydantic<2.11`
- External Allen Institute repos installed in `postInstall`: `comb` (branch: `for_gcamp_validation`) and `lamf-analysis`

To run tests:
```bash
pytest code/test.py
```

To run notebooks interactively, use the VS Code + Jupyter environment with kernel at `/opt/conda/bin/python`.

## Repository Structure

- **`code/`** — All source code and Jupyter notebooks
  - `landmark_filtering.py` — Grid-based 3D landmark spatial sampling and filtering
  - `manual_coreg_utils.py` — KNN matching and one-to-one assignment algorithms
  - `test.py` — Minimal test stub
  - `step_*.ipynb` — Ordered workflow notebooks (see Pipeline below)
  - `*_iter_generate_landmarks.ipynb` — Iteration-specific landmark generation
  - `*.csv` — Per-subject landmark/centroid data files
- **`data/`** — Input datasets (gitignored); contains HCR processed data and ophys z-stack segmentation for subjects 754803, 755252, 767018, 767022, 782149, 788406, 790322
- **`environment/`** — `Dockerfile` and `postInstall` script
- **`/results/`**, **`/scratch/`** — Symlinked external Code Ocean volumes for outputs and temp files

## Processing Pipeline

The workflow runs in order:

1. **`step_1_process_files.ipynb`** — Load HCR and cortical z-stack data; swap TIFF dimensions; save segmentation outlines and centroids; attach Code Ocean data assets
2. **`step_2_automatic_mapping_for_qc.ipynb`** — First-pass automatic landmark generation for QC review
3. **`step_3_more_iterations.ipynb`** — Iterative refinement using `1st_iter_` through `4th_iter_generate_landmarks.ipynb`
4. **`step_4_generate_coreg_table.ipynb`** — Produce final co-registration mapping tables

## Core Algorithms

### `landmark_filtering.py`
- `grid_sample_landmarks()` — Spatially samples 3D landmarks using a grid approach, with separate keep-proportions for interior vs. edge landmarks and a per-cell minimum to ensure spatial coverage
- `visualize_landmark_distribution()` — 4-panel visualization (XY, YZ, XZ projections + Z histogram)

### `manual_coreg_utils.py`
- `choose_max_count_nearest_neighbor()` — KNN matching using spot counts/density as features with duplicate resolution
- `one_to_one_matching()` — Greedy iterative one-to-one assignment
- Landmark identifiers follow the format `czXXXX-hcrYYYY`

## Data Conventions

- Subjects are identified by numeric IDs (e.g., 754803, 755252)
- 3D coordinates use columns named `x`, `y`, `z` or `hcr_x`, `hcr_y`, `hcr_z`
- Zarr format is used for volumetric image data
- BigWarp (`bigwarp-project.json`) is used for manual registration in Fiji/ImageJ
- Resolution variants: 400×400 or 700×700 micrometers field of view

## Code Ocean-Specific Notes

- Resource class: `large` (configured in `.codeocean/resources.json`)
- Data assets are attached and mounted programmatically in notebooks
- Never commit the `data/` directory (gitignored)
- Results go to `/results/`, temporary files to `/scratch/`

# Co-registration Pipeline: Planning Summary

## Goal

Develop a **modular, semi-automated 3D co-registration pipeline** that aligns in vivo two-photon (2P) calcium imaging volumes with ex vivo lightsheet fluorescence volumes, ultimately producing **matched neuron identity lists** (which neuron in 2P corresponds to which neuron in lightsheet). Final matches are verified by **morphological comparison of matched cell pairs and their local surroundings** across modalities.

---

## Data Specifications

### Modality 1 ? Two-Photon (2P) In Vivo
| Property | Value |
|---|---|
| Fluorophore | GCaMP (inhibitory neuron-specific; pan-inhibitory in next batch) |
| Volume size | ~400 � 400 � 400 �m |
| XY resolution | 0.78 �m/pixel |
| Z resolution | 1 �m/pixel |
| Format | TBD |
| Noise characteristics | Shot noise, sparse bright blobs on dark background; quality degrades with depth |
| Z coverage | ~40 �m above pia surface included at top of stack |
| Cell count | ~600 GCaMP+ cells per volume |

### Modality 2 ? Lightsheet Ex Vivo
| Property | Value |
|---|---|
| Channels | GFP (488nm, inhibitory neuron-specific, spot-count based) + RN28S (structural/nuclear, probes all cells) |
| Volume size | ~1200?1600 � 1200?1600 �m in XY, ~350 �m in Z |
| XY resolution | 0.25 �m/pixel native; use 1 �m/pixel via zarr pyramid |
| Z resolution | 1 �m/pixel |
| Format | Zarr (multiscale pyramid) |
| Z coverage | Includes margin above and below the 2P Z extent |
| GFP match rate | ~75% of GCaMP+ cells have a matched GFP+ cell |
| Note | One of the 6 pairs has a thinner lightsheet volume |

### Segmentation ROIs
- **2P**: 3D ROIs of GCaMP+ cells
- **Lightsheet**: 3D ROIs of RN28S+ cells (all cells, not inhibitory-specific)
- GFP positiveness is determined by **spot counts in the 488 channel**, not by ROI segmentation
- Known errors: **false negatives** and **merged cells**
- Depth-dependent reliability: 2P ROI quality degrades in deeper regions
- Treat as soft/noisy landmarks only ? **not ground truth**

### Spatial Relationship
- 2P volume is spatially **inside** the lightsheet volume (roughly centered, with lightsheet margin top and bottom)
- Rotation between modalities: **~180 degrees**
- Deformation model: **non-rigid / local** (spatially variable, position-dependent distortion from ex vivo tissue processing; not a simple global compression)

---

## Training Data
- **6 co-registered example pairs** (one has thinner lightsheet volume)
- **Matched centroid coordinates available in BigWarp landmark format** for all 6 pairs
