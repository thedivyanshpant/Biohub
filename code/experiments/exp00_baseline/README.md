# Biohub Experiment 00: Baseline Reproduction & Local Evaluation

## Overview
This directory contains the implementation, configuration, execution scripts, and benchmark results for **Experiment 00 (Baseline)** of the Kaggle Biohub Cell Tracking Challenge.

The goal of Experiment 00 is to faithfully reproduce the heuristic baseline pipeline currently common among top public notebooks and benchmark it locally against ground truth using the official competition evaluation metric.

---

## Pipeline Architecture
For every 3D timepoint frame $t \in [0, 99]$ of each 4D Zarr volume:
1. **Downsampling**: Frame downsampled spatially by a factor of 4: `[::4, ::4, ::4]` (voxel grid $64 \times 256 \times 256 \to 16 \times 64 \times 64$).
2. **Filtering**: 3D uniform filter `scipy.ndimage.uniform_filter(size=3)`.
3. **Thresholding**: 90th percentile intensity thresholding `np.percentile(filtered, 90)`.
4. **Segmentation & Centroids**: Connected component labeling (`scipy.ndimage.label`) and center of mass extraction (`scipy.ndimage.center_of_mass`).
5. **Coordinate Restoration**: Centroid coordinates rescaled by factor of 4 back to original voxel coordinates $(z, y, x)$.
6. **Tracking & Bipartite Association**:
   - Voxel coordinates converted to physical space ($\mu\text{m}$) using spacing `SCALE = (1.625, 0.40625, 0.40625)`.
   - Pairwise Euclidean distance matrix computed between frame $t-1$ and $t$.
   - Hungarian assignment via `scipy.optimize.linear_sum_assignment`.
   - Edges accepted if Euclidean distance $\le 15.0\ \mu\text{m}$.

---

## Directory Structure
```
exp00_baseline/
├── config.yaml            # Experiment parameters and data paths
├── run.py                 # Pipeline execution, tracking, and evaluation script
├── submission_exp00.csv   # Generated Kaggle-format submission (nodes + edges)
├── RESULTS.md             # Markdown results table and metric breakdown
└── README.md              # Documentation and reproduction guide
```

---

## How to Reproduce

### Environment Setup
Run within the `ML_DFT_V2` Conda environment, including the cluster's `sweets` packages for Zarr v3 support:
```bash
export PYTHONPATH="/home/apps/MLDL/DL-CondaPy3.10/envs/sweets/lib/python3.11/site-packages:$PYTHONPATH"
```

### Execution
Run the pipeline from the project root or experiment directory:
```bash
python /scratch/deepanthpcu/Biohub/code/experiments/exp00_baseline/run.py
```

---

## Summary of Results

| Metric | Score |
| :--- | :--- |
| **Total Test Frames Processed** | 400 frames (4 volumes) |
| **Total Execution Time** | 8.50 s (~2.1 s / volume) |
| **Total Predicted Nodes** | 7,087 |
| **Total Predicted Edges** | 4,657 |
| **Micro-averaged Edge Jaccard** | **0.04132** |
| **Weighted Adjusted Edge Jaccard** | **0.04519** |
| **Micro-averaged Division Jaccard** | **0.00000** |
| **FINAL COMBINED METRIC SCORE** | **0.04519** |
