# Biohub Baseline: Kaggle Notebook Local Recreation

## 1. Overview
This module is a faithful local recreation of the top public competition notebook (`cell-tracking-getting-started-w-nearest-neighbor.ipynb`) for the **Biohub Cell Tracking Challenge**.

It serves as the reference implementation and benchmark for all subsequent developments, establishing an exact local reproduction of the baseline score on the cluster without neural networks or GPU acceleration.

---

## 2. Directory Structure
```
code/baseline/
├── config.yaml             # All configurable parameters and data paths
├── notebook_baseline.py    # Faithful modular implementation of the notebook pipeline
├── evaluate_baseline.py    # Ground truth loader and official metric evaluation interface
├── run_baseline.py         # Standalone CLI runner with --dataset support
├── submission.csv          # Generated submission file in Kaggle competition format
├── BASELINE_RESULTS.md     # Detailed benchmark results, breakdown, and resource profile
└── README.md               # Documentation and execution instructions
```

---

## 3. Algorithm Specification

### Data Ingestion & Geometry
* **Zarr v3 Access**: Reads each 3D timepoint frame ($t \in [0, 99]$) of shape `(64, 256, 256)` and dtype `uint16` from OME-NGFF Zarr v3 groups.
* **Coordinate Spacing**: Physical spatial scale is $\text{SCALE} = [1.625, 0.40625, 0.40625]\ \mu\text{m/voxel}$ for $[Z, Y, X]$.
  * The axial ($Z$) axis is sampled at $4.0\times$ lower physical resolution than lateral ($XY$).

### Cell Detection (Per-Frame)
For every frame $t$:
1. **Downsampling**: Volume downsampled by a factor of 4: `vol[::4, ::4, ::4]` ($64\times$ reduction to $16 \times 64 \times 64$).
2. **Filtering**: 3D uniform box filter `scipy.ndimage.uniform_filter(size=3)`.
3. **Thresholding**: Global 90th percentile intensity threshold: `thresh = np.percentile(smoothed, 90.0)`.
4. **Connected Components**: 3D labeling via `scipy.ndimage.label(binary)`.
5. **Centroid Localization**: Center of mass computed per component, rescaled back by 4, and rounded to integer native voxel grid:
   $$z = \text{round}(c_z \times 4),\quad y = \text{round}(c_y \times 4),\quad x = \text{round}(c_x \times 4)$$

### Temporal Tracking (Between Consecutive Frames)
Between frame $t-1$ and $t$:
1. **Physical Distance**: Centroids scaled to micrometers: $P_{\text{phys}} = P_{\text{voxel}} \times \text{SCALE}$.
2. **Cost Matrix**: Euclidean distance matrix $D_{ij} = \|P_i(t-1) - P_j(t)\|_2$.
3. **Hungarian Bipartite Matching**: `scipy.optimize.linear_sum_assignment(D)`.
4. **Distance Gating**: Pairs accepted if and only if $D_{ij} \le 15.0\ \mu\text{m}$.
5. Strictly 1-to-1 matching ($d_{\text{out}} \le 1$); no cell division handling.

---

## 4. Submission Format
Generated file `code/baseline/submission.csv` strictly adheres to Kaggle competition requirements:
* Header: `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`
* **Node rows**: `row_type = node`, `node_id = <int>`, coordinates `t, z, y, x`, `source_id = -1`, `target_id = -1`.
* **Edge rows**: `row_type = edge`, `node_id = -1`, coordinates `-1, -1, -1, -1`, `source_id = <src>`, `target_id = <tgt>`.

---

## 5. Execution Instructions

### Environment
Run inside the `ML_DFT_V2` Conda environment on the compute cluster:
```bash
export PYTHONPATH="/home/apps/MLDL/DL-CondaPy3.10/envs/sweets/lib/python3.11/site-packages:$PYTHONPATH"
```

### Full Validation Suite (All 4 Public Volumes)
```bash
python code/baseline/run_baseline.py
```

### Single Volume Execution
```bash
python code/baseline/run_baseline.py --dataset 44b6_0113de3b
```

---

## 6. Baseline Performance Summary

| Metric | Measured Baseline Score | Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **Combined Score** | **0.04519** | **0.04519** | **EXACT REPRODUCTION** |
| **Micro-Averaged Edge Jaccard** | **0.04132** | 0.04132 | Exact Match |
| **Sample-Weighted Adjusted Edge Jaccard** | **0.04519** | 0.04519 | Exact Match |
| **Division Jaccard** | **0.00000** | 0.00000 | Exact Match |
| **Edge TP / FP / FN** | **89 / 27 / 2,038** | 89 / 27 / 2,038 | Exact Match |
| **Total Runtime (4 Volumes / 400 Frames)** | **~8.3 s** | < 15 s | Blazing Fast CPU |
| **Throughput** | **~48 frames/second** | - | Pure CPU |
| **Peak RAM** | **~1.05 GB** | - | Minimal Footprint |
| **GPU Utilization** | **0%** | - | Pure CPU Reference |
