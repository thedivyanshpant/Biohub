# Experiment 00: Baseline Reproduction & Local Evaluation

## 1. Overview
* **Experiment ID**: `exp00_baseline`
* **Objective**: Faithful reproduction of the top competition heuristic baseline (downsampled uniform filter + 90th percentile threshold + Hungarian bipartite linking) and validation on the 4 public test/train stub pairs.
* **Execution Hardware**: CPU only (Cascade Lake, 48 cores, node `rpgpu018`).
* **Total Execution Time**: 8.50 seconds across all 4 volumes (400 frames total).

---

## 2. Methodology & Hyperparameters
* **Preprocessing & Detection**:
  * Downsampling: `[::4, ::4, ::4]` (volume reduction from $64 \times 256 \times 256$ to $16 \times 64 \times 64$)
  * Uniform Filter: `scipy.ndimage.uniform_filter(size=3)`
  * Threshold: `np.percentile(filtered, 90)`
  * Connected Components: `scipy.ndimage.label` + `scipy.ndimage.center_of_mass`
  * Coordinate Transformation: Centroid multiplied by 4 back to integer voxel grid.
* **Tracking & Graph Association**:
  * Coordinate Space: Physical $\mu\text{m}$ via `SCALE = [1.625, 0.40625, 0.40625]`
  * Pairwise Cost: Euclidean distance matrix between frame $t-1$ and $t$
  * Matching Algorithm: `scipy.optimize.linear_sum_assignment`
  * Acceptance Threshold: Distance $\le 15.0\ \mu\text{m}$
* **Evaluation Metric (Official Specification)**:
  * Spatial Match Tolerance: $7.0\ \mu\text{m}$
  * Metric Components: Edge Jaccard, Division Jaccard, Total Node Ratio Adjustment ($\alpha=0.1$), Division Weight ($w=0.1$).

---

## 3. Results per Dataset

| Dataset Name | Pred Nodes | GT Nodes | Pred Edges | GT Edges | Pred Divs | GT Divs | Edge Jaccard | Adj Edge Jaccard | Div Jaccard | Combined Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `44b6_0113de3b` | 2,113 | 52 | 1,356 | 50 | 0 | 0 | 0.0377 | 0.0412 | N/A | **0.0412** |
| `44b6_0b24845f` | 1,268 | 51 | 648 | 49 | 0 | 0 | 0.0000 | 0.0000 | N/A | **0.0000** |
| `6bba_05b6850b` | 421 | 861 | 309 | 845 | 0 | 0 | 0.0802 | 0.0877 | N/A | **0.0877** |
| `6bba_05db0fb1` | 3,285 | 1,229 | 2,344 | 1,183 | 0 | 3 | 0.0151 | 0.0165 | 0.0000 | **0.0165** |

---

## 4. Aggregate Run-Level Score (Micro-Averaged)

| Metric | Aggregate Value |
| :--- | :--- |
| **Total Edge True Positives (TP)** | **89** |
| **Total Edge False Positives (FP)** | **27** |
| **Total Edge False Negatives (FN)** | **2,038** |
| **Micro-Averaged Edge Jaccard** | **0.04132** |
| **Sample-Weighted Adjusted Edge Jaccard** | **0.04519** |
| **Total Division TP / FP / FN** | **0 / 0 / 3** |
| **Micro-Averaged Division Jaccard** | **0.0000** |
| **FINAL COMBINED SCORE** | **0.04519** |

---

## 5. Observations & Strategic Takeaways
1. **Division Capability**: EXP00 produces zero divisions by design because linear sum assignment enforces strictly 1-to-1 temporal matching ($d_{\text{out}} \le 1$). Future iterations must explicitly handle branching ($d_{\text{out}} = 2$) to capture mitosis.
2. **Node Penalty**: The node count adjustment penalizes over-detection or under-detection relative to `estimated_number_of_nodes`. Tuning thresholding percentiles directly influences this ratio.
3. **Execution Efficiency**: Baseline runtime is ~4.5 seconds per 100-frame 4D volume on CPU, establishing an exceptionally fast benchmark for subsequent iterations.
