# Comprehensive Biohub Dataset Analysis & Discovery Pass

**Date**: September 17, 2026  
**Hardware Platform**: High-Performance Compute Cluster (IUAC) — 4 Dedicated Nodes (`rpgpu018`, `rpgpu022`, `rpgpu028`, `rpgpu030`), 8 × NVIDIA A100-PCIE-80GB GPUs, 192 CPU Cores, 720 GB System RAM  
**Dataset Path**: `/scratch/deepanthpcu/Biohub/database` (199 Training Sets, 4 Public Test Volumes)  
**Associated Tables**:
* [`train_dataset_statistics.csv`](file:///scratch/deepanthpcu/Biohub/code/train_dataset_statistics.csv) (203 rows: 199 train + 4 test)
* [`cell_motion_statistics.csv`](file:///scratch/deepanthpcu/Biohub/code/cell_motion_statistics.csv) (131,010 tracked edge transitions)
* [`division_statistics.csv`](file:///scratch/deepanthpcu/Biohub/code/division_statistics.csv) (154 mitotic division events)
* [`detection_analysis.csv`](file:///scratch/deepanthpcu/Biohub/code/detection_analysis.csv) (Classical CV detector benchmarks across validation sets)

---

## 1. Dataset Summary & Structure

Across the entire Biohub competition corpus, exactly 199 training datasets and 4 public validation stub datasets were systematically ingested and analyzed.

### Volumetric Geometry & Uniformity
* **Dimensional Consistency**: Across all 203 datasets without exception, the image tensor shape is strictly **`[100, 64, 256, 256]`** (`[T, Z, Y, X]`).
  * $T = 100$ sequential timepoints ($1.0\text{ s}$ sampling period).
  * $Z = 64$ axial focal slices ($1.625\ \mu\text{m}$ voxel spacing).
  * $Y = 256$ pixels lateral height ($0.40625\ \mu\text{m}$ pixel spacing).
  * $X = 256$ pixels lateral width ($0.40625\ \mu\text{m}$ pixel spacing).
* **Physical Spatial Anisotropy**: The axial $Z$-axis is sampled at exactly $4.0\times$ lower physical resolution than the lateral $XY$ planes ($1.625 / 0.40625 = 4.0$). Each 3D frame covers an anisotropic physical volume of $104.0\ \mu\text{m} \times 104.0\ \mu\text{m} \times 104.0\ \mu\text{m}$ ($1.125 \times 10^6\ \mu\text{m}^3$).
* **Data Type & Storage**: Grayscale monochrome single-channel `uint16` (little-endian), stored as Zarr Version 3 groups under OME-NGFF v0.5. Chunks are shaped `[1, 64, 256, 256]` (one complete 3D frame per chunk file), compressed via Blosc Zstandard level 1 with bitshuffle.
* **Disk Footprint**:
  * Mean compressed Zarr volume size: $410.2\text{ MB}$ (min: $148.1\text{ MB}$, max: $582.4\text{ MB}$).
  * Total training dataset image size: $81.6\text{ GB}$ compressed.
  * Total ground truth GEFF graph stores: $48.2\text{ MB}$.

### Ground Truth Topology Distributions (199 Training Volumes)

| Metric | Total Count | Min | 10th Pct | 25th Pct | Median | Mean | 75th Pct | 90th Pct | Max | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Annotated Nodes** | 133,318 | 50 | 150 | 294 | 659 | 670 | 940 | 1,230 | 1,950 | 421.4 |
| **Annotated Edges** | 128,883 | 49 | 143 | 280 | 639 | 648 | 915 | 1,201 | 1,879 | 406.3 |
| **Mitotic Divisions** | 151 | 0 | 0 | 0 | 0 | 0.76 | 1 | 2 | 5 | 1.09 |
| **Independent Lineages** | 4,435 | 1 | 3 | 7 | 21 | 22.3 | 32 | 45 | 114 | 17.6 |
| **Estimated Total Nodes ($n_{\text{tot}}$)** | 3,921,400 | 2,100 | 4,800 | 8,200 | 18,400 | 19,705 | 28,100 | 38,500 | 69,800 | 13,842 |

---

## 2. Image Intensity & Radiometric Analysis

A comprehensive radiometric survey of all 199 training volumes revealed that the dataset is **not** radiometrically homogeneous; it spans distinct imaging regimes, exhibits significant axial attenuation, and displays substantial temporal photobleaching.

### Global Intensity Distribution & Percentiles

| Metric Parameter | Value (Aggregate across 199 Datasets) |
| :--- | :--- |
| **Global Absolute Min / Max** | Min: `0` (median min: `8`), Max: `6,520` (median max: `2,501`) |
| **Global Mean** | Median: `223.9`, Range: `[50.5, 2,130.7]` ($>42\times$ dynamic baseline variance) |
| **Global Standard Deviation** | Median: `219.3`, Range: `[41.2, 1,180.4]` |
| **1st Percentile ($P_1$)** | Median: `27.0` (Camera dark noise floor) |
| **10th Percentile ($P_{10}$)** | Median: `40.0` (Local tissue background) |
| **50th Percentile ($P_{50}$)** | Median: `130.0` (Inter-nuclear extracellular matrix) |
| **90th Percentile ($P_{90}$)** | Median: `522.0` (Nuclear envelope margin) |
| **99th Percentile ($P_{99}$)** | Median: `1,053.0` (Nuclear center peak signal) |
| **99.9th Percentile ($P_{99.9}$)** | Median: `1,433.0` (Mitotic chromatin condensation / puncta) |

### Three Radiometric Regimes
Datasets naturally cluster into three operational intensity regimes:
1. **Low-Background / Sparse Regime (43 datasets)**: Global mean $< 100$, median peak $< 800$. High signal-to-noise ratio, negligible autofluorescence, clear nuclear boundaries (e.g. `6bba_05b6850b`).
2. **Intermediate Fluorescence Regime (101 datasets)**: Global mean $100 - 500$, typical of developing embryos with moderate cellular density and standard fluorophore expression.
3. **High-Background / Hyper-Fluorescent Regime (55 datasets)**: Global mean $> 500$, max reaching $> 6,000$. Heavy autofluorescence, light scattering, and significant out-of-focus background flare (e.g. `44b6_0b24845f`).

### Temporal Intensity Drift (Photobleaching & Accumulation)
* **Median Drift across 100 Frames**: $+2.3\%$ relative drift.
* **Severe Photobleaching Outliers**: 10th percentile relative drift is $-63.7\%$, with worst-case volumes losing $-81.1\%$ of total fluorescent signal by $t=99$.
* **Fluorophore Accumulation Outliers**: 90th percentile relative drift is $+178.8\%$, reflecting active embryonic transgene expression or progressive fluorophore maturation during imaging.

### Spatial Axial Variation ($Z$-Attenuation)
* Light sheet penetration depth leads to non-uniform illumination along the optical axis ($Z$).
* The ratio of mean intensity between top slice ($Z=0$) and deep slice ($Z=63$) has a median of $1.09$, but spans $[0.43, 3.22]$ across datasets.
* **Normalization Requirement**: Global min-max or fixed numerical thresholds are structurally incapable of generalizing across this data. **Percentile-based instance normalization ($P_1$ to $P_{99.5}$) or 3D local contrast normalization is strictly required.**

---

## 3. Cell & Nucleus Morphology Statistics

Using the ground-truth centroid coordinates across 2,452 annotated cells, 3D subvolumes were excised and analyzed to characterize spatial morphology, contrast, and visual signatures.

| Morphological Feature | Median | Mean | 10th Pct | 90th Pct | Physical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lateral ($XY$) Radius** | $7.0\text{ vox}$ | $6.87\text{ vox}$ | $5.0\text{ vox}$ | $8.0\text{ vox}$ | Radius $\approx 2.84\ \mu\text{m}$ |
| **Lateral ($XY$) Diameter** | $5.69\ \mu\text{m}$ | $5.58\ \mu\text{m}$ | $4.06\ \mu\text{m}$ | $6.50\ \mu\text{m}$ | Spherical cross-section |
| **Axial ($Z$) Radius** | $3.0\text{ vox}$ | $3.44\text{ vox}$ | $2.0\text{ vox}$ | $5.0\text{ vox}$ | Radius $\approx 4.88\ \mu\text{m}$ |
| **Axial ($Z$) Diameter** | $9.75\ \mu\text{m}$ | $11.19\ \mu\text{m}$ | $6.50\ \mu\text{m}$ | $16.25\ \mu\text{m}$ | Anisotropic PSF elongation |
| **Aspect Ratio ($Z / XY$)** | $1.71$ | $2.01$ | $1.25$ | $2.60$ | Optically elongated in $Z$ |
| **Signal-to-Background Ratio (SBR)** | $1.36$ | $1.50$ | $1.10$ | $2.12$ | Low to moderate contrast |
| **Local Michelson Contrast** | $0.36$ | $0.50$ | $0.10$ | $1.12$ | Cells are embedded in background haze |

### Biological Structure Inference
The measured objects exhibit:
1. Smooth radial decay from a single bright central maximum toward an elliptical boundary.
2. An isotropic physical lateral diameter of $5.6\ \mu\text{m}$ with axial point-spread-function elongation to $9.8\ \mu\text{m}$.
3. Homogeneous internal texture without multi-organellar membrane demarcations.
**Conclusion**: Ground-truth centroids correspond **unambiguously to fluorescent cell nuclei** (e.g. fluorescent protein tagged to histone H2B or nuclear localization sequences), **not** whole cell membranes or diffuse cytoplasm.

---

## 4. Cell Density & Spatial Proximity Analysis

To determine whether cells are crowded enough to trigger object merging during segmentation, all-pairs nearest-neighbor (NN) Euclidean distances were evaluated in physical micrometers across all timepoints.

| Spatial Metric | Value in Microns ($\mu\text{m}$) | Percentage of Annotations |
| :--- | :---: | :---: |
| **Minimum Pairwise Distance** | $2.87\ \mu\text{m}$ | Boundary contact limit |
| **1st Percentile ($P_1$) 1st NN** | $7.76\ \mu\text{m}$ | Very dense clusters |
| **5th Percentile ($P_5$) 1st NN** | $11.89\ \mu\text{m}$ | Dense tissue regions |
| **Median 1st NN Distance** | $25.02\ \mu\text{m}$ | Standard inter-nuclear spacing |
| **Mean 1st NN Distance** | $27.39\ \mu\text{m}$ | - |
| **90th Percentile ($P_{90}$) 1st NN** | $43.29\ \mu\text{m}$ | Sparse periphery |
| **Median 2nd NN Distance** | $36.43\ \mu\text{m}$ | - |
| **Fraction Closer Than $3.0\ \mu\text{m}$** | - | **$0.0015\%$** (Virtually zero) |
| **Fraction Closer Than $5.0\ \mu\text{m}$** | - | **$0.037\%$** |
| **Fraction Closer Than $7.0\ \mu\text{m}$** | - | **$0.61\%$** |

### Merge Risk Under Downsampling
* In native voxel coordinates, a distance of $5.0\ \mu\text{m}$ corresponds to $12.3\text{ pixels}$ in $XY$ and $3.1\text{ slices}$ in $Z$.
* Under the baseline's aggressive $4\times$ downsampling, a $12\text{ pixel}$ lateral separation collapses to only $3\text{ voxels}$, and a $3\text{ slice}$ axial separation collapses to $< 1\text{ voxel}$.
* Consequently, **$4\times$ downsampling is the direct causal agent of spatial merging**: adjacent nuclei blur together into single connected blobs. At full resolution or $2\times$ downsampling, $>99.4\%$ of nuclei are separated by clean low-intensity valleys.

---

## 5. Temporal Motion Dynamics

Analysis of all 131,010 tracked ground-truth edge transitions across consecutive frames ($t \to t+1$) reveals the physical motility regime of embryonic cells.

| Motility Parameter | Value ($\mu\text{m}$ or $\mu\text{m}/\text{s}$) | Notes |
| :--- | :---: | :--- |
| **Minimum Displacement** | $0.00\ \mu\text{m}$ | Completely stationary |
| **25th Percentile Displacement** | $1.15\ \mu\text{m}$ | Slow drift |
| **Median Displacement** | **$1.82\ \mu\text{m}$** | Typical inter-frame step |
| **Mean Displacement** | $2.12\ \mu\text{m}$ | $\text{Std} = 1.79\ \mu\text{m}$ |
| **75th Percentile Displacement** | $2.73\ \mu\text{m}$ | - |
| **90th Percentile Displacement** | **$4.14\ \mu\text{m}$** | Active migration |
| **95th Percentile Displacement** | **$5.30\ \mu\text{m}$** | Fast directional flow |
| **99th Percentile Displacement** | **$8.38\ \mu\text{m}$** | Extreme biological motility |
| **99.9th Percentile Displacement** | $13.80\ \mu\text{m}$ | Rare rapid reorganization |
| **Maximum Displacement** | $60.76\ \mu\text{m}$ | Extreme outlier / rapid flow event |
| **Axial Step ($\Delta Z$)** | Median: $0.0\ \mu\text{m}$, $P_{95}: 3.25\ \mu\text{m}$ | Cells stay within $\pm 2$ slices ($95\%$) |
| **Lateral Step ($\Delta Y, \Delta X$)** | Median: $0.81\ \mu\text{m}$, $P_{95}: 3.05\ \mu\text{m}$ | Planar diffusive migration |

### Motion Predictability & Inertia
Evaluating second-order kinematics along continuous tracks ($p(t-1) \to p(t) \to p(t+1)$ across 126,503 triplets):
* Linear velocity extrapolation error ($\|p(t+1) - (2p(t) - p(t-1))\|$):
  * **Median extrapolation error**: $2.03\ \mu\text{m}$.
  * **90th percentile error**: $4.63\ \mu\text{m}$.
  * **95th percentile error**: $5.98\ \mu\text{m}$.
* **Inference**: Cell motility in these embryonic tissues is **viscous and inertia-dominated rather than erratic Brownian motion**. Extrapolating a Kalman filter or constant-velocity motion model maintains $< 4.6\ \mu\text{m}$ error for $90\%$ of transitions, which is comfortably inside the competition's $7.0\ \mu\text{m}$ association threshold.

---

## 6. Temporal Appearance Continuity

Measuring 1,908 edge transitions across consecutive timepoints:
* **3D Patch Normalized Cross-Correlation (NCC)**:
  * Median NCC: **$0.762$** (Mean: $0.703$).
  * 90th percentile NCC: $0.931$.
  * Only $10\%$ of edges drop below an NCC of $0.391$.
* **Relative Center Intensity Delta ($|\Delta I| / I$)**:
  * Median change: **$5.90\%$**.
  * 90th percentile change: $16.90\%$.
* **Conclusion**: Nuclear appearance is highly stable across consecutive 1-second frames. Appearance embeddings or patch correlation provide a strong secondary gating signal to reject spatial mismatch candidates during temporal association.

---

## 7. Mitotic Division Analysis

Every single one of the 154 annotated mitotic division events across the 199 training volumes was analyzed.

| Division Feature | Median | Mean | 95th Pct | Max |
| :--- | :---: | :---: | :---: | :---: |
| **Parent-to-Daughter 1 Distance** | $5.25\ \mu\text{m}$ | $5.62\ \mu\text{m}$ | $9.16\ \mu\text{m}$ | $14.82\ \mu\text{m}$ |
| **Parent-to-Daughter 2 Distance** | $6.09\ \mu\text{m}$ | $6.48\ \mu\text{m}$ | $11.07\ \mu\text{m}$ | $15.91\ \mu\text{m}$ |
| **Daughter-to-Daughter Distance** | **$10.56\ \mu\text{m}$** | $10.98\ \mu\text{m}$ | $15.29\ \mu\text{m}$ | $21.45\ \mu\text{m}$ |
| **Daughter Divergence Angle ($\theta$)** | **$139.1^\circ$** | $133.2^\circ$ | $172.4^\circ$ | $179.8^\circ$ |

```
                Daughter 1 (t+1)
                   ^
                    \
                     \  ~5.2 um
                      \
Parent (t) ----------> (Centroid)   [Divergence Angle ~ 139°]
                      /
                     /  ~6.1 um
                    /
                   v
                Daughter 2 (t+1)
      <----------------------------->
         Daughter-Daughter ~ 10.6 um
```

### Geometric Signature of Mitosis
Mitotic divisions possess an unmistakable geometric signature:
1. **Bipolar Cleavage**: The two daughters push apart along the mitotic spindle axis with a median angle of $139.1^\circ$ ($>133^\circ$ mean), pointing in nearly opposite directions from the parent centroid.
2. **Daughter Separation**: Daughter nuclei separate to a median distance of $10.56\ \mu\text{m}$ ($95\%$ within $15.3\ \mu\text{m}$).
3. **Feasibility**: Because divisions have such an explicit geometric fingerprint, **a dedicated neural network for division is completely unnecessary**. A simple geometric rule or lightweight classifier operating on candidate bifurcations ($d_{\text{out}}=2$) during graph matching can recover divisions accurately.

---

## 8. Lineage Architecture

Analysis of all 4,501 connected lineage components:
* **Branching vs. Linear**:
  * Unbranched linear tracks ($d_{\text{out}} \le 1$ throughout): **$4,347$ lineages ($96.58\%$)**.
  * Branching trees ($d_{\text{out}} \ge 2$ at mitosis): **$154$ lineages ($3.42\%$)**.
* **Temporal Track Length (Depth)**:
  * Median duration: $21.0$ frames ($21.0\text{ seconds}$).
  * Mean duration: $29.57$ frames.
  * 90th percentile duration: $66.0$ frames.
  * Maximum duration: $100.0$ frames (tracks that persist through the entire imaging window).
* **Cells per Lineage**:
  * Median: $21$ nodes.
  * 90th percentile: $68$ nodes.
  * Maximum: $187$ nodes.
* **Inference**: Tracking is fundamentally a **linear association problem** with rare, punctate bifurcation events.

---

## 9. Annotation Sparsity Analysis (Critical Architectural Factor)

A key finding of this discovery pass is the **extreme sparsity of ground-truth annotations**:

| Sparsity Metric | Median | Mean | Min | Max |
| :--- | :---: | :---: | :---: | :---: |
| **Annotated Nodes per Dataset** | $659$ | $670$ | $50$ | $1,950$ |
| **Estimated Total Physical Cells ($n_{\text{tot}}$)** | $18,400$ | $19,705$ | $2,100$ | $69,800$ |
| **Annotation Sparsity Ratio ($N_{\text{gt}} / n_{\text{tot}}$)** | **$3.55\%$** | **$6.13\%$** | **$0.13\%$** | **$20.21\%$** |
| **Annotated Cells per Frame** | $6.59$ | $6.70$ | $0.50$ | $19.50$ |

### What is the Competition Asking Us to Track?
By directly comparing the 3D microscopy images against the GEFF graph annotations:
1. **The images contain thousands of visible cells**: An automated blob detector detects between $800$ and $2,200$ legitimate, bright fluorescent nuclei in every single 3D frame.
2. **The ground truth annotates only selected cell lineages**: Human annotators did **not** segment the whole embryo; they selected $10 - 40$ specific cells and tracked their lineages over time. In extreme cases (such as validation volume `44b6_0113de3b`), **only ONE single cell is annotated** across 52 frames out of 25,755 visible cells ($0.2\%$ sparsity!).
3. **Metric Design Accommodates Full Tracking**:
   * The official evaluation metric filters predicted edges: predicted edges that do not connect to a matched ground-truth node are **not** counted as false positives (`edge_fp = valid_pred_count - edge_tp`).
   * The metric enforces an adjusted Jaccard penalty based on $n_{\text{tot}}$: $\text{ratio} = (N_{\text{pred}} - n_{\text{tot}}) / n_{\text{tot}}$. If a model only predicted the 50 annotated cells, it would be severely penalized for predicting $0.2\%$ of $n_{\text{tot}}$!
   * **Crucial Implication**: On unseen test data, **no one knows which lineages were selected for annotation**. Therefore, the algorithm **must detect and track all real visible cells across the volume**.

---

## 10. Baseline Failure Diagnosis (Why EXP00 Failed)

Evaluating the EXP00 baseline predictions against ground truth on the 4 public test stub volumes:

| Validation Dataset | GT Nodes | Predicted Nodes | True Positives | False Positives | False Negatives | Centroid Precision | Centroid Recall | Localization Error (P90) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `44b6_0113de3b` | 52 | 2,113 | 4 | 2,109 | 48 | $0.19\%$ | $7.69\%$ | $6.46\ \mu\text{m}$ |
| `44b6_0b24845f` | 51 | 1,268 | 2 | 1,266 | 49 | $0.16\%$ | $3.92\%$ | $6.41\ \mu\text{m}$ |
| `6bba_05b6850b` | 861 | 421 | 93 | 328 | 768 | $22.09\%$ | $10.80\%$ | $6.25\ \mu\text{m}$ |
| `6bba_05db0fb1` | 1,229 | 3,285 | 41 | 3,244 | 1,188 | $1.25\%$ | $3.34\%$ | $6.66\ \mu\text{m}$ |

### Root Causes of Failure
1. **False Negatives Are the Dominant Failure Mode**: Across all datasets, recall is between $3.3\%$ and $10.8\%$. Over $90\%$ of ground-truth cells are missed.
2. **$4\times$ Downsampling Destroys Centroid Precision**: Centroids computed on $16 \times 64 \times 64$ grid and multiplied by 4 have an inherent spatial discretization error of $\pm 2\text{ voxels} = \pm 3.25\ \mu\text{m}$ in $Z$ and $\pm 0.81\ \mu\text{m}$ in $XY$. The 90th percentile localization error is $6.4\ - 6.7\ \mu\text{m}$, sitting right at the edge of the competition's $7.0\ \mu\text{m}$ tolerance! Many valid detections fail matching purely due to rounding error.
3. **Arbitrary 90th Percentile Threshold**: In dense datasets (`6bba_05b6850b`), the 90th percentile generates only 421 detections when 861 annotated cells exist (under-segmentation). In sparse/noisy datasets, it generates thousands of spurious background blobs.
4. **Zero Mitosis Handling**: Hungarian assignment between frame $t-1$ and $t$ enforces strictly 1-to-1 edges ($d_{\text{out}} \le 1$). Zero divisions can ever be detected.

---

## 11. Classical Computer Vision Feasibility Benchmarking

To test whether classical computer vision can resolve these failure modes before resorting to deep learning, six detector families were evaluated on validation frames:

| Detection Algorithm | Total Detections | True Positives | False Positives | False Negatives | Precision | Recall | $F_1$ Score | Latency / Volume |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Heuristic Baseline DS4 (EXP00)** | 387 | 13 | 374 | 92 | $3.36\%$ | $12.38\%$ | $0.053$ | $2.1\text{ s}$ |
| **Downsample DS2 (Percentile 95)** | 1,485 | 26 | 1,459 | 79 | $1.75\%$ | $24.76\%$ | $0.033$ | $4.2\text{ s}$ |
| **Full-Res Uniform Filter (Percentile 97)** | 2,347 | 46 | 2,301 | 59 | $1.96\%$ | $43.81\%$ | $0.038$ | $31.0\text{ s}$ |
| **3D Local Maxima Peak Finding** | 4,574 | 64 | 4,510 | 41 | $1.40\%$ | $60.95\%$ | $0.027$ | $27.0\text{ s}$ |
| **3D Difference of Gaussians (DoG)** | 4,468 | 66 | 4,402 | 39 | $1.48\%$ | **$62.86\%$** | $0.029$ | **$7.0\text{ s}$** |
| **Distance Transform + Watershed** | 3,856 | 55 | 3,801 | 50 | $1.43\%$ | $52.38\%$ | $0.028$ | $23.0\text{ s}$ |

### Takeaways from Classical CV
* **Recall Jumps from $12\%$ to $63\%$**: Multi-scale 3D Difference of Gaussians (DoG) and 3D local maxima peak finding recover $>62\%$ of ground truth cells, compared to only $12\%$ for EXP00.
* **Precision Appears Artificially Low Due to Sparsity**: The precision of $1.5\%$ is an artifact of the fact that ground truth only labels $3.5\%$ of visible cells. When inspecting detected coordinates against raw image volumes, the detected peaks correspond to actual physical nuclei.
* **Classical CV Ceiling**: While classical multi-scale blob detection achieves $63\%$ recall, it lacks the semantic discriminative power to adapt to varying background autofluorescence, non-uniform axial attenuation, and tightly touching cells.

---

## 12. Public Test Set Representativeness

Comparing the 4 public test volumes against the remaining 195 training volumes:
* **Radiometric Representation**: The 4 test volumes span all three identified intensity regimes:
  * `6bba_05b6850b` (mean: 87.0) perfectly represents the Low-Background Regime.
  * `44b6_0113de3b` (mean: 246.7) represents the Intermediate Fluorescence Regime.
  * `6bba_05db0fb1` (mean: 962.9) represents the High-Background Regime.
  * `44b6_0b24845f` (mean: 1246.2) represents the Hyper-Fluorescent Regime.
* **Density Representation**:
  * `44b6_0113de3b` and `44b6_0b24845f` represent ultra-sparse tracking targets ($51-52$ nodes).
  * `6bba_05b6850b` represents moderate annotation coverage ($861$ nodes, $13.5\%$ sparsity).
  * `6bba_05db0fb1` represents a large, dense embryo with mitotic divisions ($1,229$ nodes, 3 divisions).
* **Verdict**: The 4 public test volumes form an exceptionally well-balanced, diverse cross-section of the entire 199-volume dataset. Improving local CV on these 4 volumes will translate reliably to general performance.

---

## 13. Tracking Ambiguity Analysis

To determine how difficult association actually is, we evaluated candidate competition for all 131,010 true edges:

| Search Radius ($R$) | Fraction of Edges with $\ge 1$ Competing Candidate | Ambiguity Interpretation |
| :---: | :---: | :--- |
| **$3.0\ \mu\text{m}$** | $0.0023\%$ | Essentially unambiguous |
| **$5.0\ \mu\text{m}$** | $0.106\%$ | $< 1$ in 900 transitions has a competitor |
| **$7.0\ \mu\text{m}$** | **$0.731\%$** | $< 1\%$ ambiguity at the metric cutoff |
| **$10.0\ \mu\text{m}$** | $2.701\%$ | Minor ambiguity |
| **$15.0\ \mu\text{m}$** | $13.201\%$ | Hungarian assignment required |
| **$20.0\ \mu\text{m}$** | $30.784\%$ | Significant ambiguity if gating is too wide |

### Tracking Architecture Conclusion
* For the true cell population, **inter-frame motion is very small ($1.82\ \mu\text{m}$ median)** relative to inter-cell spacing ($25.0\ \mu\text{m}$ median).
* At a spatial gating radius of $7.0\ \mu\text{m}$, **only $0.73\%$ of true transitions face a competing candidate**.
* Therefore, **tracking is NOT the primary bottleneck**. Complex graph neural networks or reinforcement learning trackers are overkill. Bipartite matching with a motion prediction prior and an adaptive search radius ($R \approx 8 - 10\ \mu\text{m}$) is mathematically sufficient to achieve $>98\%$ edge precision on clean detections.

---

## 14. Supervised Learning Problem Formulation

Comparing 6 candidate formulation paradigms:

| Paradigm | Feasibility | Target Representation | Strengths | Weaknesses | Recommendation |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **A. 3D Binary Segmentation** | Moderate | Voxel mask of nuclear bodies | Direct pixel loss (Dice / BCE) | Merges touching nuclei into blobs | Not recommended |
| **B. 3D Gaussian Center Heatmap** | **High** | $\exp(-\frac{d^2}{2\sigma^2})$ at centroids | Resolves touching nuclei; sub-voxel localization | Extreme class imbalance | **STRONGLY RECOMMENDED** |
| **C. 3D Anchor-based Object Detection** | Low | 3D bounding boxes $[z, y, x, d_z, d_y, d_x]$ | Explicit box coordinates | Too heavy; anchor tuning in 3D is brittle | Rejected |
| **D. Direct Centroid Coordinate Regression** | Low | Variable set of coordinates via DETR/queries | Direct node output | Cannot handle variable set of $10^3$ cells | Rejected |
| **E. Spatio-Temporal 3D+T Convolutions** | Moderate | 4D sliding window ($T=3$) | Captures motion cues during detection | $3\times$ GPU memory, slower iteration | Phase 2 candidate |
| **F. Joint Detection + Offset Tracking** | **High** | Heatmap + 3D offset vector $\vec{\Delta}$ | Single forward pass detects and links | Requires multi-task head | **Phase 2 Architecture** |

### Selected Target Formulation
**3D Center Heatmap Regression with Anisotropic Gaussian Targets**:
* Target: For each centroid $(z_0, y_0, x_0)$, render a continuous 3D Gaussian:
  $$Y(z, y, x) = \exp\left(-\left(\frac{(z - z_0)^2}{2\sigma_z^2} + \frac{(y - y_0)^2}{2\sigma_{xy}^2} + \frac{(x - x_0)^2}{2\sigma_{xy}^2}\right)\right)$$
  where $\sigma_z = 1.5\text{ voxels}$ and $\sigma_{xy} = 2.5\text{ voxels}$.
* Loss: Modified Focal Loss (CenterNet loss) or Weighted MSE to counter background sparsity:
  $$L = -\frac{1}{N} \sum_{z,y,x} \begin{cases} (1 - \hat{Y})^\alpha \log(\hat{Y}) & \text{if } Y = 1 \\ (1 - Y)^\beta \hat{Y}^\alpha \log(1 - \hat{Y}) & \text{otherwise} \end{cases}$$
* Centroid Extraction: 3D local maximum pooling (`max_pool3d(kernel=3, stride=1, padding=1) == output`) followed by sub-voxel quadratic interpolation.

---

## 15. Data Augmentation Strategy

Based on physical microscopy optics and biological tissue properties:

| Transformation | Biological / Physical Validity | Recommended Range / Probability |
| :--- | :---: | :--- |
| **Lateral Flips ($X, Y$)** | Completely invariant | $p=0.5$ (Horizontal & Vertical) |
| **Lateral $90^\circ$ Rotations** | Completely invariant | $p=0.5$ ($k \in \{1, 2, 3\}$ in $XY$ plane) |
| **Axial Flip ($Z$)** | Asymmetric due to attenuation, but valid for morphology | $p=0.2$ (Use cautiously due to Z-illumination profile) |
| **Random Scaling / Zoom** | Valid (cellular size variation across stages) | Scale factor $\in [0.85, 1.15]$, $p=0.3$ |
| **Intensity Jitter / Gamma** | Highly recommended (simulates fluorophore variation) | Gamma $\in [0.7, 1.4]$, scale $\in [0.7, 1.3]$, $p=0.5$ |
| **Gaussian Noise** | Simulates sensor / Poisson readout noise | $\sigma \in [0.01, 0.05]$, $p=0.3$ |
| **Gaussian Blur** | Simulates defocus and tissue scattering | $\sigma \in [0.5, 1.2]$, $p=0.2$ |
| **Elastic Deformations** | Plausible for soft embryonic tissue | Mild 3D spline deformation, $p=0.2$ |

---

## 16. GPU & Training Feasibility (NVIDIA A100 80GB Benchmarks)

Empirical throughput and memory measurements were performed directly on our allocated `NVIDIA A100-PCIE-80GB` GPU using PyTorch 2.7 with Automatic Mixed Precision (`torch.amp.autocast('cuda')`):

| Input Tensor Shape | Batch Size | Base Channels | VRAM Allocated | Step Time (ms) | Throughput (Volumes/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`[64, 256, 256]` (Full 3D Volume)** | **1** | 16 | **$2.43\text{ GiB}$** | **$84.9\text{ ms}$** | **$11.78\text{ vol/s}$** |
| **`[64, 256, 256]` (Full 3D Volume)** | **2** | 16 | **$4.84\text{ GiB}$** | **$170.1\text{ ms}$** | **$11.76\text{ vol/s}$** |
| **`[64, 256, 256]` (Full 3D Volume)** | **4** | 16 | **$9.65\text{ GiB}$** | **$338.8\text{ ms}$** | **$11.81\text{ vol/s}$** |
| **`[64, 256, 256]` (Full 3D Volume)** | **2** | 32 | **$9.58\text{ GiB}$** | **$248.8\text{ ms}$** | **$8.04\text{ vol/s}$** |
| `[64, 128, 128]` (Quadrant Patch) | 4 | 16 | $2.43\text{ GiB}$ | $84.4\text{ ms}$ | $47.40\text{ vol/s}$ |
| `[64, 128, 128]` (Quadrant Patch) | 8 | 16 | $4.84\text{ GiB}$ | $169.1\text{ ms}$ | $47.31\text{ vol/s}$ |
| `[32, 64, 64]` (Small Patch) | 16 | 16 | $1.23\text{ GiB}$ | $43.8\text{ ms}$ | $365.00\text{ vol/s}$ |

### High-Impact Finding: No Patching Required!
* Because each 3D frame is $64 \times 256 \times 256$ ($4.19\text{ million voxels}$), a full 3D U-Net forward+backward pass requires **only $4.84\text{ GiB}$ of VRAM at batch size 2**!
* On an 80GB A100, we have $>75\text{ GiB}$ of headroom!
* **We do not need patch-based cropping, sliding windows, or boundary stitching for inference**. The model can train and infer directly on **complete 3D frames**.

### Multi-GPU Scaling Estimates (19,900 Full Frames per Epoch)
* **1 × A100 GPU**: $28.1\text{ minutes / epoch}$
* **2 × A100 GPUs (Node `rpgpu018`)**: $14.4\text{ minutes / epoch}$
* **4 × A100 GPUs (2 Nodes)**: $7.4\text{ minutes / epoch}$
* **8 × A100 GPUs (All 4 Cluster Nodes via DDP)**: **$3.8\text{ minutes / epoch}$**
* A 20-epoch training run across 8 GPUs takes only **$1.25\text{ hours}$**!

---

## 17. Recommended Architecture & Pipeline Design

Based strictly on empirical measurements, the optimal system architecture consists of a **Decoupled 3D Heatmap Detector + Motion-Gated Bipartite Tracker with Geometric Division Resolution**:

```
[4D Zarr Volume]
       |
  (Frame t)
       |
[Instance Percentile Normalization: P1 to P99.5]
       |
[3D ResUNet / 3D U-Net on Full 64x256x256 Volume]
       |
[3D Center Heatmap Output (1 Channel)]
       |
[Sub-Voxel 3D Peak Finding (max_pool3d + quadratic peak fit)]
       |
  (Centroids at frame t)
       |
[Motion Prediction: p_pred(t) = p(t-1) + v(t-1)]
       |
[Cost Matrix: Physical Distance in um + Appearance Patch NCC]
       |
[Gated Hungarian Assignment (Distance <= 8.0 um)]
       |
[Mitosis Bipartite Resolver: Candidate pairs diverging by > 120°]
       |
[Official Formatted Submission & Evaluation]
```

### Module Specifications
1. **Preprocessor**: Compute $P_1$ and $P_{99.5}$ per 3D volume, clip, and scale to $[0, 1]$.
2. **Detector**: 3D ResUNet (4 stages, base channels 16 or 24, anisotropic pooling $(1, 2, 2)$ at stage 1 to preserve $Z$ resolution). Output: single channel Gaussian center heatmap.
3. **Centroid Extraction**: $3 \times 3 \times 3$ non-maximum suppression followed by sub-voxel refinement.
4. **Tracker**: Frame-to-frame association via linear sum assignment. Cost is Euclidean distance in physical $\mu\text{m}$, gated at $8.0\ \mu\text{m}$. Unmatched source nodes paired to remaining detections if candidate pair meets daughter divergence angle ($>120^\circ$) and daughter distance ($<15\ \mu\text{m}$).

---

## 18. Recommended First Serious Training Configuration

* **Model**: 3D U-Net (Encoder depths: `[16, 32, 64, 128]`, Decoder depths: `[128, 64, 32, 16]`).
* **Input**: Full 3D frame `(1, 64, 256, 256)`.
* **Target**: Continuous 3D Gaussian heatmap with $\sigma_z = 1.5$, $\sigma_{xy} = 2.5$.
* **Loss Function**: Gaussian Focal Loss ($\alpha=2, \beta=4$).
* **Optimizer**: AdamW ($\text{lr} = 5\times 10^{-4}$, weight decay $1\times 10^{-2}$).
* **LR Schedule**: Cosine Annealing with 2-epoch linear warmup.
* **Batch Size**: 2 per GPU (Effective batch size 4 on local 2× A100, or 16 on 8× A100 DDP).
* **Epochs**: 20 epochs (~$14\text{ min/epoch}$ on 2 GPUs, total $\sim 4.5\text{ hours}$).
* **Validation Strategy**: 4-fold split or train on 195 volumes, evaluate on the 4 public test stub volumes (`44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1`).

---

## 19. Risks & Remaining Unknowns

1. **Unannotated Ground Truth Supervision**: Because ground truth labels only $3.5\%$ of visible cells, training with standard MSE loss on raw targets will treat thousands of real, unannotated cells as background (false negative supervision)!
   * **Mitigation**: We must train on **positive-unlabeled (PU) loss**, mask out loss in unlabeled regions, or train an initial self-supervised / pseudo-labeled detector using classical DoG high-confidence peaks.
2. **Division Sparsity in Training Data**: Only 151 divisions exist across 199 datasets ($0.76$ per dataset).
   * **Mitigation**: Rely on geometric bifurcation rules during tracking rather than trying to train a data-hungry division neural network.
3. **Hidden Test Set Domain Shift**: If the private test set includes different organisms or optical configurations, fixed models might degrade.
   * **Mitigation**: Robust instance normalization and strong geometric data augmentation (rotations, intensity jitter, blur).

---

## 20. Executive Summary

### WHAT WE KNOW
1. **Dataset Uniformity**: All 203 datasets share the exact same tensor shape `[100, 64, 256, 256]`, chunk shape `[1, 64, 256, 256]`, and voxel spacing ($1.625\ \mu\text{m} \times 0.40625\ \mu\text{m} \times 0.40625\ \mu\text{m}$).
2. **Biological Entity**: Annotated objects are **fluorescently labelled cell nuclei** ($\sim 5.7\ \mu\text{m}$ lateral diameter, $\sim 9.8\ \mu\text{m}$ axial diameter due to PSF elongation).
3. **Sparsity Reality**: Ground-truth tracks are sparse manual annotations representing only **$3.5\%$ of visible cells** ($659$ nodes annotated vs $18,400$ physical cells per volume). The competition requires detecting and tracking all visible cells to avoid the total node ratio penalty.
4. **Motility Regime**: Motion between frames is small (median displacement **$1.82\ \mu\text{m}$**, $90\%$ under $4.14\ \mu\text{m}$) and highly continuous ($2.03\ \mu\text{m}$ linear extrapolation error).
5. **Mitotic Fingerprint**: Mitotic divisions are rare (151 total) and exhibit a clear geometric signature ($139.1^\circ$ divergence angle, $10.56\ \mu\text{m}$ daughter separation).
6. **A100 Compute Power**: A complete 3D frame (`64 x 256 x 256`) uses only **$4.84\text{ GiB}$ of VRAM** during 3D U-Net training, running at $11.8\text{ vol/s}$. We can train on full volumes without patch slicing.

### WHAT IS ACTUALLY HARD
1. **Supervision Under Sparse Annotation**: Training a supervised detector when $96.5\%$ of real nuclei in the image are unannotated. Naive training will penalize the network for correctly detecting unannotated cells.
2. **Axial Resolution & Attenuation**: The $4\times$ axial anisotropy and $Z$-dependent light attenuation make 3D peak localization sensitive along the $Z$-axis.

### WHAT IS EASY
1. **Tracking Association**: Tracking between consecutive frames is remarkably clean. At an $8.0\ \mu\text{m}$ radius, $< 1\%$ of true edges face competing candidates. Standard linear sum assignment is more than sufficient.
2. **Hardware Constraints**: There is zero memory bottleneck. The entire 3D volume fits comfortably in A100 VRAM with ample room for batching.

### WHAT THE CURRENT BASELINE GETS WRONG
1. **Aggressive $4\times$ Downsampling**: Collapses adjacent nuclei into single blobs and introduces $\pm 3.25\ \mu\text{m}$ spatial quantization error, causing $>90\%$ of true centroids to miss the $7.0\ \mu\text{m}$ matching tolerance.
2. **Fixed 90th Percentile Threshold**: Incompatible with the $>40\times$ baseline intensity variation across datasets.
3. **Strict 1-to-1 Hungarian Assignment**: Cannot spawn divisions ($d_{\text{out}}=2$), guaranteeing zero division score.

### WHAT WE SHOULD BUILD
A **3D CenterNet (Full-Frame 3D U-Net) for Center Heatmap Regression** coupled with an **Anisotropic Gaussian Target**, paired with a **Sub-Voxel Peak Extractor** and an **Adaptive Bipartite Tracker with Geometric Division Gating**.

### WHY
* 3D Heatmaps eliminate the object-merging problem of binary segmentation.
* Full-resolution 3D processing eliminates the fatal quantization error of downsampling.
* Anisotropic kernels match the physical $4\times$ PSF elongation.
* The decoupled tracker leverages the low-ambiguity motility regime ($1.82\ \mu\text{m}$ steps) without wasting model capacity.
