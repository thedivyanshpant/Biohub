# Biohub Cell Tracking — Submission 1 (Kaggle Offline Pipeline)

This directory contains the self-contained, reproducible Kaggle Code Competition submission for the **Biohub Cell Tracking During Development** competition. It reproduces our validated **0.90017 benchmark pipeline** using an offline, resource-constrained architecture.

---

## Notebook Overview

* **File:** [`notebook/submission1.ipynb`](notebook/submission1.ipynb)
* **Target output:** `/kaggle/working/submission.csv`
* **Format:** Jupyter Notebook (`nbformat` 4.5), strictly 11 sequential code cells.
* **Execution limit:** Designed to complete comfortably within 7.5 hours (well under the 12-hour limit).
* **Internet access:** **DISABLED** (`internet = False`). Zero PyPI, Git, or external API calls.

---

## 11-Cell Sequential Structure

1. **Cell 1:** Environment inspection, random seeds (42), and deterministic locking of all 40+ `BIOHUB_*` configuration variables (incorporating the PPSWEEP winning overrides).
2. **Cell 2:** Automatic discovery of Kaggle input directories (`/kaggle/input/...`), checkpoints, and test volumes (`*.zarr`).
3. **Cell 3:** Materialize tracking repository (`tracking_repo`) to `/kaggle/working` and install missing wheels offline (`--no-index --no-deps`).
4. **Cell 4:** Verify that all required dependencies (`tracksdata`, `zarr`, `ilpy`, `pyscipopt`, `geff`, `polars`, `blosc2`, etc.) import cleanly without internet.
5. **Cell 5:** Cryptographic SHA256 integrity check and architecture verification for all three models (Model A, Model B, DeepCenter).
6. **Cell 6:** Execute primary inference with dynamic AST patches (D4 8-way TTA, dual-seed consensus blending, retention guard, bidirectional harmonic probability fusion, multi-GPU sharding / single-GPU fallback, ILP solving).
7. **Cell 7:** Verify and inspect raw candidate prediction graphs (`.geff`).
8. **Cell 8:** Execute the full 11-stage graph healing pipeline with DeepCenter spatial veto (`load_deepcenter_veto_detector` and `deepcenter_accept_repair_point`) and flow-mode motion relinking.
9. **Cell 9:** Verify generated `/kaggle/working/submission.csv` file presence, size, and row count.
10. **Cell 10:** Strict Kaggle submission schema and graph invariant validation (contiguous IDs, binary division degree <= 2, in-degree <= 1, strictly forward-in-time `t -> t+1`, zero NaNs).
11. **Cell 11:** Final concise diagnostic summary (per-dataset statistics, row counts, total run time).

---

## Attached Kaggle Datasets & Mount Paths

When creating the Kaggle submission notebook, attach the following **5 input resources**:

| Input Type | Resource Name | Expected Kaggle Mount Path | Contents |
| :--- | :--- | :--- | :--- |
| **Competition Data** | `biohub-cell-tracking-during-development` | `/kaggle/input/competitions/biohub-cell-tracking-during-development` | Test `.zarr` volumes |
| **Model A (Primary)** | `thedivyanshpant/biohub-tracking-support-pack-50ep-v1` | `/kaggle/input/models/thedivyanshpant/biohub-tracking-support-pack-50ep-v1/pytorch/default/1` | `edge_predictor_best.pth` (8.4 MB) |
| **Model B (Seed 314159)** | `thedivyanshpant/biohub-temporal-unet3d-seed314159-v1` | `/kaggle/input/models/thedivyanshpant/biohub-temporal-unet3d-seed314159-v1/pytorch/default/1` | `edge_predictor_best.pth` (8.4 MB) |
| **DeepCenter Veto** | `thedivyanshpant/biohub-deepcenter-unet3d-center-prior-v1` | `/kaggle/input/models/thedivyanshpant/biohub-deepcenter-unet3d-center-prior-v1/pytorch/default/1` | `best.pt` (37.9 MB) |
| **Support Dataset** | `thedivyanshpant/biohub-kaggle-support` | `/kaggle/input/datasets/thedivyanshpant/biohub-kaggle-support` | `repo/`, `wheels/` (62 .whl), `configs/` |

*(Note: The notebook also supports dynamic discovery across any username and flat `/kaggle/input/<slug>` layouts.)*

---

## Model Architecture Details

### Model A & Model B: Composite `UNetNodeTransformer`
* Trained weights inside `edge_predictor_best.pth`:
  * `unet.*`: 3D temporal convolutional feature backbone (`TemporalUNet3D`)
  * `detect_head.*`: cell center regression and detection head
  * `transformer.*`: **`SimpleNodeTransformer`** cross-frame edge prediction attention layers
* Both models run simultaneously in-memory during Cell 6.

### DeepCenter: 3D Full-Frame Center Prior U-Net
* Trained weights inside `best.pt`.
* Evaluated dynamically during Cell 8 post-processing to validate/veto candidate gap bridges and cell divisions.

---

## Local Testing

To run the local dry-run verification suite:
```bash
python3 scripts/test_pipeline_local.py
```
