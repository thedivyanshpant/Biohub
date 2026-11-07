# Dry-Run & Test Report — Biohub Cell Tracking Submission 1

**Date of Execution:** 2026-09-26  
**Environment:** Linux (Ubuntu x86_64), Python 3.12 (Kaggle compatible), CUDA PyTorch  
**Test Suite:** `scripts/test_pipeline_local.py`  
**Overall Status:** **PASSED (100% SUCCESS)**

---

## Component Verification Results

| Stage | Test Description | Verification Target | Status | Details |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1** | Deterministic Setup & Env Vars | All 40+ `BIOHUB_*` variables | **PASS** | PPSWEEP winning overrides & runtime caps locked |
| **Step 2** | Checkpoint & Input Discovery | Paths to Models A, B, DeepCenter | **PASS** | Auto-detection works across Kaggle & local paths |
| **Step 3** | Repo Materialization & Import | `biohub_tracking` package | **PASS** | `TemporalUNet3D` & `SimpleNodeTransformer` loaded |
| **Step 4** | Checkpoint Cryptographic Hash | SHA256 checksums | **PASS** | All 3 model checkpoints match expected hashes bit-for-bit |
| **Step 5** | AST Patches on Inference Script | `predict_unet_transformer.py` | **PASS** | All 6 AST patches compile cleanly without anchor mismatches |
| **Step 6** | DeepCenter 3D U-Net Architecture | Forward tensor pass | **PASS** | Input `(1, 1, 8, 32, 32)` -> Output `(1, 1, 8, 32, 32)` |
| **Step 7** | Submission Format Validation | Schema & Graph Constraints | **PASS** | Contiguous IDs, binary divisions, zero NaNs validated |

---

## Checkpoint Checksum Verification

* **Model A (`edge_predictor_best.pth`):**
  * Computed SHA256: `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`
  * Expected SHA256: `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`
  * Status: **MATCH**

* **Model B (`edge_predictor_best.pth`):**
  * Computed SHA256: `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f`
  * Expected SHA256: `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f`
  * Status: **MATCH**

* **DeepCenter (`best.pt`):**
  * Computed SHA256: `8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0`
  * Expected SHA256: `8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0`
  * Status: **MATCH**

---

## Offline Wheel Compatibility Verification

A clean Python 3.12 virtual environment was created to simulate Kaggle's offline container:
```bash
python3.12 -m pip install --no-index --find-links ./wheels tracksdata zarr pyscipopt geff geff-spec ilpy polars blosc2
```
* **Exit code:** 0 (clean install)
* **Import test:** All modules (`tracksdata`, `zarr`, `pyscipopt`, `geff`, `ilpy`, `polars`, `blosc2`) imported without network access.
* **Result:** **100% Offline Compatible**.
