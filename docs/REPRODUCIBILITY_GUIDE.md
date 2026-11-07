# Reproducibility & Execution Guide

This guide details how to reproduce inference pipelines, local evaluations, and multi-GPU training experiments across both workstation environments and Kaggle competition containers.

---

## 1. Environment Setup

### Option A: Conda / Mamba Environment (Recommended for HPC / Local GPU)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate biohub

# Install repository in editable mode
pip install -e .
```

### Option B: Pip Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 2. Pipeline Verification & Dry-Run Tests

To test the entire pipeline without running full inference on multi-gigabyte volumes, run the local verification suite:

```bash
# 1. Check Kaggle submission package readiness and checkpoint integrity
python3 Submission1/scripts/package_submission.py

# 2. Run local end-to-end component dry-run test
python3 Submission1/scripts/test_pipeline_local.py
```

The test suite validates:
1. Deterministic seeding and environment variables.
2. Checkpoint SHA256 integrity checks.
3. AST patch compilation for inference engines.
4. DeepCenter 3D U-Net forward tensor shapes.
5. Submission schema and graph invariant checks.

---

## 3. Kaggle Offline Submission Execution

The Kaggle Code Competition operates under strict offline constraints (0 internet access, max 12 hours run time).

### Uploading to Kaggle
1. Open [`Submission1/notebook/submission1.ipynb`](../Submission1/notebook/submission1.ipynb) or [`biohub-harmonic-fusion-v3.ipynb`](../biohub-harmonic-fusion-v3.ipynb).
2. Attach the 5 Kaggle resources as documented in [`docs/DATASET_AND_WEIGHTS.md`](DATASET_AND_WEIGHTS.md):
   - Competition Data: `biohub-cell-tracking-during-development`
   - Model A: `thedivyanshpant/biohub-tracking-support-pack-50ep-v1`
   - Model B: `thedivyanshpant/biohub-temporal-unet3d-seed314159-v1`
   - DeepCenter: `thedivyanshpant/biohub-deepcenter-unet3d-center-prior-v1`
   - Support Dataset: `biohub-kaggle-support` (packaged from `Submission1/kaggle_support_pack`)
3. Ensure GPU accelerator is enabled (1× or 2× NVIDIA T4).
4. Toggle **Internet OFF**.
5. Click **Run All** or **Submit to Competition**.

---

## 4. End-to-End HPC Leader Reproduction

To execute the full multi-GPU HPC inference pipeline:

```bash
# Export paths if custom locations are used
export BIOHUB_ROOT="$(pwd)"
export BIOHUB_DATA_DIR="$(pwd)/database"

# Run end-to-end reproduction orchestrator
python3 -m leader_reproduction.run
```

This runs:
- Stage 1: Environment locking (Cell 0).
- Stage 2: Artifact integrity verification (Cell 3).
- Stage 3: AST patch compilation and multi-GPU sharded inference.
- Stage 4: 11-stage graph healing and DeepCenter spatial veto post-processing.
- Stage 5: Local scoring on the 4 held-out ground-truth validation volumes.

---

## 5. Model C Training from Scratch (Seed 42)

To retrain Model C across 4 GPUs using PyTorch Distributed Data Parallel (DDP):

```bash
# 1. Build training dataset metadata cache
python3 experiments/model_c_seed42/build_metadata_cache.py

# 2. Run quick smoke test on 1 volume
python3 experiments/model_c_seed42/smoke_test.py

# 3. Launch 4-GPU distributed training
torchrun --nproc_per_node=4 experiments/model_c_seed42/train_model_c.py \
    --config experiments/model_c_seed42/config.yaml \
    --epochs 50 \
    --lr 1e-4

# 4. Evaluate trained checkpoint and analyze predictive diversity
python3 experiments/model_c_seed42/evaluate_and_analyze.py
```

---

## 6. Dense-Volume Post-Processing Experiments

To inspect error sources and run density-adaptive gap closing experiments on dense embryos (`6bba_05db0fb1`, >70,000 cells):

```bash
# 1. Error source attribution diagnostics
python3 experiments/dense_volume/run_diagnostics.py

# 2. Dense-volume detection threshold sweep (Experiment 1)
python3 experiments/dense_volume/run_exp1.py

# 3. Density-adaptive gap closing sweep (Experiment 2)
python3 experiments/dense_volume/run_exp2.py

# 4. Generate optimized dense-volume submission
python3 experiments/dense_volume/generate_best_submission.py
```
