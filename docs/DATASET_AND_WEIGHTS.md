# Dataset Setup & Pretrained Model Weights

This guide outlines the dataset directory structure, file formats, and instructions for downloading required model weights.

---

## 1. Expected Dataset Directory Structure

The competition dataset is organized into 3D volumetric image stores (`.zarr`) and directed lineage graphs (`.geff`):

```
database/
├── sample_submission.csv
├── test/
│   ├── 44b6_0113de3b.zarr/
│   ├── 44b6_0b24845f.zarr/
│   ├── 6bba_05b6850b.zarr/
│   └── 6bba_05db0fb1.zarr/
└── train/
    ├── 44b6_0113de3b.geff/
    ├── 44b6_0113de3b.zarr/
    ├── 44b6_0b24845f.geff/
    ├── 44b6_0b24845f.zarr/
    ├── 6bba_05b6850b.geff/
    ├── 6bba_05b6850b.zarr/
    ├── 6bba_05db0fb1.geff/
    ├── 6bba_05db0fb1.zarr/
    └── ... (400+ training embryo volume pairs)
```

### Data Formats & Coordinate System
* **3D Volumetric Images (`.zarr`):** 4D arrays $(T, Z, Y, X)$ representing time series of 3D light-sheet microscopy volumes.
* **Physical Voxel Scale:**
  $$\text{Scale} = (\Delta z, \Delta y, \Delta x) = (1.625, 0.40625, 0.40625)\,\mu\text{m/voxel}$$
* **Lineage Graphs (`.geff`):** Zarr-based graph specification stores:
  - `nodes/ids`: Unique integer cell identifier.
  - `nodes/props/t/values`: Time frame index $t \in [0 \dots T-1]$.
  - `nodes/props/z/values`, `y/values`, `x/values`: Physical cell centroid coordinates.
  - `edges/ids`: Directed track association links connecting $(u, v)$ with $t(u) \to t(v)$.
  - `attributes/geff/extra/estimated_number_of_nodes`: Ground-truth embryo node count estimation.

---

## 2. Obtaining Competition Data

To download the official dataset via the Kaggle CLI:

```bash
# Ensure your Kaggle API credentials (~/.kaggle/kaggle.json) are active
mkdir -p database && cd database
kaggle competitions download -c biohub-cell-tracking-during-development
unzip biohub-cell-tracking-during-development.zip
rm biohub-cell-tracking-during-development.zip
```

Environment variable override:
```bash
export BIOHUB_DATA_DIR="/path/to/database"
```

---

## 3. Obtaining Pretrained Model Weights

Due to file size constraints, large binary model checkpoints (`*.pth`, `*.pt`) are excluded from Git version control. You can obtain the exact weights using the Kaggle API or direct links below:

### Model Checkpoints Inventory & Checksums

| Model | Resource Slug | File Path | Size | SHA256 Checksum |
| :--- | :--- | :--- | :---: | :--- |
| **Model A (Primary 50ep)** | `thedivyanshpant/biohub-tracking-support-pack-50ep-v1` | `weights/unet_transformer/split_0/edge_predictor_best.pth` | 8.4 MB | `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771` |
| **Model B (Seed 314159)** | `thedivyanshpant/biohub-temporal-unet3d-seed314159-v1` | `weights/unet_transformer/split_0/edge_predictor_best.pth` | 8.4 MB | `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f` |
| **DeepCenter Veto** | `thedivyanshpant/biohub-deepcenter-unet3d-center-prior-v1` | `weights/full_frame_center/best.pt` | 37.9 MB | `8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0` |
| **Model C (Seed 42)** | Trained locally via DDP | `experiments/model_c_seed42/edge_predictor_best.pth` | 8.0 MB | Checksum recorded in experiment report |

### Download Commands

Run the following commands from the repository root:

```bash
mkdir -p kaggle_artifacts && cd kaggle_artifacts

# 1. Download Model A (Support Pack)
kaggle models instances versions download thedivyanshpant/biohub-tracking-support-pack-50ep-v1/PyTorch/default/1 --untar
mv 1 biohub-tracking-support-pack-50ep-v1

# 2. Download Model B (Seed 314159)
kaggle models instances versions download thedivyanshpant/biohub-temporal-unet3d-seed314159-v1/PyTorch/default/1 --untar
mv 1 biohub-temporal-unet3d-seed314159-v1

# 3. Download DeepCenter 3D U-Net
kaggle models instances versions download thedivyanshpant/biohub-deepcenter-unet3d-center-prior-v1/PyTorch/default/1 --untar
mv 1 biohub-deepcenter-unet3d-center-prior-v1

cd ..
```

### Checkpoint Verification

You can verify the cryptographic integrity of all downloaded model checkpoints by running:

```bash
python3 Submission1/scripts/package_submission.py
```
This utility automatically validates file paths, sizes, and JSON schemas.
