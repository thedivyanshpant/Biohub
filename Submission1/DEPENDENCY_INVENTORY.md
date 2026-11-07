# Dependency Inventory — Biohub Cell Tracking Submission 1

This document lists all Python dependencies, runtime requirements, and offline package wheels required to execute the pipeline with **internet access disabled**.

---

## 1. Runtime Environment Assumptions
* **Platform:** Kaggle Linux GPU Container (Ubuntu 22.04 LTS)
* **Python Version:** 3.12 (specifically `cp312`)
* **Accelerator:** 1 or 2 NVIDIA Tesla T4 GPUs (or single P100 / A100)
* **RAM:** ~30 GB system RAM
* **Disk:** `/kaggle/working` with >= 15 GB free space

---

## 2. Standard Kaggle Pre-Installed Packages (No installation required)
These packages are provided natively by Kaggle's base Python 3.12 Docker image:
* `torch` / `torchvision` (CUDA-enabled)
* `numpy`
* `scipy`
* `pandas`
* `scikit-image` (`skimage`)
* `tqdm`
* `pyyaml`
* `requests`

---

## 3. Bundled Offline Wheels (`wheels/` in Support Pack)
The support pack at `biohub-tracking-support-pack-50ep-v1/wheels/` contains 62 pre-compiled wheels installed offline via `pip install --no-index --no-deps --find-links ...`:

| Category | Package Name | Wheel Filename |
| :--- | :--- | :--- |
| **Tracking & Graph** | `tracksdata` | `tracksdata-0.1.0rc6.dev3+g980c2d30a-py3-none-any.whl` |
| **Graph Format** | `geff` | `geff-1.2.0.1.1-py3-none-any.whl` |
| **Graph Spec** | `geff_spec` | `geff_spec-1.1.1-py3-none-any.whl` |
| **ILP Solver** | `ilpy` | `ilpy-0.6.0-py3-none-any.whl` |
| **SCIP Optimizer** | `pyscipopt` | `pyscipopt-6.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` |
| **Data Storage** | `zarr` | `zarr-3.2.1-py3-none-any.whl` |
| **Dataframes** | `polars` | `polars-1.42.0-py3-none-any.whl` |
| **Polars Runtime** | `polars_runtime_32` | `polars_runtime_32-1.42.0-cp310-abi3-manylinux2014_x86_64.whl` |
| **Compression** | `blosc2` | `blosc2-4.7.0-cp312-cp312-manylinux2014_x86_64.whl` |
| **Codec Engine** | `numcodecs` | `numcodecs-0.15.1-cp312-cp312-manylinux2014_x86_64.whl` |
| **Image Codecs** | `imagecodecs` | `imagecodecs-2026.6.26-cp312-abi3-manylinux_2_28_x86_64.whl` |
| **Graph Network** | `rustworkx` | `rustworkx-0.18.0-cp310-abi3-manylinux2014_x86_64.whl` |
| **Data Validation** | `pydantic` | `pydantic-2.13.4-py3-none-any.whl` |
| **Pydantic Core** | `pydantic_core` | `pydantic_core-2.46.4-cp312-cp312-manylinux2014_x86_64.whl` |
| **Config Management**| `donfig` | `donfig-0.8.1.post1-py3-none-any.whl` |
| **Utilities** | `psygnal`, `rich`, `bidict`, `click`, `cloudpickle`, `dask`, etc. | Included in bundle |

---

## 4. Custom Local Packages
* **`biohub_tracking`**: Located inside the support pack at `repo/src/biohub_tracking/`.
  * `biohub_tracking.models.temporal_unet.TemporalUNet3D`
  * `biohub_tracking.models.simple_node_transformer.SimpleNodeTransformer`
  * `biohub_tracking.io` (GEFF and Zarr data loading)
  * `biohub_tracking.img_proc` (local coordinate extraction)
  * `biohub_tracking.metrics` (graph scoring)
* **`_DCDeepCenterUNet3D`**: DeepCenter 3D U-Net center prior architecture declared natively in post-processing.
