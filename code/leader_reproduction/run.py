"""
run.py — Main orchestrator for the leader reproduction pipeline.

Reproduces biohub-geometric-fusion.ipynb as a single runnable command:

    conda run -n ML_DFT_V2 python -m leader_reproduction.run

Execution order (faithful to notebook cells):
  1. Set all environment variables (Cell 0)
  2. Parse configuration (Cell 2)
  3. Materialize artifacts + verify integrity (Cell 3)
  4. Apply dynamic source patches (Cell 4)
  5. Run inference on test volumes (Cell 4)
  6. Run post-processing (Cell 5)
  7. Write submission CSV (Cell 5)
  8. Local validation on 4 overlapping volumes (Cell 7+8)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project code directory is on the path
PROJECT_CODE = Path(__file__).resolve().parent.parent
if str(PROJECT_CODE) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE))

from leader_reproduction import config
from leader_reproduction.setup_artifacts import setup_all
from leader_reproduction.patch_predict import apply_all_patches


def step1_environment():
    """Set all environment variables (Cell 0)."""
    print("\n" + "=" * 70)
    print("STEP 1: Setting environment variables")
    print("=" * 70)
    config.set_all_env_vars()
    # HPC-specific: point working dir env var for guard logs etc
    os.environ["BIOHUB_WORKING_DIR"] = str(config.WORKING_DIR)
    print(f"BIOHUB_PRESET: {config.BIOHUB_PRESET}")
    print(f"BIOHUB_SCORE_AXIS: {config.BIOHUB_SCORE_AXIS}")
    cfg = config.parse_config()
    print(f"Config: {json.dumps(cfg, indent=2, default=str)}")
    return cfg


def step2_setup_artifacts():
    """Materialize artifacts and verify integrity (Cell 3)."""
    print("\n" + "=" * 70)
    print("STEP 2: Setting up artifacts")
    print("=" * 70)
    setup_all()


def step3_patch_predict():
    """Apply all dynamic source patches (Cell 4)."""
    print("\n" + "=" * 70)
    print("STEP 3: Applying dynamic source patches")
    print("=" * 70)
    apply_all_patches()


def step4_run_inference(cfg: dict):
    """Run neural network inference (Cell 4)."""
    print("\n" + "=" * 70)
    print("STEP 4: Running inference")
    print("=" * 70)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for inference")

    available_gpu_count = torch.cuda.device_count()
    print(f"CUDA devices: {available_gpu_count}")
    for i in range(available_gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    repo_dir = config.REPO_DIR

    # List test stems
    test_dir = config.TEST_DIR
    test_stems = sorted(
        p.name[:-5] for p in test_dir.iterdir() if p.name.endswith(".zarr")
    )
    print(f"Found {len(test_stems)} test videos")

    # Write splits file
    splits_path = repo_dir / "kaggle_test_splits_50ep.json"
    splits_path.write_text(
        json.dumps([{"split": 0, "train": [], "test": test_stems}], indent=2)
    )

    # Build predict command
    predict_cmd = [
        sys.executable,
        "scripts/predict_unet_transformer.py",
        "--data-dir", str(test_dir),
        "--splits", str(splits_path.resolve()),
        "--split", "0",
        "--weights", config.WEIGHTS_RELATIVE,
        "--unet-batch-size", str(cfg["unet_batch_size"]),
        "--det-threshold", str(cfg["det_threshold"]),
        "--ilp-edge-weight", str(cfg["ilp_edge_weight"]),
        "--ilp-appearance-weight", str(cfg["ilp_appearance_weight"]),
        "--ilp-disappearance-weight", str(cfg["ilp_disappearance_weight"]),
        "--ilp-division-weight", str(cfg["ilp_division_weight"]),
    ]
    if cfg["use_ilp"]:
        predict_cmd.append("--use-ilp")

    # Determine GPU sharding (notebook uses min(2, gpu_count))
    worker_count = min(2, available_gpu_count, len(test_stems))

    env = {**os.environ, "PYTHONPATH": "src"}

    start_time = time.time()

    if worker_count >= 2:
        cuda_tokens = _visible_cuda_tokens(available_gpu_count)[:worker_count]
        processes: dict[int, subprocess.Popen] = {}
        commands: dict[int, list[str]] = {}
        print(f"Launching {worker_count} GPU shards on CUDA devices {cuda_tokens}")

        for shard_index in range(worker_count):
            shard_method = f"{config.METHOD}_gpu{shard_index}"
            shard_cmd = [
                *predict_cmd,
                "--method", shard_method,
                "--slice", f"{shard_index}::{worker_count}",
            ]
            shard_env = {**env}
            shard_env["CUDA_VISIBLE_DEVICES"] = cuda_tokens[shard_index]
            shard_env["BIOHUB_GPU_SHARD"] = f"{shard_index}/{worker_count}"
            print(f"  GPU shard {shard_index}: CUDA={cuda_tokens[shard_index]}")
            commands[shard_index] = shard_cmd
            processes[shard_index] = subprocess.Popen(
                shard_cmd, cwd=repo_dir, env=shard_env,
            )

        _wait_for_prediction_shards(processes, commands)
        _merge_prediction_shards(worker_count, test_stems, repo_dir)
    else:
        print("Using single-process prediction")
        print(" ".join(predict_cmd))
        subprocess.run(predict_cmd, cwd=repo_dir, env=env, check=True)

    elapsed = time.time() - start_time
    print(f"\nInference completed in {elapsed / 60:.2f} minutes")

    return test_stems, elapsed


def step5_post_processing(test_stems: list[str], elapsed: float = 0.0):
    """Run post-processing and write submission (Cell 5)."""
    print("\n" + "=" * 70)
    print("STEP 5: Post-processing and submission")
    print("=" * 70)

    # Import the post-processing module which contains the Cell 5 logic
    # We run it as a subprocess for isolation (same as notebook execution)
    postproc_script = Path(__file__).parent / "postprocess.py"
    if not postproc_script.exists():
        raise FileNotFoundError(
            f"Post-processing script not found: {postproc_script}\n"
            "Run postprocess.py creation first."
        )

    cmd = [
        sys.executable, str(postproc_script),
        "--repo-dir", str(config.REPO_DIR),
        "--test-dir", str(config.TEST_DIR),
        "--working-dir", str(config.WORKING_DIR),
        "--submission-path", str(config.SUBMISSION_PATH),
        "--predict-seconds", str(elapsed),
    ]
    print(" ".join(cmd))
    sub_env = {**os.environ, "PYTHONPATH": str(PROJECT_CODE)}
    subprocess.run(cmd, check=True, env=sub_env)

    if config.SUBMISSION_PATH.exists():
        import csv
        with open(config.SUBMISSION_PATH) as f:
            row_count = sum(1 for _ in csv.reader(f)) - 1
        print(f"\nSubmission written: {config.SUBMISSION_PATH}")
        print(f"Total rows: {row_count}")
    else:
        print("WARNING: Submission file was not created")


def step6_local_validation():
    """Evaluate on the 4 validation volumes (Cells 7+8)."""
    print("\n" + "=" * 70)
    print("STEP 6: Local validation")
    print("=" * 70)

    eval_script = Path(__file__).parent / "evaluate.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {eval_script}")

    cmd = [
        sys.executable, str(eval_script),
        "--repo-dir", str(config.REPO_DIR),
        "--train-dir", str(config.TRAIN_DIR),
        "--test-dir", str(config.TEST_DIR),
        "--working-dir", str(config.WORKING_DIR),
    ]
    sub_env = {**os.environ, "PYTHONPATH": str(PROJECT_CODE)}
    subprocess.run(cmd, check=True, env=sub_env)


# ============================================================
# Helper functions (from Cell 4)
# ============================================================

def _visible_cuda_tokens(count: int) -> list[str]:
    """Parse CUDA_VISIBLE_DEVICES or generate token list."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw and raw != "-1":
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        return tokens[:count]
    return [str(i) for i in range(count)]


def _wait_for_prediction_shards(
    processes: dict[int, subprocess.Popen],
    commands: dict[int, list[str]],
) -> None:
    """Wait for all GPU shards to complete."""
    while processes:
        failed = None
        for shard_index, process in list(processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del processes[shard_index]
            if return_code != 0:
                failed = (shard_index, return_code)
                break
        if failed is None:
            if processes:
                time.sleep(1.0)
            continue
        failed_index, failed_code = failed
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise subprocess.CalledProcessError(failed_code, commands[failed_index])


def _merge_prediction_shards(
    worker_count: int,
    test_stems: list[str],
    repo_dir: Path,
) -> None:
    """Merge prediction outputs from multiple GPU shards."""
    import shutil

    shard_dirs = []
    seen = set()
    expected_all = set(test_stems)

    for shard_index in range(worker_count):
        shard_method = f"{config.METHOD}_gpu{shard_index}"
        matches = sorted((repo_dir / "predictions").glob(f"*/{shard_method}/split_0"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one prediction dir for {shard_method}, found {matches}"
            )
        shard_dir = matches[0]
        expected = set(test_stems[shard_index::worker_count])
        found = {p.stem for p in shard_dir.glob("*.geff")}
        if found != expected:
            raise RuntimeError(
                f"GPU shard {shard_index} mismatch: "
                f"missing={sorted(expected - found)}"
            )
        seen.update(found)
        shard_dirs.append(shard_dir)

    if seen != expected_all:
        raise RuntimeError(f"Merged shards incomplete: missing={sorted(expected_all - seen)}")

    # Merge into single directory
    username_roots = {d.parents[1] for d in shard_dirs}
    final_root = next(iter(username_roots)) / config.METHOD
    final_dir = final_root / "split_0"
    staging_dir = final_root / "split_0_staging"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    for shard_dir in shard_dirs:
        for src in sorted(shard_dir.glob("*.geff")):
            shutil.move(str(src), str(staging_dir / src.name))

    if final_dir.exists():
        shutil.rmtree(final_dir)
    staging_dir.rename(final_dir)

    for shard_dir in shard_dirs:
        shutil.rmtree(shard_dir.parent)

    merged = {p.stem for p in final_dir.glob("*.geff")}
    print(f"Merged {len(merged)} prediction graphs into {final_dir}")


def main():
    """Run the complete reproduction pipeline."""
    print("=" * 70)
    print("BIOHUB LEADER REPRODUCTION PIPELINE")
    print("Faithful reproduction of biohub-geometric-fusion.ipynb")
    print("=" * 70)
    start = time.time()

    # Step 1: Environment
    cfg = step1_environment()

    # Step 2: Artifacts
    step2_setup_artifacts()

    # Step 3: Patches
    step3_patch_predict()

    # Step 4: Inference
    test_stems, elapsed = step4_run_inference(cfg)

    # Step 5: Post-processing
    step5_post_processing(test_stems, elapsed)

    # Step 6: Local validation
    step6_local_validation()

    total = time.time() - start
    print("\n" + "=" * 70)
    print(f"REPRODUCTION COMPLETE in {total / 60:.2f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
