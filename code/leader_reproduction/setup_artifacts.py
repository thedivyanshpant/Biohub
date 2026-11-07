"""
setup_artifacts.py — Faithful reproduction of Cell 3

Materializes the inference repo, verifies SHA256 checksums of all
source files and model weights, and sets up the secondary seed.

Adapted from notebook Cell 3 for HPC paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

from . import config


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_path(path: Path) -> None:
    """Remove file or directory."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def copy_or_extract_tree(src_dir: Path, src_zip: Path, dst: Path) -> None:
    """Copy directory tree or extract zip archive to dst."""
    remove_path(dst)
    if src_dir.exists() and src_dir.is_dir():
        shutil.copytree(src_dir, dst)
        return
    if src_zip.exists() and src_zip.is_file():
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src_zip) as zf:
            zf.extractall(dst)
        return
    raise FileNotFoundError(f"Missing source tree or zip: {src_dir} / {src_zip}")


def link_or_copy_tree(src: Path, dst: Path) -> None:
    """Symlink or copy src to dst."""
    remove_path(dst)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except Exception:
        shutil.copytree(src, dst)


# Expected SHA256 for all support repo Python files
SUPPORT_EXPECTED_SHA256 = {
    "scripts/augmentations.py": "13db09817bf492f8d0f710a0a4d09776320b262060167055090a303fc6057f4e",
    "scripts/dataspec.py": "e69bf952fb985477ac50ff8598a35020c95d20a035a09b81ab4056e655dd311f",
    "scripts/evaluate.py": "614813cc51c3581c6ccda4bb20725a19da8ecac4a27620654bfca58319cffa3c",
    "scripts/predict_unet_transformer.py": "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9",
    "scripts/train_unet_transformer.py": "c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea",
    "src/biohub_tracking/__init__.py": "26a18d8da84e40da73281a48ebc3017d847a2e57431ab63e8629d2109e6e8571",
    "src/biohub_tracking/division_metrics.py": "d1cf1e0a43009d02174f1699ce2aa28458a2220ac4b521731d3bcf31cf8c76be",
    "src/biohub_tracking/img_proc.py": "00e8ef0adc8b39f1aaaa547ea6197b906bf9e8c009e339d3e95f8f8dbf31be3f",
    "src/biohub_tracking/io.py": "efae135b088cecaab463d889f16c885ef6da3ad27b0747327d8ddc28d866b7bd",
    "src/biohub_tracking/metrics.py": "31baf45b54c78f68bab4f65dd8f4b38bca702abb644171c6df7c46cdeef55d83",
    "src/biohub_tracking/models/__init__.py": "ab7587ef79856bae50d24b62e5805092d0459ee1c586522b763f9ef70c093e1d",
    "src/biohub_tracking/models/simple_node_transformer.py": "b97209edeb03840e80d903e3e2a8c81c520641c8ef343f6ca2904d0f80db064e",
    "src/biohub_tracking/models/temporal_unet.py": "d809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac",
}
SUPPORT_EXPECTED_MANIFEST_SHA256 = "978b626d1fd1e7397435a437dfe68691defe1572fc3c20e61012d7c9b52ed029"


def materialize_inference_repo() -> None:
    """Copy support pack repo + weights to REPO_DIR (Cell 3 logic)."""
    artifacts = config.PRIMARY_ARTIFACT_DIR
    repo_dir = config.REPO_DIR

    # Copy repo
    copy_or_extract_tree(
        artifacts / "repo",
        artifacts / "repo.zip",
        repo_dir,
    )

    # Copy or link weights
    weights_src = artifacts / "weights"
    weights_zip = artifacts / "weights.zip"
    weights_dst = repo_dir / "weights"
    if weights_src.exists() and weights_src.is_dir():
        link_or_copy_tree(weights_src, weights_dst)
    elif weights_zip.exists() and weights_zip.is_file():
        remove_path(weights_dst)
        weights_dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(weights_zip) as zf:
            zf.extractall(weights_dst)
    else:
        raise FileNotFoundError(f"Missing weights tree or zip under {artifacts}")

    # Verify required files exist
    required = [
        repo_dir / "scripts" / "predict_unet_transformer.py",
        repo_dir / config.WEIGHTS_RELATIVE,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Materialized inference repo is incomplete:\n" + "\n".join(missing)
        )
    print(f"Inference repo: {repo_dir}")
    print(f"Weights: {repo_dir / config.WEIGHTS_RELATIVE}")


def verify_support_repo_integrity() -> None:
    """Verify SHA256 of all Python files in the support repo (Cell 3 logic)."""
    repo_dir = config.REPO_DIR

    materialized_paths = {
        path.relative_to(repo_dir).as_posix(): path
        for path in repo_dir.rglob("*.py")
    }
    actual_names = set(materialized_paths)
    expected_names = set(SUPPORT_EXPECTED_SHA256)

    if actual_names != expected_names:
        raise RuntimeError({
            "support_repo_python_files_missing": sorted(expected_names - actual_names),
            "support_repo_python_files_extra": sorted(actual_names - expected_names),
        })

    actual_sha256 = {
        relative: sha256_file(materialized_paths[relative])
        for relative in sorted(materialized_paths)
    }
    if actual_sha256 != SUPPORT_EXPECTED_SHA256:
        mismatches = {
            relative: {
                "expected": SUPPORT_EXPECTED_SHA256[relative],
                "actual": actual_sha256[relative],
            }
            for relative in sorted(SUPPORT_EXPECTED_SHA256)
            if actual_sha256[relative] != SUPPORT_EXPECTED_SHA256[relative]
        }
        raise RuntimeError({"support_repo_python_checksum_mismatch": mismatches})

    # Verify manifest SHA256
    manifest_bytes = "".join(
        f"{actual_sha256[relative]}  {relative}\n"
        for relative in sorted(actual_sha256)
    ).encode("utf-8")
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_sha256 != SUPPORT_EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Support repo manifest checksum mismatch: "
            f"expected {SUPPORT_EXPECTED_MANIFEST_SHA256}, "
            f"got {actual_manifest_sha256}"
        )

    print(f"Support repo Python manifest SHA256: {actual_manifest_sha256}")


def verify_primary_checkpoint() -> None:
    """Verify SHA256 of the primary model checkpoint."""
    primary_path = config.REPO_DIR / config.WEIGHTS_RELATIVE
    actual = sha256_file(primary_path)
    expected = config.EXPECTED_SHA256["primary"]
    if actual != expected:
        raise RuntimeError(
            f"Primary model checksum mismatch: expected {expected}, got {actual}"
        )
    print(f"Primary checkpoint SHA256: {actual}")


def verify_deepcenter_checkpoint() -> None:
    """Verify SHA256 of the DeepCenter checkpoint."""
    dc_path = Path(os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"])
    if not dc_path.is_file():
        raise FileNotFoundError(f"DeepCenter checkpoint not found: {dc_path}")
    actual = sha256_file(dc_path)
    expected = config.EXPECTED_SHA256["deepcenter"]
    if actual != expected:
        raise RuntimeError(
            f"DeepCenter checkpoint checksum mismatch: expected {expected}, got {actual}"
        )
    print(f"DeepCenter checkpoint SHA256: {actual}")


def setup_secondary_seed() -> None:
    """Set up secondary seed weights (Cell 3 logic)."""
    secondary_root = config.WORKING_DIR / "secondary_seed_weights"
    copy_or_extract_tree(
        config.SECONDARY_ARTIFACT_DIR / "weights",
        config.SECONDARY_ARTIFACT_DIR / "weights.zip",
        secondary_root,
    )
    secondary_weights = (
        secondary_root / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
    )
    secondary_config = secondary_weights.parent / "config.json"

    for required_path in (secondary_weights, secondary_config):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing secondary model file: {required_path}")

    # Verify SHA256
    actual = sha256_file(secondary_weights)
    expected = config.EXPECTED_SHA256["secondary"]
    if actual != expected:
        raise RuntimeError(
            f"Secondary model checksum mismatch: expected {expected}, got {actual}"
        )

    # Update env var to point to materialized path
    os.environ["BIOHUB_SECONDARY_WEIGHTS"] = str(secondary_weights)
    print(f"Secondary weight: {secondary_weights}")
    print(f"Secondary SHA256: {actual}")


def setup_all() -> None:
    """Complete setup: materialize repo, verify integrity, set up weights."""
    print("=" * 60)
    print("STEP 1: Materializing inference repo")
    print("=" * 60)
    materialize_inference_repo()

    print("\n" + "=" * 60)
    print("STEP 2: Verifying support repo integrity")
    print("=" * 60)
    verify_support_repo_integrity()

    print("\n" + "=" * 60)
    print("STEP 3: Verifying primary checkpoint")
    print("=" * 60)
    verify_primary_checkpoint()

    print("\n" + "=" * 60)
    print("STEP 4: Verifying DeepCenter checkpoint")
    print("=" * 60)
    verify_deepcenter_checkpoint()

    print("\n" + "=" * 60)
    print("STEP 5: Setting up secondary seed")
    print("=" * 60)
    setup_secondary_seed()

    # Write integrity receipt
    receipt = {
        "status": "hpc_reproduction_integrity_verified",
        "verified_before_dynamic_source_patch": True,
        "checkpoint_sha256": {
            "primary": config.EXPECTED_SHA256["primary"],
            "secondary": config.EXPECTED_SHA256["secondary"],
            "deepcenter": config.EXPECTED_SHA256["deepcenter"],
        },
    }
    receipt_path = config.WORKING_DIR / "reproduction_integrity.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nIntegrity receipt: {receipt_path}")
    print("\n✓ All artifacts verified and ready for inference.")
