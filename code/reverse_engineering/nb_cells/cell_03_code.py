import re

os.environ.setdefault("POLARS_PREFER_PKG", "32")

PACKAGE_SPECS = {
    "tracksdata": ("tracksdata", "tracksdata"),
    "zarr": ("zarr", "zarr>=3.0.10,<4"),
    "pyscipopt": ("pyscipopt", "pyscipopt"),
    "geff": ("geff", "geff>=1.1.3.1.1"),
    "geff_spec": ("geff_spec", "geff-spec<1.2"),
    "ilpy": ("ilpy", "ilpy>=0.5.1"),
    "polars": ("polars", "polars>=1.36"),
    "blosc2": ("blosc2", "blosc2"),
    "dask": ("dask", "dask"),
    "imagecodecs": ("imagecodecs", "imagecodecs"),
    "skimage": ("skimage", "scikit-image>=0.24"),
    "pyarrow": ("pyarrow", "pyarrow"),
    "rustworkx": ("rustworkx", "rustworkx>=0.17.1"),
    "sqlalchemy": ("sqlalchemy", "sqlalchemy>=2"),
    "numcodecs": ("numcodecs", "numcodecs>=0.13,<0.16"),
    "donfig": ("donfig", "donfig>=0.8"),
    "google_crc32c": ("google_crc32c", "google-crc32c>=1.5"),
    "bidict": ("bidict", "bidict>=0.23.1"),
    "psygnal": ("psygnal", "psygnal>=0.14"),
    "rich": ("rich", "rich"),
    "networkx": ("networkx", "networkx>=3.2.1"),
    "pydantic": ("pydantic", "pydantic>=2.11"),
    "pydantic_core": ("pydantic_core", "pydantic-core"),
    "annotated_types": ("annotated_types", "annotated-types"),
    "typing_extensions": ("typing_extensions", "typing-extensions>=4.13"),
    "typing_inspection": ("typing_inspection", "typing-inspection"),
    "markdown_it": ("markdown_it", "markdown-it-py"),
    "pygments": ("pygments", "pygments"),
    "click": ("click", "click"),
    "cloudpickle": ("cloudpickle", "cloudpickle"),
    "fsspec": ("fsspec", "fsspec"),
    "partd": ("partd", "partd"),
    "locket": ("locket", "locket"),
    "toolz": ("toolz", "toolz"),
    "yaml": ("yaml", "pyyaml"),
    "ndindex": ("ndindex", "ndindex"),
    "msgpack": ("msgpack", "msgpack"),
    "numexpr": ("numexpr", "numexpr"),
    "deprecated": ("deprecated", "deprecated"),
    "wrapt": ("wrapt", "wrapt"),
    "imageio": ("imageio", "imageio"),
    "PIL": ("PIL", "pillow"),
    "tifffile": ("tifffile", "tifffile"),
    "lazy_loader": ("lazy_loader", "lazy-loader"),
    "tqdm": ("tqdm", "tqdm"),
}
EXTRA_SPECS_BY_NAME = {
    "tracksdata": ["bidict>=0.23.1", "psygnal>=0.14", "rich"],
    "zarr": ["donfig>=0.8", "google-crc32c>=1.5", "numcodecs>=0.13,<0.16"],
    "geff": ["geff-spec<1.2", "networkx>=3.2.1", "pydantic>=2.11", "numcodecs>=0.13,<0.16"],
    "geff_spec": ["pydantic>=2.11", "annotated-types", "pydantic-core", "typing-inspection"],
    "polars": ["polars-runtime-32"],
    "dask": ["click", "cloudpickle", "fsspec", "partd", "pyyaml", "toolz"],
    "partd": ["locket"],
    "blosc2": ["ndindex", "msgpack", "numexpr"],
    "numcodecs": ["deprecated", "msgpack", "wrapt"],
    "rich": ["markdown-it-py", "pygments"],
    "pydantic": ["annotated-types", "pydantic-core", "typing-extensions>=4.13", "typing-inspection"],
    "skimage": ["imageio", "pillow", "tifffile", "lazy-loader", "networkx"],
}
PIP_DEPENDENCIES = [spec for _, spec in PACKAGE_SPECS.values()]
REQUIRED_MODULES = {name: module for name, (module, _) in PACKAGE_SPECS.items() if module}
FALLBACK_ARTIFACT_SLUGS = ["biohub-tracking-support-pack-v1"]



ALLOW_PIP_INSTALL = os.environ.get("BIOHUB_ALLOW_PIP_INSTALL", "0") != "0"


def module_missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


def has_model_artifact(path: Path) -> bool:
    has_repo_dir = (path / "repo").exists()
    has_weights_dir = (path / "weights" / METHOD / "split_0" / "edge_predictor_best.pth").exists()
    has_repo_zip = (path / "repo.zip").exists()
    has_weights_zip = (path / "weights.zip").exists()
    return (has_repo_dir and has_weights_dir) or (has_repo_zip and has_weights_zip)


def artifact_manifest(path: Path) -> dict:
    manifest = path / "ARTIFACT_MANIFEST.json"
    if not manifest.exists():
        return {}
    try:
        return json.loads(manifest.read_text())
    except Exception:
        return {}


def artifact_matches_target(path: Path) -> bool:
    if ALLOW_ARTIFACT_FALLBACK:
        return True
    manifest = artifact_manifest(path)
    artifact_name = str(manifest.get("artifact_name", ""))
    path_text = str(path)
    return TARGET_ARTIFACT_SLUG in {artifact_name, path.name} or TARGET_ARTIFACT_SLUG in path_text


def candidate_roots_for_slug(slug: str) -> list[Path]:
    return [
        Path(f"/kaggle/input/datasets/pilkwang/{slug}"),
        Path(f"/kaggle/input/{slug}"),
        Path(f"/kaggle/input/{slug}/{slug}"),
        Path(f"PublicNotebook/{slug}"),
    ]


def find_artifacts_root() -> Path:
    candidates: list[Path] = []
    for env_name in ["BIOHUB_MODEL_ARTIFACTS", "BIOHUB_ARTIFACTS"]:
        explicit = os.environ.get(env_name, "").strip()
        if explicit:
            candidates.append(Path(explicit))

    candidates.append(PRIMARY_ARTIFACT_MANIFEST.parent)
    candidates.extend(candidate_roots_for_slug(TARGET_ARTIFACT_SLUG))

    if ALLOW_ARTIFACT_FALLBACK:
        for slug in FALLBACK_ARTIFACT_SLUGS:
            candidates.extend(candidate_roots_for_slug(slug))

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for child in input_root.iterdir():
            if not child.is_dir():
                continue
            child_text = str(child)
            if TARGET_ARTIFACT_SLUG in child_text or ALLOW_ARTIFACT_FALLBACK:
                candidates.append(child)
                candidates.append(child / child.name)
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        candidates.append(grandchild)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if has_model_artifact(candidate) and artifact_matches_target(candidate):
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(
        "Could not find the required model artifact. "
        f"Expected slug: {TARGET_ARTIFACT_SLUG}\n"
        "Attach the newly uploaded support dataset, or set BIOHUB_MODEL_ARTIFACTS.\n"
        "To debug with an older artifact, set BIOHUB_ALLOW_ARTIFACT_FALLBACK=1.\n"
        "Checked:\n" + checked
    )


def _has_package_file(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    patterns = ("*.whl", "*.tar.gz", "*.zip")
    return any(any(path.glob(pattern)) for pattern in patterns)


def find_offline_package_dirs(artifacts: Path) -> list[Path]:
    candidates: list[Path] = [
        artifacts / "wheels",
        artifacts,
        Path("/kaggle/working"),
        Path("/kaggle/working/wheels"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for child in input_root.iterdir():
            if child.is_dir():
                candidates.extend([child / "wheels", child])
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        candidates.extend([grandchild / "wheels", grandchild])

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_package_file(candidate):
            out.append(candidate)
    return out


def purge_imported_modules(package_names: list[str]) -> None:
    roots = {"tracksdata"}
    for name in package_names:
        if name in PACKAGE_SPECS:
            module = PACKAGE_SPECS[name][0]
            roots.add(module.split(".")[0])
        if name == "polars":
            roots.add("polars")
    for root in roots:
        for module_name in list(sys.modules):
            if module_name == root or module_name.startswith(root + "."):
                sys.modules.pop(module_name, None)


def polars_runtime_ready() -> bool:
    try:
        import polars as _pl
        from polars._plr import PySeries as _PySeries

        _ = _PySeries
        return hasattr(_pl, "Float16") and _pl.Series([-999999.0], dtype=_pl.Float64).dtype == _pl.Float64
    except Exception:
        return False


def packages_requiring_refresh() -> list[str]:
    refresh: list[str] = []
    if not module_missing("polars") and not polars_runtime_ready():
        refresh.append("polars")

    if not module_missing("zarr"):
        try:
            import zarr as _zarr
            version_text = str(getattr(_zarr, "__version__", "0"))
            major = int(version_text.split(".", 1)[0])
            if major < 3:
                refresh.append("zarr")
        except Exception:
            refresh.append("zarr")
    return refresh


def dependency_specs_for(missing: list[str]) -> list[str]:
    specs: list[str] = []
    seen: set[str] = set()

    def add(spec: str) -> None:
        key = spec.lower()
        if key not in seen:
            seen.add(key)
            specs.append(spec)

    for name in missing:
        if name in PACKAGE_SPECS:
            add(PACKAGE_SPECS[name][1])
        for spec in EXTRA_SPECS_BY_NAME.get(name, []):
            add(spec)
    return specs


def import_failures() -> dict[str, str]:
    failures: dict[str, str] = {}
    for name, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    return failures


def missing_names_from_failures(failures: dict[str, str]) -> list[str]:
    names: list[str] = []
    module_to_name = {module: name for name, module in REQUIRED_MODULES.items()}
    for message in failures.values():
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
        if match:
            module = match.group(1).split(".")[0]
        else:
            match = re.search(r"module ['\"]([^'\"]+)['\"] has no attribute", message)
            if not match:
                continue
            module = match.group(1).split(".")[0]
        name = module_to_name.get(module)
        if name and name not in names:
            names.append(name)
    return names


def install_missing_dependencies(missing: list[str], artifacts: Path) -> None:
    specs = dependency_specs_for(missing)
    force_reinstall = bool({"polars", "zarr"} & set(missing))
    if not specs:
        return

    package_dirs = find_offline_package_dirs(artifacts)
    if package_dirs:
        offline_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
        if force_reinstall:
            offline_cmd.append("--force-reinstall")
        for package_dir in package_dirs:
            offline_cmd.extend(["--find-links", str(package_dir)])
        offline_cmd.extend(specs)
        print("Installing missing packages from offline package dirs:", missing)
        print("Dependency resolver is disabled with --no-deps to avoid replacing Kaggle numpy/scipy in a live kernel.")
        print("Offline package dirs:", [str(path) for path in package_dirs])
        result = subprocess.run(offline_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            purge_imported_modules(missing)
            print("Offline dependency install succeeded.")
            return
        print("Offline dependency install failed. Last pip output:")
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])

    if ALLOW_PIP_INSTALL:
        online_cmd = [sys.executable, "-m", "pip", "install", "--no-deps"]
        if force_reinstall:
            online_cmd.append("--force-reinstall")
        online_cmd.extend(specs)
        print("Installing missing packages from PyPI:", missing)
        result = subprocess.run(online_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            purge_imported_modules(missing)
            print("PyPI dependency install succeeded.")
            return
        print("PyPI dependency install failed. Last pip output:")
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])

    command = "pip install tracksdata zarr>=3.0.10,<4 pyscipopt geff geff-spec ilpy polars blosc2 dask imagecodecs pyarrow rustworkx sqlalchemy donfig numcodecs"
    raise ImportError(
        "Missing required packages or dependency wheels: " + ", ".join(missing) + "\n"
        "Attach the support dataset with offline wheels. If supplying Kaggle dependency input instead, use:\n"
        + command + "\n"
        "Do not quote zarr>=3.0.10,<4 in Kaggle dependency input."
    )


def ensure_dependencies(artifacts: Path) -> None:
    for _ in range(5):
        refresh = packages_requiring_refresh()
        if refresh:
            install_missing_dependencies(refresh, artifacts)
            continue

        missing = [pkg for pkg, module in REQUIRED_MODULES.items() if module_missing(module)]
        if missing:
            install_missing_dependencies(missing, artifacts)
            continue

        failures = import_failures()
        if not failures:
            print("Required graph/Zarr/ILP packages import successfully.")
            return

        missing_from_import = missing_names_from_failures(failures)
        if missing_from_import:
            install_missing_dependencies(missing_from_import, artifacts)
            continue

        raise ImportError(
            "Required packages are present but failed to import. "
            "This may indicate a binary dependency mismatch in the live notebook kernel. "
            "Keep Kaggle dependency input empty and attach the wheels artifact.\n"
            + json.dumps(failures, indent=2)
        )

    failures = import_failures()
    raise ImportError(
        "Dependency recovery did not converge after repeated offline installs. "
        "The attached support artifact may be missing wheels.\n"
        + json.dumps(failures, indent=2)
    )


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def copy_or_extract_tree(src_dir: Path, src_zip: Path, dst: Path) -> None:
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
    remove_path(dst)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except Exception:
        shutil.copytree(src, dst)


def materialize_inference_repo(artifacts: Path) -> None:
    copy_or_extract_tree(artifacts / "repo", artifacts / "repo.zip", REPO_DIR)

    weights_src = artifacts / "weights"
    weights_zip = artifacts / "weights.zip"
    weights_dst = REPO_DIR / "weights"
    if weights_src.exists() and weights_src.is_dir():
        link_or_copy_tree(weights_src, weights_dst)
    elif weights_zip.exists() and weights_zip.is_file():
        remove_path(weights_dst)
        weights_dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(weights_zip) as zf:
            zf.extractall(weights_dst)
    else:
        raise FileNotFoundError(f"Missing weights tree or zip under {artifacts}")

    required = [
        REPO_DIR / "scripts" / "predict_unet_transformer.py",
        REPO_DIR / WEIGHTS_RELATIVE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Materialized inference repo is incomplete:\n" + "\n".join(missing))
    print("Inference repo:", REPO_DIR)
    print("Weights:", REPO_DIR / WEIGHTS_RELATIVE)


ARTIFACTS = find_artifacts_root()
print("ARTIFACTS:", ARTIFACTS)
print("Has offline wheels:", (ARTIFACTS / "wheels").exists())
manifest_info = artifact_manifest(ARTIFACTS)
if manifest_info:
    print("Artifact name:", manifest_info.get("artifact_name"))
    print("Weight sha256:", manifest_info.get("model", {}).get("weight_sha256"))
    print("Weight path:", manifest_info.get("model", {}).get("weight_path"))
    _expected_primary_sha256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
    _actual_primary_sha256 = str(manifest_info.get("model", {}).get("weight_sha256", ""))
    if _actual_primary_sha256 != _expected_primary_sha256:
        raise RuntimeError(
            "Primary model checksum mismatch: "
            f"expected {_expected_primary_sha256}, got {_actual_primary_sha256 or 'missing'}"
        )

ensure_dependencies(ARTIFACTS)
materialize_inference_repo(ARTIFACTS)




import hashlib as _integrity_hashlib

_support_expected_sha256 = {
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
    "src/biohub_tracking/models/temporal_unet.py": "d809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac"
}
_support_expected_manifest_sha256 = "978b626d1fd1e7397435a437dfe68691defe1572fc3c20e61012d7c9b52ed029"
_primary_expected_sha256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
_deepcenter_expected_sha256 = "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0"  


def _integrity_sha256_file(path: Path) -> str:
    digest = _integrity_hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_support_materialized_paths = {
    path.relative_to(REPO_DIR).as_posix(): path
    for path in REPO_DIR.rglob("*.py")
}
_support_actual_names = set(_support_materialized_paths)
_support_expected_names = set(_support_expected_sha256)
if _support_actual_names != _support_expected_names:
    raise RuntimeError({
        "support_repo_python_files_missing": sorted(
            _support_expected_names - _support_actual_names
        ),
        "support_repo_python_files_extra": sorted(
            _support_actual_names - _support_expected_names
        ),
    })
_support_actual_sha256 = {
    relative: _integrity_sha256_file(_support_materialized_paths[relative])
    for relative in sorted(_support_materialized_paths)
}
if _support_actual_sha256 != _support_expected_sha256:
    raise RuntimeError({
        "support_repo_python_checksum_mismatch": {
            relative: {
                "expected": _support_expected_sha256[relative],
                "actual": _support_actual_sha256[relative],
            }
            for relative in sorted(_support_expected_sha256)
            if _support_actual_sha256[relative]
            != _support_expected_sha256[relative]
        }
    })
_support_manifest_bytes = "".join(
    f"{_support_actual_sha256[relative]}  {relative}\n"
    for relative in sorted(_support_actual_sha256)
).encode("utf-8")
_support_actual_manifest_sha256 = _integrity_hashlib.sha256(
    _support_manifest_bytes
).hexdigest()
if _support_actual_manifest_sha256 != _support_expected_manifest_sha256:
    raise RuntimeError(
        "Support repo manifest checksum mismatch: "
        f"expected {_support_expected_manifest_sha256}, "
        f"got {_support_actual_manifest_sha256}"
    )

_primary_materialized_path = REPO_DIR / WEIGHTS_RELATIVE
_primary_actual_sha256 = _integrity_sha256_file(_primary_materialized_path)
if _primary_actual_sha256 != _primary_expected_sha256:
    raise RuntimeError(
        "Materialized primary model checksum mismatch: "
        f"expected {_primary_expected_sha256}, got {_primary_actual_sha256}"
    )

_deepcenter_candidate_strings = [
    os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT", "").strip(),
    "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/"
    "full_frame_center/best.pt",
    "/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/"
    "weights/full_frame_center/best.pt",
]
_deepcenter_candidates = []
for _candidate_string in _deepcenter_candidate_strings:
    if not _candidate_string:
        continue
    _candidate_path = Path(_candidate_string)
    if _candidate_path not in _deepcenter_candidates:
        _deepcenter_candidates.append(_candidate_path)
_deepcenter_materialized_path = next(
    (path for path in _deepcenter_candidates if path.is_file()),
    None,
)
if _deepcenter_materialized_path is None:
    raise FileNotFoundError({
        "missing_deepcenter_checkpoint": [str(path) for path in _deepcenter_candidates]
    })
_deepcenter_actual_sha256 = _integrity_sha256_file(
    _deepcenter_materialized_path
)
if _deepcenter_actual_sha256 != _deepcenter_expected_sha256:
    raise RuntimeError(
        "DeepCenter checkpoint checksum mismatch: "
        f"expected {_deepcenter_expected_sha256}, "
        f"got {_deepcenter_actual_sha256}"
    )
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(
    _deepcenter_materialized_path
)

print("Support repo Python manifest SHA256:", _support_actual_manifest_sha256)
print("Primary materialized SHA256:", _primary_actual_sha256)
print("DeepCenter materialized SHA256:", _deepcenter_actual_sha256)



import hashlib as _hashlib

_secondary_manifest_explicit = Path(os.environ.get(
    "BIOHUB_SECONDARY_ARTIFACT_MANIFEST",
    "/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/ARTIFACT_MANIFEST.json",
))
_secondary_expected_sha256 = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
_secondary_slug = "biohub-temporal-unet3d-seed314159-v1"


def _find_secondary_artifact_root() -> tuple[Path, dict]:
    candidates = [
        _secondary_manifest_explicit,
        Path(f"/kaggle/input/{_secondary_slug}/ARTIFACT_MANIFEST.json"),
        Path(f"/kaggle/input/datasets/pilkwang/{_secondary_slug}/ARTIFACT_MANIFEST.json"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.rglob("ARTIFACT_MANIFEST.json"))

    seen = set()
    for manifest_path in candidates:
        manifest_path = manifest_path.expanduser()
        if manifest_path in seen or not manifest_path.is_file():
            continue
        seen.add(manifest_path)
        try:
            info = json.loads(manifest_path.read_text())
        except Exception:
            continue
        sha256 = str(info.get("model", {}).get("weight_sha256", ""))
        if sha256 == _secondary_expected_sha256:
            return manifest_path.parent, info
    raise FileNotFoundError(
        "Could not find the independent-seed artifact with weight SHA256 "
        + _secondary_expected_sha256
    )


SECONDARY_ARTIFACTS, secondary_manifest_info = _find_secondary_artifact_root()
SECONDARY_WEIGHTS_ROOT = WORKING_DIR / "secondary_seed_weights"
copy_or_extract_tree(
    SECONDARY_ARTIFACTS / "weights",
    SECONDARY_ARTIFACTS / "weights.zip",
    SECONDARY_WEIGHTS_ROOT,
)
SECONDARY_WEIGHTS_PATH = (
    SECONDARY_WEIGHTS_ROOT
    / "unet_transformer"
    / "split_0"
    / "edge_predictor_best.pth"
)
SECONDARY_CONFIG_PATH = SECONDARY_WEIGHTS_PATH.parent / "config.json"
for _required_secondary_path in (SECONDARY_WEIGHTS_PATH, SECONDARY_CONFIG_PATH):
    if not _required_secondary_path.is_file():
        raise FileNotFoundError(f"Missing secondary model file: {_required_secondary_path}")


def _sha256_file(path: Path) -> str:
    digest = _hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_secondary_actual_sha256 = _sha256_file(SECONDARY_WEIGHTS_PATH)
if _secondary_actual_sha256 != _secondary_expected_sha256:
    raise RuntimeError(
        "Secondary model checksum mismatch: "
        f"expected {_secondary_expected_sha256}, got {_secondary_actual_sha256}"
    )

os.environ["BIOHUB_SECONDARY_WEIGHTS"] = str(SECONDARY_WEIGHTS_PATH)
os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"] = "0.15"
print("Secondary artifact:", SECONDARY_ARTIFACTS)
print("Secondary weight:", SECONDARY_WEIGHTS_PATH)
print("Secondary SHA256:", _secondary_actual_sha256)
print("Secondary edge-logit weight:", os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"])

os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.80"  
os.environ["BIOHUB_SECONDARY_LINK_MODE"] = "low_margin_consensus"
os.environ["BIOHUB_SECONDARY_MIX_TEMPERATURE"] = "1"
os.environ["BIOHUB_SECONDARY_LOW_MARGIN_MAX"] = "0.35"
os.environ["BIOHUB_DUAL_SEED_EDGE_THRESHOLD"] = "0.48"

_runtime_integrity_receipt = {
    "status": "complete_label_free_runtime_integrity",
    "verified_before_dynamic_source_patch": True,
    "support_repo_python_file_count": len(_support_actual_sha256),
    "support_repo_python_sha256": _support_actual_sha256,
    "support_repo_python_manifest_sha256": _support_actual_manifest_sha256,
    "checkpoint_sha256": {
        "primary": _primary_actual_sha256,
        "secondary": _secondary_actual_sha256,
        "deepcenter": _deepcenter_actual_sha256,
    },
    "materialized_paths": {
        "primary": str(_primary_materialized_path),
        "secondary": str(SECONDARY_WEIGHTS_PATH),
        "deepcenter": str(_deepcenter_materialized_path),
    },
    "ground_truth_accessed": False,
}
_runtime_integrity_receipt_path = (
    WORKING_DIR / "bidirectional_production_runtime_integrity.json"
)
_runtime_integrity_receipt_path.write_text(
    json.dumps(_runtime_integrity_receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("Runtime integrity receipt:", _runtime_integrity_receipt_path)
