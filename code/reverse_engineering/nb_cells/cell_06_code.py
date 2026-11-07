
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

_guard_submission = Path("/kaggle/working/submission.csv")
_guard_columns = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
]
if not _guard_submission.is_file():
    raise FileNotFoundError(_guard_submission)
_guard_frame = pd.read_csv(_guard_submission)
if _guard_frame.empty or _guard_frame.columns.tolist() != _guard_columns:
    raise RuntimeError("Retention-guard submission schema changed")
if _guard_frame["id"].tolist() != list(range(len(_guard_frame))):
    raise RuntimeError("Retention-guard row IDs are not contiguous")
if set(_guard_frame["row_type"].unique()) != {"node", "edge"}:
    raise RuntimeError("Retention-guard row types changed")

_guard_datasets = sorted(_guard_frame["dataset"].astype(str).unique())
_guard_expected = sorted(
    path.name.removesuffix(".zarr")
    for path in TEST_DIR.iterdir()
    if path.name.endswith(".zarr")
)
if _guard_datasets != _guard_expected:
    raise RuntimeError({"expected": _guard_expected, "actual": _guard_datasets})

_guard_records = []
for _guard_path in sorted(Path("/kaggle/working").glob("retention_guard_*.jsonl")):
    for _guard_line in _guard_path.read_text().splitlines():
        if _guard_line.strip():
            _guard_records.append(json.loads(_guard_line))
if not _guard_records:
    raise RuntimeError("No frame-retention diagnostics were produced")

_guard_keys = [
    (str(row["dataset"]), int(row["frame"])) for row in _guard_records
]
if len(_guard_keys) != len(set(_guard_keys)):
    raise RuntimeError("Duplicate frame-retention diagnostics")
if sorted(set(movie for movie, _ in _guard_keys)) != _guard_expected:
    raise RuntimeError("Frame-retention diagnostics do not cover every movie")
for _guard_record in _guard_records:
    if (
        float(_guard_record["minimum_retention"])
        != 0.9
        or int(_guard_record["primary_candidates"]) < 0
        or int(_guard_record["blended_candidates"]) < 0
    ):
        raise RuntimeError("Frame-retention diagnostic contract changed")
    _guard_expected_use_primary = bool(
        int(_guard_record["primary_candidates"]) > 0
        and float(_guard_record["retention"])
        < 0.9
    )
    if bool(_guard_record["use_primary"]) != _guard_expected_use_primary:
        raise RuntimeError("Frame-retention decision is inconsistent")

_guard_topology = {}
for _guard_movie, _guard_group in _guard_frame.groupby("dataset", sort=True):
    _guard_nodes = _guard_group[_guard_group["row_type"].eq("node")]
    _guard_edges = _guard_group[_guard_group["row_type"].eq("edge")]
    if _guard_nodes.empty or _guard_nodes["t"].lt(0).any():
        raise RuntimeError(f"{_guard_movie}: invalid biological node time")
    if _guard_nodes[["z", "y", "x"]].lt(0).any().any():
        raise RuntimeError(f"{_guard_movie}: negative biological coordinate")
    _guard_node_time = dict(zip(
        _guard_nodes["node_id"].astype(int),
        _guard_nodes["t"].astype(int),
    ))
    _guard_incoming = Counter()
    _guard_outgoing = Counter()
    for _guard_edge in _guard_edges.itertuples():
        _guard_source = int(_guard_edge.source_id)
        _guard_target = int(_guard_edge.target_id)
        if (
            _guard_source not in _guard_node_time
            or _guard_target not in _guard_node_time
            or _guard_node_time[_guard_target]
            != _guard_node_time[_guard_source] + 1
        ):
            raise RuntimeError(f"{_guard_movie}: invalid lineage edge")
        _guard_incoming[_guard_target] += 1
        _guard_outgoing[_guard_source] += 1
    _guard_max_in = max(_guard_incoming.values(), default=0)
    _guard_max_out = max(_guard_outgoing.values(), default=0)
    if _guard_max_in > 1 or _guard_max_out > 2:
        raise RuntimeError(f"{_guard_movie}: invalid lineage degree")
    _guard_topology[_guard_movie] = {
        "nodes": int(len(_guard_nodes)),
        "edges": int(len(_guard_edges)),
        "max_indegree": int(_guard_max_in),
        "max_outdegree": int(_guard_max_out),
        "division_parents": int(sum(
            value == 2 for value in _guard_outgoing.values()
        )),
    }

_guard_by_movie = {}
for _guard_movie in _guard_expected:
    _guard_movie_records = [
        row for row in _guard_records if row["dataset"] == _guard_movie
    ]
    _guard_by_movie[_guard_movie] = {
        "frames": int(len(_guard_movie_records)),
        "fallback_frames": int(sum(
            bool(row["use_primary"]) for row in _guard_movie_records
        )),
        "minimum_retention": float(min(
            row["retention"] for row in _guard_movie_records
        )),
        "median_retention": float(pd.Series(
            [row["retention"] for row in _guard_movie_records]
        ).median()),
    }

_guard_digest = hashlib.sha256(_guard_submission.read_bytes()).hexdigest()
_guard_report = {
    "experiment": "harmonic_bidirectional_association_v1",
    "status": "clean_graph_audit_pass_candidate_unverified_quality",
    "parent_experiment": "paired_bidirectional_primary_weight020_vs_forward_v1",
    "method_attribution": "fixed-90 dual-seed baseline with harmonic mutual-support association fusion (rule from public CC0 notebook yusuketogashi/no-hack-biohub-cell-another-approch-3rd v18)",
    "source_kernel": "raykkretzschmar/biohub-bidirectional-primary-union13-diagnostic-v1",
    "source_notebook_sha256": "3e65ca691941949196bf417030ea84fccafe16baaec174b2a63540451bb937e8",
    "public_output_used": False,
    "metric_hack_used": False,
    "organizer_labels_used_for_configuration": False,
    "leaderboard_feedback_used_for_configuration": True,
    "configuration": {
        "minimum_candidate_retention": 0.9,
        "fallback_scope": "individual_frame",
        "detector_threshold": 0.96875,
        "secondary_detection_weight": 0.475,
        "secondary_edge_weight": 0.15,
        "bidirectional_primary_weight": 0.30,
        "secondary_link_mode": "low_margin_consensus",
        "secondary_low_margin_max": 0.35,
        "edge_candidate_threshold": 0.48,
        "ilp_appearance_weight": 0.0,
        "ilp_disappearance_weight": 1.5,
        "gap_close_um": 5.8,
        "deepcenter_gap_threshold": 0.25,
        "deepcenter_gap_confirm_min_span_um": 8.5,
    },
    "hardware": {
        "visible_gpu_count": int(torch.cuda.device_count()),
    },
    "diagnostics": {
        "rows": int(len(_guard_records)),
        "fallback_frames": int(sum(
            bool(row["use_primary"]) for row in _guard_records
        )),
        "by_movie": _guard_by_movie,
    },
    "submission": {
        "sha256": _guard_digest,
        "rows": int(len(_guard_frame)),
        "datasets": _guard_datasets,
    },
    "topology": _guard_topology,
    "quality_promotion": {
        "status": "candidate_unverified",
        "required_receipt": "bidirectional_blend_union13_receipt.json",
        "required_condition": "promote=true",
        "execute_push_submit": "FORBIDDEN_UNTIL_REQUIRED_CONDITION",
        "validated_receipt_sha256": None,
    },
}
Path("/kaggle/working/dual_seed_frame_retention_guard_report.json").write_text(
    json.dumps(_guard_report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(_guard_report, indent=2, sort_keys=True))
