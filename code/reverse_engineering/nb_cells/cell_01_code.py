
import json as _guard_json
import math as _guard_math
import os as _guard_os

_EXPECTED_NUMERIC = {
    "BIOHUB_DET_THRESHOLD": 0.965,
    "BIOHUB_ILP_APPEARANCE_WEIGHT": 0.0,
    "BIOHUB_ILP_DISAPPEARANCE_WEIGHT": 2,
    "BIOHUB_GAP_CLOSE_UM": 5.0,
    "BIOHUB_OUTPUT_MIN_TRACK_LEN": 6.0,
    "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": 0.15,
}

_EXPECTED_TEXT = {
    "BIOHUB_BIDIRECTIONAL_FUSION_MODE": "harmonic_probability",
    "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION": "0.90",
}

_drift = {}
for _key, _want in _EXPECTED_NUMERIC.items():
    _raw = _guard_os.environ.get(_key)
    if _raw is None:
        _drift[_key] = "missing"
        continue
    _got = float(_raw)
    if not _guard_math.isclose(_got, _want, rel_tol=0.0, abs_tol=1e-12):
        _drift[_key] = {"expected": _want, "actual": _got}

for _key, _want in _EXPECTED_TEXT.items():
    _got = _guard_os.environ.get(_key)
    if _got != _want:
        _drift[_key] = {"expected": _want, "actual": _got}

if _drift:
    raise RuntimeError(
        "Configuration drift detected: " + _guard_json.dumps(_drift, sort_keys=True)
    )

print("Configuration guard: PASS")
print("Baseline: fixed-90 dual-seed clean pipeline (public LB 0.913)")
print("Single model-level change: harmonic mutual-support association fusion")
print("Reverse-time association weight: 0.200")
