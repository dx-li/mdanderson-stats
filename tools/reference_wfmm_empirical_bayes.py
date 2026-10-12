"""Trace native WFMM empirical-Bayes input/output on synthetic coefficients."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from reference_wfmm_initialization import SHA, Y, trace

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--directory", required=True, type=Path)
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "tests/fixtures/wfmm-native-eb.json",
)
args = parser.parse_args()
binary = args.directory.resolve() / "wfmm1"
if hashlib.sha256(binary.read_bytes()).hexdigest() != SHA:
    raise ValueError("native executable checksum mismatch")
wave = dict(
    transformtype="wavelet", wavelet="db1", boundary="periodic", nlevels=2.0, extended_mode=1.0
)
models = [
    ("identity", {}, {}, Y),
    ("odds_cap", {"maxO": 2.0}, {}, Y),
    ("pi_floor", {"minp": 0.3}, {}, Y - Y.mean(axis=0)),
    ("large_signal", {}, {}, Y + 100),
    ("small_floor", {"minT": 0.01}, {}, Y - Y.mean(axis=0)),
    ("zero_mean", {}, {}, Y - Y.mean(axis=0)),
    ("minT", {"minT": 50.0}, {}, Y),
    ("wavelet", {}, wave, np.tile(Y, (1, 4))),
    ("coarse", {"nj_nosmooth": 2.0}, wave, np.tile(Y, (1, 4))),
    ("coarse_settings", {"nj_nosmooth": 1.0, "bigT": 30.0, "minT": 2.0}, wave, np.tile(Y, (1, 4))),
]
with tempfile.TemporaryDirectory(prefix="wfmm-eb-reference-") as temporary:
    cases = []
    for name, settings, basis, y in models:
        case = trace(
            binary,
            Path(temporary),
            dict(X=np.ones((8, 1)), C=np.ones((8, 1))),
            name,
            mode="eb",
            coefficients=y,
            mcmc_settings=settings,
            basis_settings=basis,
        )
        case.update(settings=settings, partition_sizes=[4, 4, 8] if basis else [4])
        cases.append(case)
args.output.write_text(
    json.dumps(dict(executable_sha256=SHA, cases=cases), indent=2, allow_nan=False) + "\n"
)
print(len(cases), "designs", sum(len(c["traces"]) for c in cases), "EB calls")
