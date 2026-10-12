"""Read original conditional optimizer bounds, including invalid lower fallbacks."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from reference_wfmm_initialization import SHA, trace

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--directory", required=True, type=Path)
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "tests/fixtures/wfmm-native-bounds.json",
)
args = parser.parse_args()
binary = args.directory.resolve() / "wfmm1"
if hashlib.sha256(binary.read_bytes()).hexdigest() != SHA:
    raise ValueError("native executable checksum mismatch")
source = json.loads(
    (
        Path(__file__).resolve().parents[1] / "tests/fixtures/wfmm-native-initialization.json"
    ).read_text()
)
with tempfile.TemporaryDirectory(prefix="wfmm-bounds-reference-") as temporary:
    cases = [
        trace(
            binary,
            Path(temporary),
            {k: np.array(v) for k, v in c["model"].items()},
            c["case"],
            mode="bounds",
        )
        for c in source["cases"]
    ]
    mixed = next(c for c in source["cases"] if c["case"] == "random_intercept")
    cases.append(
        trace(
            binary,
            Path(temporary),
            {k: np.array(v) for k, v in mixed["model"].items()},
            "floor_freeze_odds_cap",
            mode="bounds",
            mcmc_settings={"maxO": 2.0},
        )
    )
args.output.write_text(
    json.dumps(dict(executable_sha256=SHA, cases=cases), indent=2, allow_nan=False) + "\n"
)
print(len(cases), "original optimizer/bound matrices")
