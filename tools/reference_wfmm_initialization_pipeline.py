"""Original complete initialization outputs on wholly synthetic study designs."""

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
    default=Path(__file__).resolve().parents[1]
    / "tests/fixtures/wfmm-native-initialization-pipeline.json",
)
args = parser.parse_args()
binary = args.directory.resolve() / "wfmm1"
if hashlib.sha256(binary.read_bytes()).hexdigest() != SHA:
    raise ValueError("native executable checksum mismatch")
x = np.ones((8, 1))
z = np.repeat(np.eye(4), 2, axis=0)
strata = np.column_stack((np.r_[np.ones(3), np.zeros(5)], np.r_[np.zeros(3), np.ones(5)]))
wave = dict(
    transformtype="wavelet", wavelet="db1", boundary="periodic", nlevels=2.0, extended_mode=1.0
)
models = [
    ("identity", dict(X=x, C=x), Y, {}, {}),
    ("strata", dict(X=x, C=strata), Y, {}, {}),
    ("random_intercept", dict(X=x, C=x, Z=z, m=np.array([4.0])), Y, {}, {}),
    (
        "two_random_levels",
        dict(
            X=x,
            C=x,
            Z=np.column_stack((z, np.repeat(np.eye(2), 4, axis=0))),
            m=np.array([4.0, 2.0]),
        ),
        Y,
        {},
        {},
    ),
    ("floor_freeze", dict(X=x, C=x, Z=z, m=np.array([4.0])), Y, {}, dict(maxO=2.0)),
    ("zero_signal", dict(X=x, C=x), np.zeros_like(Y), {}, {}),
    ("fixed_trend", dict(X=np.column_stack((x, np.arange(8) / 8)), C=x), Y, {}, {}),
    ("wavelet", dict(X=x, C=x), np.tile(Y, (1, 4)), wave, {}),
    ("coarse_wavelet", dict(X=x, C=x), np.tile(Y, (1, 4)), wave, dict(nj_nosmooth=2.0)),
]
with tempfile.TemporaryDirectory(prefix="wfmm-init-pipeline-reference-") as temporary:
    cases = []
    for name, model, y, basis, settings in models:
        case = trace(
            binary,
            Path(temporary),
            model,
            name,
            mode="bounds",
            coefficients=y,
            basis_settings=basis,
            mcmc_settings=settings,
            extra_outputs=(
                "D",
                "Vbetastar_ns",
                "pi_MLE",
                "tau_MLE",
                "alpha_MLE",
                "prior_omega_a",
                "prior_omega_b",
            ),
        )
        case.update(settings=settings, partition_sizes=[4, 4, 8] if basis else [4])
        cases.append(case)
args.output.write_text(
    json.dumps(dict(executable_sha256=SHA, cases=cases), indent=2, allow_nan=False) + "\n"
)
print(len(cases), "original full initialization studies")
