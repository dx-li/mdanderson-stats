"""Capture native MOM normal equations; shares the checksum-verified child tracer."""

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
    default=Path(__file__).resolve().parents[1] / "tests/fixtures/wfmm-native-mom.json",
)
args = parser.parse_args()
binary = args.directory.resolve() / "wfmm1"
if hashlib.sha256(binary.read_bytes()).hexdigest() != SHA:
    raise ValueError("native executable checksum mismatch")
x = np.ones((8, 1))
z = np.repeat(np.eye(4), 2, axis=0)
strata = np.column_stack((np.r_[np.ones(3), np.zeros(5)], np.r_[np.zeros(3), np.ones(5)]))
unbalanced = np.eye(3)[[0, 0, 1, 1, 1, 2, 2, 2]]
fixed = np.column_stack((x, np.arange(8) / 8))
models = [
    ("identity", dict(X=x, C=x), Y, {}),
    ("strata", dict(X=x, C=strata), Y, {}),
    ("random_intercept", dict(X=x, C=x, Z=z, m=np.array([4.0])), Y, {}),
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
    ),
    ("fixed_trend", dict(X=fixed, C=x), Y, {}),
    ("trend_strata", dict(X=fixed, C=strata), Y, {}),
    ("unbalanced", dict(X=x, C=x, Z=unbalanced, m=np.array([3.0])), Y, {}),
    ("trend_unbalanced", dict(X=fixed, C=x, Z=unbalanced, m=np.array([3.0])), Y, {}),
    (
        "crossed",
        dict(
            X=x,
            C=x,
            Z=np.column_stack((unbalanced, np.tile(np.eye(2), (4, 1)))),
            m=np.array([3.0, 2.0]),
        ),
        Y,
        {},
    ),
    ("small_scale", dict(X=x, C=strata), 0.001 * Y, {}),
    ("large_scale", dict(X=x, C=strata), 1000 * Y, {}),
    ("short_iteration", dict(X=x, C=strata), Y, {"omega_MOM_maxiter": 2.0}),
]
models.append(
    (
        "mixed_residual_floor",
        dict(X=fixed, C=strata, Z=unbalanced, m=np.array([3.0])),
        np.array(
            [
                [-0.24374426665085194],
                [0.3762652959266094],
                [1.0140009549176336],
                [0.7701054285370403],
                [0.33815882424615606],
                [-0.1009190050004883],
                [-0.5708302957252598],
                [-0.9638034470143039],
            ]
        ).repeat(4, axis=1),
        {},
    )
)
models.extend(
    [
        (
            "signed_stop",
            dict(X=fixed, C=strata, Z=unbalanced, m=np.array([3.0])),
            np.array(
                [
                    -0.36053133884973027,
                    -0.3760821029081541,
                    -0.7240522709015332,
                    -0.08859563087320474,
                    -0.009421516493737068,
                    -0.35888679748627694,
                    -1.198018648085836,
                    1.0648291029374364,
                ]
            )[:, None].repeat(4, axis=1),
            {},
        ),
        (
            "three_residual_strata",
            dict(X=fixed, C=np.eye(3)[[0, 0, 1, 1, 1, 2, 2, 2]], Z=z, m=np.array([4.0])),
            Y,
            {},
        ),
        (
            "two_random_residual_strata",
            dict(
                X=x,
                C=strata,
                Z=np.column_stack((z, np.repeat(np.eye(2), 4, axis=0))),
                m=np.array([4.0, 2.0]),
            ),
            Y,
            {},
        ),
    ]
)
models.extend(
    [
        ("raised_floor", dict(X=fixed, C=x, Z=unbalanced, m=np.array([3.0])), Y, {"minVC": 1e-5}),
        (
            "raised_zero_threshold",
            dict(X=fixed, C=x, Z=unbalanced, m=np.array([3.0])),
            0.01 * Y,
            {"VC0_thresh": 0.001, "omega_MOM_convcrit": 1e-10},
        ),
        (
            "equal_floor_threshold",
            dict(X=fixed, C=x, Z=unbalanced, m=np.array([3.0])),
            Y,
            {"minVC": 1e-5, "VC0_thresh": 1e-5},
        ),
    ]
)

with tempfile.TemporaryDirectory(prefix="wfmm-mom-reference-") as temporary:
    cases = [
        trace(
            binary, Path(temporary), model, name, mode="mom", coefficients=y, mcmc_settings=settings
        )
        | {"settings": settings}
        for name, model, y, settings in models
    ]
args.output.write_text(
    json.dumps(dict(executable_sha256=SHA, cases=cases), indent=2, allow_nan=False) + "\n"
)
print(len(cases), "designs", sum(len(c["traces"]) for c in cases), "MOM systems")
