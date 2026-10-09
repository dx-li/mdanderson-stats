"""Regenerate synthetic variance-prior references with verified native WFMM1.

Run with the project Python and --archive pointing to the official Linux bundle.
No native libraries or executable files are included in the Python distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

ARCHIVE_SHA256 = "9bc5003271eebfac4a90e265d08c8f677d72252eccae5ff65c90bc25d5d309f9"
FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/wfmm-native-variance-prior.json"
Y = np.array(
    [
        [1, 4, 2, 3],
        [2, 2, 3, 1],
        [4, 3, 1, 5],
        [5, 6, 4, 2],
        [6, 3, 7, 4],
        [3, 1, 2, 6],
        [2, 5, 5, 3],
        [4, 2, 6, 2],
    ],
    dtype=float,
)


def references(archive: Path) -> dict:
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("native archive checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="wfmm-reference-") as temporary:
        root = Path(temporary)
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                name = Path(member.name).name
                if member.isfile() and (name == "wfmm1" or name.startswith("lib")):
                    stream = bundle.extractfile(member)
                    assert stream is not None
                    (root / name).write_bytes(stream.read())
        binary = root / "wfmm1"
        binary.chmod(0o755)
        environment = {**os.environ, "LD_LIBRARY_PATH": str(root), "OMP_NUM_THREADS": "1"}
        z = np.repeat(np.eye(4), 2, axis=0)
        multi_z = np.column_stack((z, np.repeat(np.eye(2), 4, axis=0)))
        y16 = np.column_stack(
            (Y, Y[:, ::-1] + np.arange(8)[:, None] / 5, Y * 0.7, Y[:, ::-1] * 1.3)
        )
        strata = np.column_stack((np.r_[np.ones(3), np.zeros(5)], np.r_[np.zeros(3), np.ones(5)]))
        cases = [
            ("identity", Y, np.ones((8, 1)), None, [], 0.1, {"transformtype": "none"}),
            ("small_delta", Y, np.ones((8, 1)), None, [], 0.0001, {"transformtype": "none"}),
            ("six_curves", Y[:6], np.ones((6, 1)), None, [], 0.2, {"transformtype": "none"}),
            ("residual_strata", Y, strata, None, [], 0.2, {"transformtype": "none"}),
            ("random_intercept", Y, np.ones((8, 1)), z, [4], 0.2, {"transformtype": "none"}),
            (
                "two_random_levels",
                Y,
                np.ones((8, 1)),
                multi_z,
                [4, 2],
                0.2,
                {"transformtype": "none"},
            ),
            (
                "haar_partitions",
                y16,
                np.ones((8, 1)),
                z,
                [4],
                0.2,
                {
                    "transformtype": "wavelet",
                    "wavelet": "db1",
                    "boundary": "periodic",
                    "nlevels": 2.0,
                    "extended_mode": 1.0,
                },
            ),
            (
                "identity_partitions",
                y16,
                np.ones((8, 1)),
                z,
                [4],
                0.2,
                {"transformtype": "none", "partitions": np.array([4.0, 4.0, 8.0])},
            ),
        ]
        output = []
        for name, y, c, random, counts, delta, basis in cases:
            model = {"X": np.ones((len(y), 1)), "C": c}
            if random is not None:
                model["Z"] = random
                model["m"] = np.asarray(counts, dtype=float)
            path = root / f"{name}.mat"
            savemat(
                path,
                {
                    "Y": y,
                    "model": model,
                    "basis_specs": {"nlevels": 1.0, **basis},
                    "MCMCspecs": {
                        "B": 8.0,
                        "burnin": 4.0,
                        "thin": 1.0,
                        "time_update": 1.0,
                        "nj_nosmooth": 0.0,
                        "delta_omega": delta,
                    },
                },
            )
            result = subprocess.run(
                [str(binary), str(path)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode:
                raise RuntimeError(f"native {name} failed: {result.stdout} {result.stderr}")
            native = loadmat(root / f"{name}_Init.mat")
            output.append(
                {
                    "case": name,
                    "delta_omega": delta,
                    "random_effect_counts": counts,
                    "residual_stratum_sizes": c.sum(axis=0).astype(int).tolist(),
                    "omega_MLE": native["omega_MLE"].tolist(),
                    "prior_omega_a": native["prior_omega_a"].tolist(),
                    "prior_omega_b": native["prior_omega_b"].tolist(),
                }
            )
        return {
            "archive_sha256": ARCHIVE_SHA256,
            "executable_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "native_version": "WFMM1 3.1.0, built Sep 14 2015",
            "inputs": "Wholly synthetic; generated by tools/reference_wfmm_variance_prior.py",
            "cases": output,
        }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = references(args.archive)
    if args.check:
        saved = json.loads(FIXTURE.read_text())
        assert {k: v for k, v in generated.items() if k != "cases"} == {
            k: v for k, v in saved.items() if k != "cases"
        }
        assert len(generated["cases"]) == len(saved["cases"])
        for actual, expected in zip(generated["cases"], saved["cases"], strict=True):
            for key, value in actual.items():
                if isinstance(value, (list, float)):
                    np.testing.assert_allclose(value, expected[key], rtol=1e-11, atol=1e-12)
                else:
                    assert value == expected[key]
        print("Verified all eight native WFMM variance-prior cases")
    else:
        FIXTURE.write_text(json.dumps(generated, indent=2) + "\n")
        print(f"Wrote {FIXTURE.name}")


if __name__ == "__main__":
    main()
