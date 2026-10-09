"""Regenerate synthetic compression/periodic Haar references with native WFMM1.

The official archive is a local research input, never part of distributions.
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
from reference_wfmm_variance_prior import ARCHIVE_SHA256, Y
from scipy.io import loadmat, savemat

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/wfmm-native-compression.json"


def references(archive: Path) -> dict:
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("native archive checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="wfmm-compression-reference-") as temporary:
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
        y16 = np.column_stack(
            (Y, Y[:, ::-1] + np.arange(8)[:, None] / 5, Y * 0.7, Y[:, ::-1] * 1.3)
        )
        rng = np.random.default_rng(841967)
        datasets = [("paperless16", y16, 2)]
        for length, levels in [(16, 1), (32, 2), (32, 3), (64, 4)]:
            curves = rng.normal(2, 1, (12, length))
            datasets.append((f"synthetic{length}_levels{levels}", curves, levels))
        output = []
        for name, curves, levels in datasets:
            runs = []
            full = None
            settings = [(1.0, 0), (0.5, 0), (0.9, 0), (0.95, 0), (0.99, 0)]
            settings += [(0.9, 1), (0.9, len(curves) // 2), (0.9, len(curves) - 1)]
            settings += [(1.0, len(curves))]
            if name == "paperless16":
                settings += [(0.5, len(curves) - 1)]  # Native empty-selection error.
            for index, (alpha, t) in enumerate(settings):
                path = root / f"{name}_{index}.mat"
                savemat(
                    path,
                    {
                        "Y": curves,
                        "model": {"X": np.ones((len(curves), 1)), "C": np.ones((len(curves), 1))},
                        "basis_specs": {
                            "transformtype": "wavelet",
                            "wavelet": "db1",
                            "boundary": "periodic",
                            "nlevels": float(levels),
                            "extended_mode": 1.0,
                            "alphawav": alpha,
                            "t": float(t),
                        },
                        "MCMCspecs": {
                            "B": 8.0,
                            "burnin": 4.0,
                            "thin": 1.0,
                            "time_update": 1.0,
                            "nj_nosmooth": 0.0,
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
                if result.returncode and "ColQty=0" in result.stderr:
                    runs.append({"alpha": alpha, "t": t, "error": "empty selection"})
                    continue
                if result.returncode:
                    raise RuntimeError(f"native {name} failed: {result.stdout} {result.stderr}")
                # Preserve two-dimensional arrays: simplify_cells squeezes Kstar=1.
                native = loadmat(root / f"{name}_{index}_Init.mat")
                specs = loadmat(root / f"{name}_{index}_Init.mat", simplify_cells=True)[
                    "basis_specs"
                ]
                d = native["D"]
                if index == 0:
                    full = d.tolist()
                indices = (
                    list(range(curves.shape[1]))
                    if alpha == 1
                    else np.atleast_1d(specs["DIndex"]).astype(int).tolist()
                )
                runs.append({"alpha": alpha, "t": t, "indices": indices, "D": d.tolist()})
            output.append(
                {
                    "case": name,
                    "Y": curves.tolist(),
                    "levels": levels,
                    "D_full": full,
                    "runs": runs,
                }
            )
        return {
            "archive_sha256": ARCHIVE_SHA256,
            "executable_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "native_version": "WFMM1 3.1.0, built Sep 14 2015",
            "inputs": "Wholly synthetic; tools/reference_wfmm_compression.py",
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
        for actual, expected in zip(generated["cases"], saved["cases"], strict=True):
            assert actual["case"] == expected["case"] and actual["levels"] == expected["levels"]
            np.testing.assert_array_equal(actual["Y"], expected["Y"])
            np.testing.assert_allclose(actual["D_full"], expected["D_full"], atol=1e-12)
            for run, reference in zip(actual["runs"], expected["runs"], strict=True):
                for key in ("alpha", "t", "indices", "error"):
                    assert run.get(key) == reference.get(key)
                if "D" in run:
                    np.testing.assert_allclose(run["D"], reference["D"], atol=1e-12)
        print("Verified five native Haar transforms and 46 compression/bypass/error runs")
    else:
        FIXTURE.write_text(json.dumps(generated, indent=2) + "\n")
        print(f"Wrote {FIXTURE.name}")


if __name__ == "__main__":
    main()
