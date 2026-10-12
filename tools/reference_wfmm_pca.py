"""Capture native PCA/hybrid transform outputs and explicit failures on synthetic data."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from reference_wfmm_initialization import SHA, Y
from scipy.io import loadmat, savemat

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--directory", required=True, type=Path)
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "tests/fixtures/wfmm-native-pca.json",
)
args = parser.parse_args()
source = args.directory.resolve()
if hashlib.sha256((source / "wfmm1").read_bytes()).hexdigest() != SHA:
    raise ValueError("native executable checksum mismatch")
diagonal = np.diag([100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03])
models = []
for typ in ["PC", "PCw", "wPC"]:
    for alpha in [0.0, 0.5, 0.9, 0.999999, 1.0]:
        models.append((f"{typ}_{alpha}", Y, dict(transformtype=typ, alphaPC=alpha)))
for base in [2.0, 10.0, 100.0]:
    models.append(
        (
            f"partition_base_{base}",
            diagonal,
            dict(transformtype="PC", alphaPC=0.999999, PCpartitionbase=base),
        )
    )
models.extend(
    [
        (
            "manual_partitions",
            Y,
            dict(transformtype="PC", alphaPC=0.999999, partitions=np.array([1.0, 3.0])),
        ),
        ("equal_partitions", Y, dict(transformtype="PC", alphaPC=0.999999, npartitions=3.0)),
        ("tall", np.vstack((Y, Y + 0.1)), dict(transformtype="PC", alphaPC=0.999999)),
        ("wide", np.column_stack((Y, Y + 0.1)), dict(transformtype="PC", alphaPC=0.99)),
        ("zero_energy", np.zeros_like(Y), dict(transformtype="PC", alphaPC=0.9)),
    ]
)


def serial(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


cases = []
with tempfile.TemporaryDirectory(prefix="wfmm-pca-reference-") as temporary:
    root = Path(temporary)
    for name, y, settings in models:
        path = root / (name + ".mat")
        basis = dict(wavelet="db1", boundary="periodic", nlevels=1.0, extended_mode=1.0) | settings
        savemat(
            path,
            dict(
                Y=y,
                model=dict(X=np.ones((len(y), 1)), C=np.ones((len(y), 1))),
                basis_specs=basis,
                MCMCspecs=dict(B=8.0, burnin=4.0, thin=1.0, time_update=1.0, nj_nosmooth=0.0),
            ),
        )
        run = subprocess.run(
            [str(source / "wfmm1"), str(path)],
            env=dict(os.environ, LD_LIBRARY_PATH=str(source), OMP_NUM_THREADS="1"),
            capture_output=True,
            text=True,
            timeout=20,
        )
        case = dict(
            case=name,
            y=y.tolist(),
            settings={k: serial(v) for k, v in basis.items()},
            exit_code=run.returncode,
        )
        if run.returncode:
            case["errors"] = [
                line
                for line in (run.stdout + run.stderr).splitlines()
                if "ERROR:" in line or "what():" in line
            ]
        else:
            data = loadmat(root / (name + "_Init.mat"), simplify_cells=True)
            # Load transform matrices without squeezing their singleton axes.
            raw = loadmat(root / (name + "_Init.mat"), struct_as_record=False, squeeze_me=False)
            bs = raw["basis_specs"][0, 0]
            case["D"] = raw["D"].tolist()
            case["basis"] = {
                k: serial(v) for k, v in data["basis_specs"].items() if k not in ("phi", "phi_inv")
            }
            case["analysis"] = bs.phi_inv.tolist()
            case["synthesis"] = bs.phi.tolist()
        cases.append(case)
args.output.write_text(
    json.dumps(dict(executable_sha256=SHA, cases=cases), indent=2, allow_nan=False) + "\n"
)
print(
    len(cases),
    "transform cases",
    sum(c["exit_code"] == 0 for c in cases),
    "successful native outputs",
)
