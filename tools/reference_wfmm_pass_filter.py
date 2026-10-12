"""Capture original periodic Haar pass selections and invalid combinations."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
from reference_wfmm_initialization import SHA
from scipy.io import loadmat, savemat

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.directory.resolve()
if hashlib.sha256((source / "wfmm1").read_bytes()).hexdigest() != SHA:
    raise ValueError("native executable checksum mismatch")
cases = []
with tempfile.TemporaryDirectory(prefix="wfmm-pass-reference-") as temporary:
    root = Path(temporary)
    rng = np.random.default_rng(8261)
    for size, levels in ((16, 1), (16, 2), (32, 3), (32, 4)):
        y = rng.normal(1, 2, size=(8, size))
        runs = []
        full = None
        for high in range(levels + 1):
            for low, alpha in product(range(levels + 1), (1.0, 0.7)):
                path = root / "input.mat"
                savemat(
                    path,
                    dict(
                        Y=y,
                        model=dict(X=np.ones((8, 1)), C=np.ones((8, 1))),
                        basis_specs=dict(
                            transformtype="wavelet",
                            wavelet="db1",
                            nlevels=float(levels),
                            boundary="periodic",
                            extended_mode=1.0,
                            highpass=float(high),
                            lowpass=float(low),
                            alphawav=alpha,
                            t=3.0 if alpha < 1 else 0.0,
                        ),
                        MCMCspecs=dict(nj_nosmooth=0.0),
                    ),
                )
                run = subprocess.run(
                    [str(source / "wfmm1"), str(path)],
                    env=dict(os.environ, LD_LIBRARY_PATH=str(source), OMP_NUM_THREADS="1"),
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                result = dict(
                    highpass=high,
                    lowpass=low,
                    alpha=alpha,
                    t=3 if alpha < 1 else 0,
                    exit_code=run.returncode,
                )
                if run.returncode:
                    result["errors"] = [
                        line for line in (run.stdout + run.stderr).splitlines() if "ERROR:" in line
                    ]
                else:
                    data = loadmat(root / "input_Init.mat", simplify_cells=True)
                    d = loadmat(root / "input_Init.mat")["D"]
                    if high == low == 0 and alpha == 1:
                        full = d.copy()
                    assert full is not None
                    indices = [
                        int(np.argmin(np.max(np.abs(full - col[:, None]), axis=0))) for col in d.T
                    ]
                    if not np.allclose(full[:, indices], d, atol=1e-12, rtol=0):
                        raise AssertionError("filtered values differ from original coefficients")
                    result.update(D=d.tolist(), indices=indices, J=int(data["basis_specs"]["J"]))
                runs.append(result)
        cases.append(dict(Y=y.tolist(), levels=levels, D_full=full.tolist(), runs=runs))
args.output.write_text(json.dumps(dict(executable_sha256=SHA, cases=cases), indent=2) + "\n")
print(sum(len(c["runs"]) for c in cases), "native pass-filter cases")
