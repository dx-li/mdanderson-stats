"""Audit Mstar/Vstar input selection without assuming an imputation likelihood."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from reference_wfmm_spatial import capture
from scipy.io import loadmat, savemat

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
native = args.directory.resolve()
sha = "0c18cac7225eca3db264b8a4e769a45960cde0bfbf68fc2c1554c83cf451cf8e"
if hashlib.sha256((native / "wfmm1").read_bytes()).hexdigest() != sha:
    raise ValueError("original executable checksum mismatch")
records = []
with tempfile.TemporaryDirectory(prefix="wfmm-imputed-audit-") as tmp:
    root = Path(tmp)
    env = dict(os.environ, LD_LIBRARY_PATH=str(native), OMP_NUM_THREADS="1")
    y = np.random.default_rng(28).normal(size=(8, 16))
    model = dict(X=np.ones((8, 1)), C=np.ones((8, 1)))
    basis = dict(
        transformtype="wavelet", wavelet="db1", nlevels=1.0, boundary="periodic", extended_mode=1.0
    )
    specs = dict(B=8.0, burnin=0.0, time_update=1.0, thin=1.0, nj_nosmooth=0.0)
    path = root / "baseline.mat"
    savemat(path, dict(Y=y, model=model, basis_specs=basis, MCMCspecs=specs))
    subprocess.run(
        [str(native / "wfmm1"), str(path)], env=env, capture_output=True, timeout=30, check=True
    )
    saved = loadmat(root / "baseline_Init.mat", simplify_cells=True)
    processed = {
        k: np.asarray(v, dtype=float) if not isinstance(v, str) else v
        for k, v in saved["basis_specs"].items()
    }
    for flag in (0, 1):
        for include_y, include_v, include_m, complete in [
            (True, True, True, False),
            (True, False, False, False),
            (False, True, True, False),
            (False, True, True, True),
            (False, True, False, True),
        ]:
            for level in [0.0, 0.1, 1.0] if flag and complete and include_m else [0.1]:
                m = saved["D"] + level
                data = dict(
                    model=model,
                    basis_specs=processed if complete else basis,
                    MCMCspecs=dict(specs, missing_data=float(flag)),
                )
                if include_y:
                    data["Y"] = y
                if include_v:
                    data["Vstar"] = np.full_like(y, level)
                if include_m:
                    data["Mstar"] = m
                path = root / "case.mat"
                log = root / "capture.log"
                savemat(path, data)
                row = dict(
                    missing_data=flag,
                    Y=include_y,
                    Vstar=include_v,
                    Mstar=include_m,
                    complete_basis=complete,
                    level=level,
                )
                try:
                    d = capture(path, log, native)
                    row.update(
                        captured_shape=list(d.shape),
                        max_error_Mstar=float(abs(d - m).max()),
                        max_error_transformed_Y=float(abs(d - saved["D"]).max()),
                    )
                except (RuntimeError, TimeoutError) as exc:
                    errors = [
                        line for line in log.read_text().splitlines() if line.startswith("ERROR:")
                    ]
                    row["capture_error"] = errors[-1] if errors else str(exc)
                if flag and complete and include_m:
                    r = subprocess.run(
                        [str(native / "wfmm1"), str(path)], env=env, capture_output=True, timeout=30
                    )
                    errors = [
                        line
                        for line in (r.stdout + r.stderr).decode().splitlines()
                        if line.startswith("ERROR:")
                    ]
                    row.update(
                        full_exit_code=r.returncode, full_error=errors[-1] if errors else None
                    )
                records.append(row)
args.output.write_text(json.dumps(dict(executable_sha256=sha, cases=records), indent=2) + "\n")
print(len(records), "imputed input-selection cases")
