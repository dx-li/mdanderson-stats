"""Audit reduced periodic/zero/reflection boundaries with supplied impulse coefficient draws.

External checksum-pinned native binaries are research dependencies only.
Failures and reconstruction errors are recorded, never interpreted as acceptance.
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
native = args.directory.resolve()
hashes = {
    "wfmm1": "0c18cac7225eca3db264b8a4e769a45960cde0bfbf68fc2c1554c83cf451cf8e",
    "wfmm2": "4db3d8724fe80e5450b6b666b67de2967d625f421c5e05a9cc0d31049e3a3f0e",
    "wfmm3": "1434793493b1718c31f18920f48d100b49563777c7e2a22c5c8796855ed99857",
}
for name, sha in hashes.items():
    if hashlib.sha256((native / name).read_bytes()).hexdigest() != sha:
        raise ValueError("original executable checksum mismatch")
temporary = tempfile.TemporaryDirectory(prefix="wfmm-reduced-audit-")
root = Path(temporary.name)
env = dict(os.environ, LD_LIBRARY_PATH=str(native), OMP_NUM_THREADS="1")
records = []
for case in [
    dict(wavelet=w, n=n, boundary=b, levels=depth)
    for w in ["db1", "db2", "db3"]
    for n in [32, 33]
    for b in ["periodic", "zero", "reflection"]
    for depth in [1, 2]
]:
    name, n, boundary, levels = case["wavelet"], case["n"], case["boundary"], case["levels"]
    extended = 0
    p = root / "inverse.mat"
    init = root / "inverse_Init.mat"
    prefix = root / "draws"
    summary = root / "summary.mat"
    savemat(
        p,
        dict(
            Y=np.eye(n),
            model=dict(X=np.ones((n, 1)), C=np.ones((n, 1))),
            basis_specs=dict(
                transformtype="wavelet",
                wavelet=name,
                nlevels=float(levels),
                boundary=boundary,
                extended_mode=float(extended),
            ),
            MCMCspecs=dict(nj_nosmooth=0.0, B=128.0, burnin=0.0, thin=1.0, time_update=1.0),
        ),
    )
    r = dict(wavelet=name, n=n, extended=extended, boundary=boundary, levels=levels)

    def run(exe, *args):
        out = subprocess.run(
            [str(native / exe), *map(str, args)], env=env, capture_output=True, timeout=30
        )
        if out.returncode:
            raise RuntimeError(exe + ": " + (out.stdout + out.stderr).decode().splitlines()[-1])

    try:
        run("wfmm1", p)
        a = loadmat(init)["D"]
        k = a.shape[1]
        run("wfmm2", init, prefix)
        draws = np.zeros((128, k), dtype="<f8")
        draws[:k] = np.eye(k)
        draws.tofile(root / "draws_0_wbeta.dat")
        run("wfmm3", init, prefix, summary, 1)
        b = np.fromfile(root / "summary_beta.dat", "<f8").reshape(128, -1)[:k]
        r.update(
            analysis=a.tolist(),
            synthesis=b.tolist(),
            reconstruction_error=float(abs(a @ b - np.eye(n)).max()),
        )
        print(name, n, extended, "inverse", r["reconstruction_error"], flush=True)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        r["error"] = str(e)
        print(name, n, extended, "ERROR", str(e).splitlines()[-1], flush=True)
    records.append(r)
args.output.write_text(
    json.dumps(dict(executable_sha256=hashes, cases=records), separators=(",", ":")) + "\n"
)
temporary.cleanup()
