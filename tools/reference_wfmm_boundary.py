"""Capture native 1D extended-mode analysis and synthesis of synthetic impulses.

Original WFMM1/2/3 binaries must be supplied outside distributable artifacts.
WFMM2 creates the source file layout; replace its beta draws by deterministic
unit vectors before WFMM3 reconstruction. No sampled values become references.
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

HASHES = {
    "wfmm1": "0c18cac7225eca3db264b8a4e769a45960cde0bfbf68fc2c1554c83cf451cf8e",
    "wfmm2": "4db3d8724fe80e5450b6b666b67de2967d625f421c5e05a9cc0d31049e3a3f0e",
    "wfmm3": "1434793493b1718c31f18920f48d100b49563777c7e2a22c5c8796855ed99857",
}
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.directory.resolve()
for name, sha in HASHES.items():
    if hashlib.sha256((source / name).read_bytes()).hexdigest() != sha:
        raise ValueError("original executable checksum mismatch: " + name)
records = []
with tempfile.TemporaryDirectory(prefix="wfmm-boundary-reference-") as temporary:
    root = Path(temporary)
    env = dict(os.environ, LD_LIBRARY_PATH=str(source), OMP_NUM_THREADS="1")

    def run(name, *arguments):
        result = subprocess.run(
            [str(source / name), *map(str, arguments)], env=env, capture_output=True, timeout=30
        )
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr).decode(errors="replace")[-2000:])

    for order in range(1, 11):
        for size, levels in [(32, 1), (33, 2)]:
            for boundary in ["periodic", "zero", "reflection"]:
                path = root / "input.mat"
                init = root / "input_Init.mat"
                prefix = root / "draws"
                summary = root / "summary.mat"
                savemat(
                    path,
                    dict(
                        Y=np.eye(size),
                        model=dict(X=np.ones((size, 1)), C=np.ones((size, 1))),
                        basis_specs=dict(
                            transformtype="wavelet",
                            wavelet=f"db{order}",
                            nlevels=float(levels),
                            boundary=boundary,
                            extended_mode=1.0,
                        ),
                        MCMCspecs=dict(
                            nj_nosmooth=0.0, B=128.0, burnin=0.0, thin=1.0, time_update=1.0
                        ),
                    ),
                )
                run("wfmm1", path)
                analysis = loadmat(init)["D"]
                k = analysis.shape[1]
                assert k <= 128
                run("wfmm2", init, prefix)
                draws = np.zeros((128, k), dtype="<f8")
                draws[:k] = np.eye(k)
                draws.tofile(root / "draws_0_wbeta.dat")
                run("wfmm3", init, prefix, summary, 1)
                synthesis = np.fromfile(root / "summary_beta.dat", dtype="<f8").reshape(128, -1)[:k]
                records.append(
                    dict(
                        order=order,
                        size=size,
                        levels=levels,
                        boundary=boundary,
                        analysis=analysis.tolist(),
                        synthesis=synthesis.tolist(),
                    )
                )
args.output.write_text(
    json.dumps(dict(executable_sha256=HASHES, cases=records), separators=(",", ":")) + "\n"
)
print(len(records), "native extended analysis/synthesis cases")
