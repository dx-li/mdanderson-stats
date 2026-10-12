"""Record deterministic frozen-component outputs of original WFMM1/WFMM2.

Active random draws are deliberately excluded: this is not an RNG fixture.
"""

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

SHA2 = "4db3d8724fe80e5450b6b666b67de2967d625f421c5e05a9cc0d31049e3a3f0e"
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.directory.resolve()
for name, digest in (("wfmm1", SHA), ("wfmm2", SHA2)):
    if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
        raise ValueError("native checksum mismatch")
cases = []
with tempfile.TemporaryDirectory(prefix="wfmm-frozen-reference-") as temporary:
    root = Path(temporary)
    environment = dict(os.environ, LD_LIBRARY_PATH=str(source), OMP_NUM_THREADS="1")
    for multiplier in (0.5, 1.0, 1.5, 4.0):
        path = root / "input.mat"
        savemat(
            path,
            dict(
                Y=np.tile(Y, (1, 4)),
                model=dict(
                    X=np.ones((8, 1)),
                    C=np.ones((8, 1)),
                    Z=np.repeat(np.eye(4), 2, axis=0),
                    m=np.array([4.0]),
                ),
                basis_specs=dict(
                    transformtype="wavelet",
                    wavelet="db1",
                    boundary="periodic",
                    nlevels=2.0,
                    extended_mode=1.0,
                ),
                MCMCspecs=dict(
                    B=32.0,
                    burnin=8.0,
                    thin=1.0,
                    time_update=1.0,
                    nj_nosmooth=0.0,
                    propvar_omega=multiplier,
                ),
            ),
        )
        for executable, inputs in (
            ("wfmm1", [str(path)]),
            ("wfmm2", [str(root / "input_Init.mat"), str(root / "output")]),
        ):
            subprocess.run(
                [str(source / executable), *inputs],
                env=environment,
                capture_output=True,
                check=True,
                timeout=30,
            )
        native = loadmat(root / "input_Init.mat")
        initial = native["omega_MLE"]
        scale = multiplier * native["se_omega"]
        frozen = scale == 0
        draws = np.fromfile(root / "output_0_theta.dat", dtype="<f8").reshape(32, *initial.shape)
        if not np.all(draws[:, frozen] == initial[frozen]):
            raise AssertionError("native zero-scale components changed")
        cases.append(
            dict(
                multiplier=multiplier,
                initial=initial.tolist(),
                proposal_sd=scale.tolist(),
                frozen_indices=np.argwhere(frozen).tolist(),
                frozen_draws=draws[:, frozen].tolist(),
                retained_draws=32,
            )
        )
args.output.write_text(
    json.dumps(
        dict(
            wfmm1_sha256=SHA,
            wfmm2_sha256=SHA2,
            scope="Zero-proposal floor components only; active draws and RNG identity excluded",
            cases=cases,
        ),
        indent=2,
    )
    + "\n"
)
print(len(cases), "native frozen-component cases")
