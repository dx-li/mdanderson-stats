"""Capture native spatial pass filtering before initialization; no native fit claim."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from reference_wfmm_spatial import capture
from scipy.io import savemat

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
native = args.directory.resolve()
sha = "0c18cac7225eca3db264b8a4e769a45960cde0bfbf68fc2c1554c83cf451cf8e"
if hashlib.sha256((native / "wfmm1").read_bytes()).hexdigest() != sha:
    raise ValueError("original executable checksum mismatch")
records = []
with tempfile.TemporaryDirectory(prefix="wfmm-spatial-pass-") as temp:
    root = Path(temp)
    for dims, j, wavelet, boundary, rect in [
        ((8, 8), 2, "db1", "periodic", False),
        ((7, 5), 2, "db2", "reflection", False),
        ((4, 4, 4), 1, "db1", "periodic", False),
        ((4, 4, 4), 2, "db1", "zero", False),
        ((8, 4), 2, "db1", "periodic", True),
    ]:
        n = int(np.prod(dims))
        full = None
        for hi in range(j + 1):
            for lo in range(j + 1):
                p = root / "input.mat"
                bs = dict(
                    transformtype="wavelet",
                    dims=np.array(dims, dtype=float),
                    wavelet=wavelet,
                    nlevels=float(j),
                    boundary=boundary,
                    extended_mode=1.0,
                    rectangular=float(rect),
                    highpass=float(hi),
                    lowpass=float(lo),
                )
                if rect:
                    bs.update(wavelet2="db1", nlevels2=1.0, boundary2="periodic")
                savemat(
                    p,
                    dict(
                        Y=np.eye(n),
                        model=dict(X=np.ones((n, 1)), C=np.ones((n, 1))),
                        basis_specs=bs,
                        MCMCspecs=dict(
                            nj_nosmooth=0.0, B=128.0, burnin=0.0, thin=1.0, time_update=1.0
                        ),
                    ),
                )
                r = dict(
                    dims=dims,
                    levels=j,
                    wavelet=wavelet,
                    boundary=boundary,
                    rectangular=rect,
                    highpass=hi,
                    lowpass=lo,
                )
                try:
                    a = capture(p, str(p) + ".log", native)
                    if hi == lo == 0:
                        full = a
                    ix = [int(np.argmin(np.max(abs(full - col[:, None]), axis=0))) for col in a.T]
                    err = float(abs(full[:, ix] - a).max())
                    assert err < 1e-12
                    r.update(indices=ix, max_error=err)
                except RuntimeError:
                    r["failure"] = Path(str(p) + ".log").read_text().splitlines()[-1]
                records.append(r)
args.output.write_text(
    json.dumps(
        dict(
            executable_sha256=sha,
            capture="ANOVA_MOM entry; analysis selection only; no successful native fit claim",
            cases=records,
        ),
        indent=2,
    )
    + "\n"
)
print(len(records), "spatial pass cases")
