"""Pinned native byte ordering and report means/SDs from synthetic supplied draws."""

import argparse
import hashlib
import json
import os
import shutil
import struct
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
hashes = dict(
    wfmm1="0c18cac7225eca3db264b8a4e769a45960cde0bfbf68fc2c1554c83cf451cf8e",
    wfmm2="4db3d8724fe80e5450b6b666b67de2967d625f421c5e05a9cc0d31049e3a3f0e",
    wfmm3="1434793493b1718c31f18920f48d100b49563777c7e2a22c5c8796855ed99857",
)
for name, sha in hashes.items():
    if hashlib.sha256((native / name).read_bytes()).hexdigest() != sha:
        raise ValueError("original executable checksum mismatch")
records = []
with tempfile.TemporaryDirectory(prefix="wfmm-file-reference-") as tmp:
    root = Path(tmp)
    env = dict(os.environ, LD_LIBRARY_PATH=str(native), OMP_NUM_THREADS="1")

    def run(name, *arguments):
        r = subprocess.run(
            [str(native / name), *map(str, arguments)], env=env, capture_output=True, timeout=30
        )
        if r.returncode:
            raise RuntimeError((r.stdout + r.stderr).decode(errors="replace")[-1000:])

    for p, chains in [(1, 1), (2, 1), (2, 2)]:
        y = np.random.default_rng(237).normal(size=(12, 8))
        x = np.column_stack([np.ones(12), np.tile([-1, 1], 6)])[:, :p]
        path, init, prefix, summary = (
            root / n for n in ["input.mat", "input_Init.mat", "draws", "summary.mat"]
        )
        savemat(
            path,
            dict(
                Y=y,
                model=dict(X=x, C=np.ones((12, 1))),
                basis_specs=dict(
                    transformtype="wavelet",
                    wavelet="db1",
                    nlevels=1.0,
                    boundary="periodic",
                    extended_mode=1.0,
                ),
                MCMCspecs=dict(B=40.0, burnin=0.0, thin=1.0, time_update=1.0, nj_nosmooth=0.0),
            ),
        )
        run("wfmm1", path)
        run("wfmm2", init, prefix)
        all_draws = []
        input_hashes = []
        for chain in range(chains):
            if chain:
                for suffix in ["theta", "newtheta"]:
                    shutil.copyfile(
                        root / f"draws_0_{suffix}.dat", root / f"draws_{chain}_{suffix}.dat"
                    )
            draws = (np.arange(40 * p * 8).reshape(40, p, 8) % 31 - 15) / 8 + chain / 4
            payload = struct.pack("<" + "d" * draws.size, *draws.ravel())
            (root / f"draws_{chain}_wbeta.dat").write_bytes(payload)
            all_draws.append(draws)
            input_hashes.append(hashlib.sha256(payload).hexdigest())
        run("wfmm3", init, prefix, summary, chains)
        raw = (root / "summary_beta.dat").read_bytes()
        beta = np.array(struct.unpack("<" + "d" * (len(raw) // 8), raw)).reshape(chains * 40, p, 8)
        coefficients = np.concatenate(all_draws)
        expected = np.empty_like(coefficients)
        expected[:, :, ::2] = (coefficients[:, :, :4] + coefficients[:, :, 4:]) / np.sqrt(2)
        expected[:, :, 1::2] = (coefficients[:, :, :4] - coefficients[:, :, 4:]) / np.sqrt(2)
        np.testing.assert_allclose(beta, expected, rtol=0, atol=2e-15)
        m = loadmat(summary)
        np.testing.assert_allclose(m["Lbeta_mean"], beta.mean(0), atol=2e-15)
        np.testing.assert_allclose(m["Lbeta_sd"], beta.std(0, ddof=1), atol=2e-15)
        records.append(
            dict(
                components=p,
                chains=chains,
                draws_per_chain=40,
                coordinates=8,
                coefficients=coefficients.tolist(),
                curves=beta.tolist(),
                input_sha256=input_hashes,
                output_sha256=hashlib.sha256(raw).hexdigest(),
                mean=m["Lbeta_mean"].tolist(),
                sd=m["Lbeta_sd"].tolist(),
            )
        )
args.output.write_text(
    json.dumps(dict(executable_sha256=hashes, cases=records), separators=(",", ":")) + "\n"
)
print(len(records), "native draw/report cases")
