"""Audit native summaries using supplied synthetic draws, separate from native MCMC.

Exports native discrepancies as research evidence, not Python scientific targets.
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
        raise ValueError("original executable checksum mismatch")
rng = np.random.default_rng(789)
y = rng.normal(size=(8, 16))
draws = rng.normal(size=(40, 1, 16)) + np.linspace(0.1, 1.6, 16)
cases = []
with tempfile.TemporaryDirectory(prefix="wfmm-postprocess-reference-") as temporary:
    root = Path(temporary)
    env = dict(os.environ, LD_LIBRARY_PATH=str(source), OMP_NUM_THREADS="1")
    for contrast in [False, True]:
        settings = dict(quantiles=np.array([0.1, 0.5, 0.9]), effect_size=np.array([0.25, 0.5]))
        if contrast:
            settings.update(
                L=np.array([[1.0], [-1.0]]),
                LT=np.column_stack([np.ones(16) / 16, np.arange(16) / 120]),
            )
        path, init, prefix, summary = (
            root / name for name in ["input.mat", "input_Init.mat", "draws", "summary.mat"]
        )
        savemat(
            path,
            dict(
                Y=y,
                model=dict(X=np.ones((8, 1)), C=np.ones((8, 1))),
                basis_specs=dict(
                    transformtype="wavelet",
                    wavelet="db1",
                    boundary="periodic",
                    nlevels=1.0,
                    extended_mode=1.0,
                ),
                MCMCspecs=dict(nj_nosmooth=0.0, B=40.0, burnin=0.0, thin=1.0, time_update=1.0),
                PostProcessSpecs=settings,
            ),
        )

        def run(name, *arguments):
            result = subprocess.run(
                [str(source / name), *map(str, arguments)], env=env, capture_output=True, timeout=30
            )
            if result.returncode:
                raise RuntimeError((result.stdout + result.stderr).decode(errors="replace")[-2000:])

        run("wfmm1", path)
        run("wfmm2", init, prefix)
        draws.astype("<f8").tofile(root / "draws_0_wbeta.dat")
        run("wfmm3", init, prefix, summary, 1)
        result = loadmat(summary)
        beta = np.fromfile(root / "summary_beta.dat", dtype="<f8").reshape(40, 1, 16)
        # Independent one-level Haar synthesis, not package production code.
        expected = np.empty_like(beta)
        expected[:, :, ::2] = (draws[:, :, :8] + draws[:, :, 8:]) / np.sqrt(2)
        expected[:, :, 1::2] = (draws[:, :, :8] - draws[:, :, 8:]) / np.sqrt(2)
        np.testing.assert_allclose(beta, expected, rtol=0, atol=2e-15)
        groups = {"Lbeta": beta if not contrast else np.einsum("ep,dpt->det", settings["L"], beta)}
        if contrast:
            groups["LbetaLt"] = groups["Lbeta"] @ settings["LT"]
        errors = {}
        for label, values in groups.items():
            mean = values.mean(0)
            sd = values.std(0, ddof=1)
            source_mean = result[label + "_mean"]
            source_sd = result[label + "_sd"]
            np.testing.assert_allclose(source_mean, mean, rtol=0, atol=2e-15)
            np.testing.assert_allclose(source_sd, sd, rtol=0, atol=2e-15)
            # Native pointwise quantiles select n*p-1 for these integer ranks,
            # and pass through float32. This probe does not infer all ranks.
            q = np.quantile(
                values, [0.1, 0.5, 0.9], axis=0, method="interpolated_inverted_cdf"
            ).astype(np.float32)
            np.testing.assert_array_equal(result[label + "_quantiles"], q.reshape(3, -1))
            for suffix, expected in [
                (
                    "peffects",
                    np.mean(abs(values)[None] > np.array([0.25, 0.5])[:, None, None, None], axis=1),
                ),
                ("p0", 2 * np.minimum(np.mean(values > 0, axis=0), np.mean(values < 0, axis=0))),
            ]:
                np.testing.assert_allclose(
                    result[label + "_" + suffix].reshape(expected.shape),
                    expected,
                    rtol=0,
                    atol=2e-15,
                )
            native_z = np.max(abs(np.trunc(values - mean)) / sd, axis=2)
            critical = np.quantile(native_z, 0.9, axis=0, method="hazen")
            np.testing.assert_allclose(
                result[label + "_upperCI"].reshape(mean.shape),
                mean + critical[:, None] * sd,
                rtol=0,
                atol=3e-15,
            )
            np.testing.assert_allclose(
                result[label + "_lowerCI"].reshape(mean.shape),
                mean - critical[:, None] * sd,
                rtol=0,
                atol=3e-15,
            )
            grid = np.r_[np.arange(1, 100) / 100, np.arange(991, 1000) / 1000]
            qz = np.quantile(native_z, grid, axis=0, method="hazen")
            scores = abs(np.trunc(mean / sd))
            simbas = np.array(
                [
                    1
                    - grid[np.clip(np.searchsorted(qz[:, i], scores[i], side="right") - 1, 0, 107)]
                    for i in range(mean.shape[0])
                ]
            )
            np.testing.assert_allclose(result[label + "_simbas"], simbas, rtol=0, atol=2e-15)
            exact_z = np.max(abs(values - mean) / sd, axis=2)
            exact_tail = np.mean(exact_z[:, :, None] >= abs(mean / sd)[None], axis=0)
            errors[label] = dict(
                native_vs_real_valued_simbas_max_abs=float(np.max(abs(simbas - exact_tail)))
            )
        cases.append(
            dict(
                contrasts={k: np.asarray(v).tolist() for k, v in settings.items()},
                outputs={k: v.tolist() for k, v in result.items() if k.startswith("Lbeta")},
                checks=errors,
            )
        )
args.output.write_text(
    json.dumps(dict(executable_sha256=HASHES, coefficients=draws.tolist(), cases=cases), indent=2)
    + "\n"
)
print(
    "2 original summary cases verified: reconstruction, moments, probabilities, "
    "contrasts, native quantiles and integer-truncated bands"
)
