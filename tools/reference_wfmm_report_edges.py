"""Capture native report edge behavior without making it a scientific target."""

import argparse
import hashlib
import json
import os
import shutil
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
native = args.directory.resolve()
for name, sha in HASHES.items():
    if hashlib.sha256((native / name).read_bytes()).hexdigest() != sha:
        raise ValueError("native checksum mismatch")
temporary = tempfile.TemporaryDirectory(prefix="wfmm-report-edges-")
root = Path(temporary.name)
rng = np.random.default_rng(1663)
records = []
random_coefficients = None
y = rng.normal(size=(8, 16))
env = dict(os.environ, LD_LIBRARY_PATH=str(native), OMP_NUM_THREADS="1")
for label, random, chains in [
    ("multi", False, 2),
    ("random", True, 1),
    ("zero", False, 1),
    ("constant", False, 1),
    ("quantile_three", False, 1),
]:
    path = root / (label + ".mat")
    init = root / (label + "_Init.mat")
    prefix = root / (label + "_out")
    summary = root / (label + "_summary.mat")
    model = dict(X=np.ones((8, 1)), C=np.ones((8, 1)))
    if random:
        model["Z"] = np.repeat(np.eye(2), 4, axis=0)
    savemat(
        path,
        dict(
            Y=y,
            model=model,
            basis_specs=dict(
                transformtype="wavelet",
                wavelet="db1",
                boundary="periodic",
                nlevels=1.0,
                extended_mode=1.0,
            ),
            MCMCspecs=dict(nj_nosmooth=0.0, B=40.0, burnin=0.0, thin=1.0, time_update=1.0),
            PostProcessSpecs=dict(
                quantiles=np.array(
                    [0.13, 0.37, 0.83]
                    if label == "quantile_three"
                    else [0.01, 0.13, 0.37, 0.5, 0.83, 0.99]
                ),
                compute_U=float(random),
                keep_U_samples=float(random),
            ),
        ),
    )

    def run(name, *args):
        r = subprocess.run(
            [str(native / name), *map(str, args)], env=env, capture_output=True, timeout=30
        )
        if r.returncode:
            raise RuntimeError((r.stdout + r.stderr).decode()[-1500:])

    run("wfmm1", path)
    run("wfmm2", init, prefix)
    theta = np.full((40, 2 if random else 1, 16), 0.5)
    if random:
        theta[:, 0] = 0.3
    theta.astype("<f8").tofile(str(prefix) + "_0_theta.dat")
    all_draws = []
    for chain in range(chains):
        if chain:
            for suffix in ["theta", "newtheta"]:
                shutil.copyfile(
                    str(prefix) + "_0_" + suffix + ".dat",
                    str(prefix) + "_" + str(chain) + "_" + suffix + ".dat",
                )
        d = rng.normal(size=(40, 1, 16)) + 0.3 * chain
        if label == "random":
            random_coefficients = d.copy()
        if label == "quantile_three":
            assert random_coefficients is not None
            d = random_coefficients.copy()
        if label == "zero":
            d[:] = 0
        if label == "constant":
            d[:] = 0.4
        d.astype("<f8").tofile(str(prefix) + "_" + str(chain) + "_wbeta.dat")
        all_draws.append(d)
    np.save(root / (label + "_input.npy"), np.concatenate(all_draws))
    run("wfmm3", init, prefix, summary, chains)
    m = loadmat(summary, simplify_cells=True)

    beta = np.fromfile(root / (label + "_summary_beta.dat"), "<f8").reshape(40 * chains, 1, 16)
    d = np.concatenate(all_draws)
    expected = np.empty_like(d)
    expected[:, :, ::2] = (d[:, :, :8] + d[:, :, 8:]) / np.sqrt(2)
    expected[:, :, 1::2] = (d[:, :, :8] - d[:, :, 8:]) / np.sqrt(2)
    np.testing.assert_allclose(beta, expected, rtol=0, atol=2e-15)
    np.testing.assert_allclose(m["Lbeta_mean"], beta.mean(0)[0], rtol=0, atol=2e-15)
    np.testing.assert_allclose(m["Lbeta_sd"], beta.std(0, ddof=1)[0], rtol=0, atol=2e-15)
    ordered = np.sort(beta[:, 0], axis=0).astype(np.float32)
    quantiles = np.asarray(m["Lbeta_quantiles"])
    ranks = np.array(
        [
            [np.argmin(abs(ordered[:, j] - value)) for j, value in enumerate(row)]
            for row in quantiles
        ]
    )
    np.testing.assert_array_equal(np.take_along_axis(ordered, ranks, axis=0), quantiles)
    record = dict(
        label=label,
        quantiles=np.asarray(m["PostProcessSpecs"]["quantiles"]).tolist(),
        chains=chains,
        Y=y.tolist(),
        coefficients=d.tolist(),
        omega=theta[0].tolist(),
        outputs={
            k: np.asarray(v).tolist()
            for k, v in m.items()
            if k.startswith("Lbeta") or k.startswith("U_")
        },
        quantile_order_statistic_ranks=ranks.tolist(),
    )
    if random:
        u = np.fromfile(root / (label + "_summary_U.dat"), "<f8").reshape(40, 2, 16)
        np.testing.assert_allclose(m["U_mean"], u.mean(0), rtol=0, atol=2e-15)
        np.testing.assert_allclose(m["U_sd"], u.std(0, ddof=1), rtol=0, atol=2e-15)
        record["U_draws"] = u.tolist()

        def analysis(v):
            return np.concatenate(
                (
                    (v[..., ::2] + v[..., 1::2]) / np.sqrt(2),
                    (v[..., ::2] - v[..., 1::2]) / np.sqrt(2),
                ),
                axis=-1,
            )

        variance = 1 / (1 / 0.3 + 4 / 0.5)
        mean = np.einsum("nm,dnk->dmk", model["Z"], analysis(y)[None] - d) / 0.5 * variance
        innovations = (analysis(u) - mean) / np.sqrt(variance)
        # Source WFMM::Abs casts to int before taking absolute value. With
        # this covariance below one, means below one suppress the normal draw.
        skipped = np.max(np.abs(mean), axis=1) < 1.0
        by_vector = np.moveaxis(innovations, 1, 2)
        np.testing.assert_allclose(by_vector[skipped], 0.0, rtol=0, atol=3e-15)
        record["conditional_variance_investigation"] = dict(
            status="resolved: integer-truncated WFMM::Abs suppresses conditional noise",
            gaussian_variance=variance,
            standardized_mean=float(innovations.mean()),
            standardized_variance=float(innovations.var(ddof=1)),
            mean_only_vectors=int(skipped.sum()),
            sampled_vectors=int((~skipped).sum()),
            sampled_standardized_variance=float(by_vector[~skipped].var(ddof=1)),
            scope="Observed native defect; not a scientific Python sampling target",
        )
    records.append(record)
args.output.write_text(
    json.dumps(
        dict(
            executable_sha256=HASHES, quantiles=[0.01, 0.13, 0.37, 0.5, 0.83, 0.99], cases=records
        ),
        indent=2,
    )
    + "\n"
)
temporary.cleanup()
print(
    "5 native edge cases: pooled files, random-function reports, noninteger quantiles and zero SD"
)
