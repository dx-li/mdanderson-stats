"""Record original MULTI model evaluations, STBETA starts and EMBETA fits."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from reference_multi import SOURCE, build
from reference_numerics import RAW, run


def evaluate(executable, mode, values, model, tolerance=1e-6):
    k = len(model["a"])
    data = f"{mode} {len(values)} {tolerance} 0\n"
    data += " ".join(map(str, sorted(values))) + "\n"
    data += f"{k} {model['null_weight']}\n"
    if k:
        for key in ("weights", "a", "b"):
            data += " ".join(map(str, model[key])) + "\n"
    return run(executable, data).splitlines()


def fitted_output(rows):
    status = int(rows[0])
    if status:
        return {"status": status}
    p0, likelihood, cvm = map(float, rows[1].split())
    parameters = [list(map(float, line.split())) for line in rows[2:]]
    return {
        "status": 0,
        "model": {
            "null_weight": p0,
            "weights": [p[0] for p in parameters],
            "a": [p[1] for p in parameters],
            "b": [p[2] for p in parameters],
        },
        "log_likelihood": likelihood,
        "cramer_von_mises": cvm,
    }


def main():
    executable = build()
    uniform = {"null_weight": 1.0, "weights": [], "a": [], "b": []}
    models = [
        uniform,
        {"null_weight": 0.65, "weights": [0.35], "a": [0.4], "b": [8.0]},
        {"null_weight": 0.2, "weights": [0.8], "a": [2.0], "b": [3.0]},
        {"null_weight": 0.5, "weights": [0.3, 0.2], "a": [0.4, 2.0], "b": [8.0, 3.0]},
        {"null_weight": 0.0, "weights": [1.0], "a": [2.0], "b": [2.0]},
    ]
    published = [float(v) for v in (SOURCE / "pvals.txt").read_text().split()]
    families = [
        [0, 1e-300, 0.001, 0.02, 0.1, 0.5, 0.99, 1],
        published,
        np.linspace(0.01, 0.99, 40).tolist(),
    ]
    evaluations = []
    for model in models:
        for values in families:
            rows = evaluate(executable, 17, values, model)
            likelihood, cvm = map(float, rows[0].split())
            outputs = [list(map(float, row.split())) for row in rows[1:]]
            evaluations.append(
                {
                    "pvalues": values,
                    "model": model,
                    "log_likelihood": likelihood,
                    "cramer_von_mises": cvm,
                    "sorted_density": [o[0] for o in outputs],
                    "sorted_cdf": [o[1] for o in outputs],
                    "sorted_null_posterior": [o[2] for o in outputs],
                }
            )
    starts, fits = [], []
    previous = uniform
    for k in (1, 2):
        start = fitted_output(evaluate(executable, 18, published, previous))
        starts.append({"pvalues": published, "previous": previous, "reference": start})
        if start["status"]:
            raise RuntimeError(f"Original start {k} failed")
        fitted = fitted_output(evaluate(executable, 19, published, start["model"]))
        fits.append(
            {
                "pvalues": published,
                "initial": start["model"],
                "tolerance": 1e-6,
                "reference": fitted,
            }
        )
        if fitted["status"]:
            raise RuntimeError(f"Original EM fit {k} failed")
        previous = fitted["model"]
    rng = np.random.default_rng(317)
    synthetic = np.r_[rng.uniform(size=300), rng.beta(0.5, 6, size=200)].tolist()
    for tolerance in (1e-4, 1e-6):
        fitted = fitted_output(evaluate(executable, 19, synthetic, models[1], tolerance))
        fits.append(
            {
                "pvalues": synthetic,
                "initial": models[1],
                "tolerance": tolerance,
                "reference": fitted,
            }
        )
    endpoints = [0, 0.01, 0.02, 0.03, 0.1, 0.2, 0.5, 0.7, 0.9, 1]
    endpoint_initial = {"null_weight": 0.5, "weights": [0.5], "a": [1.0], "b": [1.0]}
    fits.append(
        {
            "pvalues": endpoints,
            "initial": endpoint_initial,
            "tolerance": 1e-6,
            "legacy_endpoints": True,
            "reference": fitted_output(evaluate(executable, 19, endpoints, endpoint_initial)),
        }
    )
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    starts.append(
        {
            "pvalues": values,
            "previous": uniform,
            "reference": fitted_output(evaluate(executable, 18, values, uniform)),
        }
    )
    result = {
        "source_url": "https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MULTI/MULTI_V1.tar.gz",
        "sha256": hashlib.sha256((RAW / "MULTI/MULTI_V1.tar.gz").read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": ["-O2", "-std=legacy", "-fallow-argument-mismatch"],
        "notes": "Original S INITLN, BPVAL, CALCVM, STBETA and EMBETA; EMBETA renamed "
        "to distinguish its extra likelihood argument. Desktop MIXBET/MIXPRB/LGLK "
        "and identical MOMBET/SWPPAR helpers reused. INITLN endpoint approximation applied.",
        "evaluations": evaluations,
        "starts": starts,
        "em_fits": fits,
    }
    Path("tests/fixtures/beta_mixture.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Recorded {len(evaluations)} model evaluations, {len(starts)} starts, {len(fits)} EM fits"
    )


if __name__ == "__main__":
    main()
