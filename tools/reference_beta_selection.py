"""Record native EM selection chains and SIMCVM refits of fixed NumPy samples."""

import json
from pathlib import Path

import numpy as np
from reference_beta_mixture import evaluate, fitted_output
from reference_multi import SOURCE, build


def main():
    executable = build()
    provenance = json.loads(Path("tests/fixtures/beta_mixture.json").read_text())
    published = [float(v) for v in (SOURCE / "pvals.txt").read_text().split()]
    previous = {"null_weight": 1.0, "weights": [], "a": [], "b": []}
    chain = []
    desktop_chain = []
    for _ in range(3):
        start = fitted_output(evaluate(executable, 18, published, previous))
        if start["status"]:
            raise RuntimeError(f"Native initializer failed: {start}")
        fit = fitted_output(evaluate(executable, 19, published, start["model"]))
        if fit["status"]:
            raise RuntimeError(f"Native fit failed: {fit}")
        chain.append(fit)
        desktop_fit = fitted_output(evaluate(executable, 22, published, start["model"]))
        if desktop_fit["status"]:
            raise RuntimeError(f"Native desktop EM failed: {desktop_fit}")
        desktop_chain.append(desktop_fit)
        previous = fit["model"]
    model = {"null_weight": 0.6, "weights": [0.4], "a": [0.5], "b": [5.0]}
    rng = np.random.default_rng(56)
    samples = []
    for _ in range(4):
        labels = rng.choice(2, size=60, p=[0.6, 0.4])
        data = rng.beta(np.array([1, 0.5])[labels], np.array([1, 5])[labels])
        fit = fitted_output(evaluate(executable, 19, data, model, tolerance=1e-3))
        if fit["status"]:
            raise RuntimeError(f"Native simulation refit failed: {fit}")
        samples.append({"pvalues": data.tolist(), "reference": fit})
    result = {
        "provenance": {
            key: value
            for key, value in provenance.items()
            if key not in ("evaluations", "starts", "em_fits", "ml_fits")
        },
        "notes": "Original STBETA/EMBETA sequential fits and desktop EMBETA at matching starts; "
        "fixed NumPy samples refitted "
        "with original EMBETA at SIMCVM CONCRI=1e-3. This does not reproduce RANF draws.",
        "published": published,
        "em_chain": chain,
        "desktop_em_fits": desktop_chain,
        "bootstrap_model": model,
        "bootstrap_seed": 56,
        "bootstrap_samples": samples,
    }
    Path("tests/fixtures/beta_selection.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print("Recorded three S fits, three desktop fits and four native bootstrap refits")


if __name__ == "__main__":
    main()
