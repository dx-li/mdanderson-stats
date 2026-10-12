"""Compare direct prior information with independent logistic-normal quadrature."""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

from mdanderson_stats import read_uaroet_desktop_study, uaroet_desktop_prior_ess

study = read_uaroet_desktop_study(
    (Path(__file__).resolve().parents[1] / "tests/fixtures/uaroet-python-study.json").read_text()
)
result = uaroet_desktop_prior_ess(study, draws=100000, rng=np.random.default_rng(1002))
checks = []
for endpoint, mu, sd in [("efficacy", 0, 0.7), ("toxicity", -1, 0.7)]:
    mean = quad(lambda z: expit(z) * norm.pdf(z, mu, sd), -12, 12, epsabs=1e-12)[0]
    second = quad(lambda z: expit(z) ** 2 * norm.pdf(z, mu, sd), -12, 12, epsabs=1e-12)[0]
    reference = mean * (1 - mean) / (second - mean**2) - 1
    value = float(getattr(result, endpoint)[0, 1])
    error = abs(value / reference - 1)
    assert error < 0.02
    checks.append(dict(endpoint=endpoint, reference=reference, python=value, relative_error=error))
print(
    json.dumps(
        dict(
            scope="two baseline logistic-normal prior information checks",
            draws=100000,
            seed=1002,
            checks=checks,
            clinical_validation=False,
        ),
        indent=2,
    )
)
