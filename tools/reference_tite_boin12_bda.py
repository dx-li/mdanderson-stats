"""Exact small-data BDA reference by enumerating labeled missing outcomes.

Integrate each completed-data term under the Dirichlet prior analytically.
This does not run the package Gibbs sampler or its imputation implementation.
"""

import csv
import itertools
from math import lgamma
from pathlib import Path

import numpy as np
from scipy.special import betainc, betaincc, logsumexp


def exact_bda_reference() -> dict[str, float]:
    # BOIN12 order: noT/E, noT/noE, T/E, T/noE.
    prior = np.array([1.2, 0.8, 0.3, 0.7])
    utilities = np.array([100.0, 30.0, 65.0, 0.0])
    # Four labeled patients: complete noT/E; T-pending/E; T/E-pending;
    # both pending. Pending follow-up fractions are .6, .25, and (.2, .4).
    weights = np.array(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.4, 0.0], [0.0, 0.0, 0.75, 1.0], [0.6, 1.0, 0.48, 0.8]]
    )
    options = [np.flatnonzero(row > 0) for row in weights]
    counts, log_masses = [], []
    for assignment in itertools.product(*options):
        completed = np.bincount(assignment, minlength=4)
        # B(prior+counts)/B(prior): denominator terms are common to all
        # assignments, but retained here to make the integration explicit.
        log_integral = lgamma(float(prior.sum())) - lgamma(float(prior.sum() + 4))
        log_integral += sum(
            lgamma(float(a + n)) - lgamma(float(a)) for a, n in zip(prior, completed, strict=True)
        )
        log_masses.append(
            log_integral + sum(np.log(weights[i, c]) for i, c in enumerate(assignment))
        )
        counts.append(completed)
    completed = np.asarray(counts)
    mass = np.exp(np.asarray(log_masses) - logsumexp(log_masses))
    probability_mean = mass @ ((prior + completed) / (prior.sum() + 4))
    count_mean = mass @ completed
    t = completed[:, 2] + completed[:, 3]
    e = completed[:, 0] + completed[:, 2]
    x = completed @ utilities / 100.0
    phi_t, phi_e = 0.35, 0.25
    benchmark_base = utilities @ np.array(
        [(1 - phi_t) * phi_e, (1 - phi_t) * (1 - phi_e), phi_t * phi_e, phi_t * (1 - phi_e)]
    )
    benchmark = (benchmark_base + (100.0 - benchmark_base) / 2.0) / 100.0
    result = {f"p_{j}": float(v) for j, v in enumerate(probability_mean)}
    result.update({f"count_{j}": float(v) for j, v in enumerate(count_mean)})
    result.update(
        toxicity_overdose_probability=float(mass @ betaincc(1 + t, 1 + 4 - t, phi_t)),
        efficacy_futility_probability=float(mass @ betainc(1 + e, 1 + 4 - e, phi_e)),
        utility_mean=float(mass @ (100 * (1 + x) / 6)),
        utility_probability=float(mass @ betaincc(1 + x, 1 + 4 - x, benchmark)),
        utility_events=float(mass @ x),
    )
    return result


if __name__ == "__main__":
    reference = exact_bda_reference()
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/tite-boin12-bda-reference.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reference), lineterminator="\n")
        writer.writeheader()
        writer.writerow(reference)
    print(f"Wrote {path.name} from exact missing-state enumeration.")
