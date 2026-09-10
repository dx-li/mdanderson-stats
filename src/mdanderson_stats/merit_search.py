"""Monte Carlo MERIT boundary search with batched inclusion-exclusion tables."""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from ._validation import FloatArray, finite
from .boin import _owned
from .merit import MERITDesign, _integer, _isotonic
from .merit_simulation import _correlation, _draw


def _union_table(t: FloatArray, e: FloatArray, n: int) -> FloatArray:
    """P(any dose passes) for every integer (toxicity_max, efficacy_min)."""
    counts = np.zeros((n + 1, n + 1), dtype=np.int64)
    for mask in range(1, 1 << t.shape[1]):
        indices = [j for j in range(t.shape[1]) if mask & (1 << j)]
        lo = np.ceil(t[:, indices].max(1)).astype(np.int64)
        hi = np.floor(e[:, indices].min(1)).astype(np.int64)
        histogram = np.bincount(lo * (n + 1) + hi, minlength=(n + 1) ** 2)
        counts += (1 if len(indices) % 2 else -1) * histogram.reshape(n + 1, n + 1)
    # Each dose defines a rectangle: mT>=ceil(t) and mE<=floor(e).
    totals = counts.cumsum(0)[:, ::-1].cumsum(1)[:, ::-1]
    return totals / t.shape[0]


@dataclass(frozen=True)
class MERITSearch:
    design: MERITDesign
    global_type_one_error: float
    global_power_one: float
    global_power_two: float
    null_error: FloatArray
    alternative_power_one: FloatArray
    alternative_power_two: FloatArray
    null_rates: FloatArray
    alternative_rates: FloatArray
    trials: int
    correlation: float
    power_definition: int


def merit_sample_size(
    *,
    toxicity_null: float,
    toxicity_alternative: float,
    efficacy_null: float,
    efficacy_alternative: float,
    doses: int = 2,
    alpha: float = 0.1,
    power: float = 0.8,
    power_definition: int = 2,
    max_patients_per_arm: int = 100,
    trials: int = 10000,
    correlation: float = 0.5,
    isotonic_toxicity: bool = True,
    isotonic_efficacy: bool = True,
    rng: int | np.random.Generator | None = None,
) -> MERITSearch:
    """Find the first feasible per-arm n using Monte Carlo probabilities.

    Enumerates the paper's null and alternative corner configurations. These
    are not the entire continuous composite null. All alternative configurations
    are evaluated, including when isotonic pooling couples arm decisions.
    Ties: greatest chosen global power, smallest global error, smallest mT,
    then greatest mE. No interim stopping. Estimated constraints are not
    guarantees; validate the selected design with independent simulations.
    """
    doses = _integer(doses, "doses", 2, 4)
    maximum = _integer(max_patients_per_arm, "max_patients_per_arm", 1, 200)
    trials = _integer(trials, "trials", 100, 100000)
    definition = _integer(power_definition, "power_definition", 1, 2)
    # Reuse flag validation from the decision object.
    MERITDesign(1, 1, 0, doses, isotonic_toxicity, isotonic_efficacy)
    rates = finite(
        [toxicity_null, toxicity_alternative, efficacy_null, efficacy_alternative, alpha, power],
        "rates and targets",
    )
    if (
        rates.shape != (6,)
        or np.any((rates <= 0) | (rates >= 1))
        or not toxicity_alternative < toxicity_null
        or not efficacy_null < efficacy_alternative
    ):
        raise ValueError(
            "require interior rates/targets, toxicity_alternative < null "
            "and efficacy_null < alternative"
        )
    rho = _correlation(correlation)
    null, alternative, truths = [], [], []
    dose_index = np.arange(doses)
    for s in range(doses + 1):
        for k in range(s, doses + 1):
            null.append(
                np.stack(
                    [
                        np.where(dose_index < s, toxicity_alternative, toxicity_null),
                        np.where(dose_index < k, efficacy_null, efficacy_alternative),
                    ],
                    -1,
                )
            )
    for u in range(doses):
        for v in range(u + 1, doses + 1):
            alternative.append(
                np.stack(
                    [
                        np.where(dose_index < v, toxicity_alternative, toxicity_null),
                        np.where(dose_index < u, efficacy_null, efficacy_alternative),
                    ],
                    -1,
                )
            )
            truths.append((dose_index >= u) & (dose_index < v))
    null_rates, alt_rates = np.array(null), np.array(alternative)
    quantiles = ndtri(np.concatenate([null_rates, alt_rates]))
    t = np.zeros((len(quantiles), trials, doses), dtype=np.int64)
    e = np.zeros_like(t)
    generator = np.random.default_rng(rng)
    for n in range(1, maximum + 1):
        # Common random numbers across n/scenarios; independent patients and arms.
        zt, ze = _draw(generator, trials, doses, rho)
        t += zt[None, :, :] <= quantiles[:, None, :, 0]
        e += ze[None, :, :] <= quantiles[:, None, :, 1]
        tx = _isotonic(t.astype(float)) if isotonic_toxicity else t
        ex = _isotonic(e.astype(float)) if isotonic_efficacy else e
        null_tables, one_tables, two_tables = [], [], []
        for scenario in range(len(quantiles)):
            all_selected = _union_table(tx[scenario], ex[scenario], n)
            if scenario < len(null):
                null_tables.append(all_selected)
            else:
                truth = truths[scenario - len(null)]
                bad = _union_table(tx[scenario][:, ~truth], ex[scenario][:, ~truth], n)
                one_tables.append(np.clip(all_selected - bad, 0, 1))
                two_tables.append(_union_table(tx[scenario][:, truth], ex[scenario][:, truth], n))
        errors, powers_one, powers_two = (
            np.array(null_tables),
            np.array(one_tables),
            np.array(two_tables),
        )
        worst_error = errors.max(0)
        worst_one, worst_two = powers_one.min(0), powers_two.min(0)
        chosen_power = worst_one if definition == 1 else worst_two
        candidate_t, candidate_e = np.nonzero((worst_error <= alpha) & (chosen_power >= power))
        if candidate_t.size:
            ordering = np.lexsort(
                (
                    -candidate_e,
                    candidate_t,
                    worst_error[candidate_t, candidate_e],
                    -chosen_power[candidate_t, candidate_e],
                )
            )
            mt, me = int(candidate_t[ordering[0]]), int(candidate_e[ordering[0]])
            return MERITSearch(
                MERITDesign(n, mt, me, doses, isotonic_toxicity, isotonic_efficacy),
                float(worst_error[mt, me]),
                float(worst_one[mt, me]),
                float(worst_two[mt, me]),
                _owned(errors[:, mt, me]),
                _owned(powers_one[:, mt, me]),
                _owned(powers_two[:, mt, me]),
                _owned(null_rates),
                _owned(alt_rates),
                trials,
                rho,
                definition,
            )
    raise ValueError(
        f"no design meets the estimated constraints through {maximum} patients per arm"
    )
