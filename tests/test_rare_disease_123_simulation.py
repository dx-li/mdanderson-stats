from functools import lru_cache
from math import comb

import numpy as np
import pytest

from mdanderson_stats import RareDisease123Design, simulate_rare_disease_123


@pytest.mark.parametrize("vary_toxicity", [True, False])
def test_simulation_matches_exhaustive_cohort_paths(vary_toxicity):
    design = RareDisease123Design((1, 1))
    rates = (0.19, 0.32) if vary_toxicity else (0.35, 0.7)

    @lru_cache(None)
    def paths(n, t, r, j, excluded):
        size = {0: 1, 1: 2, 3: 3}[n[j]]
        result = np.zeros(5)  # no OBD, dose 1, dose 2, N at dose 1, N at dose 2
        for count in range(size + 1):
            weight = comb(size, count) * rates[j] ** count * (1 - rates[j]) ** (size - count)
            nn, tt, rr = list(n), list(t), list(r)
            nn[j] += size
            tt[j] += count if vary_toxicity else 0
            rr[j] += size if vary_toxicity else count
            step = design.next_dose(nn, tt, rr, j + 1, eliminated=excluded)
            if step.next_dose is None:
                value = np.zeros(5)
                value[step.selected_dose or 0] = 1
                value[3:] = nn
            else:
                value = paths(
                    tuple(nn), tuple(tt), tuple(rr), step.next_dose - 1, tuple(step.eliminated)
                )
            result += weight * value
        return result

    exact = paths((0, 0), (0, 0), (0, 0), 0, (False, False))
    sim = simulate_rare_disease_123(
        design,
        rates if vary_toxicity else [0, 0],
        [1, 1] if vary_toxicity else rates,
        trials=40000,
        correlation=0,
        rng=172,
    )
    np.testing.assert_allclose(sim.selection_probability, exact[:3], atol=0.012)
    np.testing.assert_allclose(sim.mean_patients, exact[3:], atol=0.05)
    assert not sim.patients.flags.writeable


def test_extremes_correlation_and_reproducibility():
    design = RareDisease123Design((1, 1))
    good = simulate_rare_disease_123(design, [0, 0], [1, 1], start_dose=2, trials=10, rng=1)
    np.testing.assert_array_equal(good.selected_dose, 2)
    np.testing.assert_array_equal(good.patients, np.tile([0, 6], (10, 1)))
    bad = simulate_rare_disease_123(design, [1, 1], [1, 1], trials=10, rng=1)
    np.testing.assert_array_equal(bad.selected_dose, 0)
    np.testing.assert_array_equal(bad.patients, np.tile([3, 0], (10, 1)))
    for rho, efficacy in [(1, [0.2, 0.6]), (-1, [0.8, 0.4])]:
        kwargs = dict(trials=500, correlation=rho, rng=123)
        sim = simulate_rare_disease_123(design, [0.2, 0.6], efficacy, **kwargs)
        repeated = simulate_rare_disease_123(design, [0.2, 0.6], efficacy, **kwargs)
        np.testing.assert_array_equal(sim.patients, repeated.patients)
        np.testing.assert_array_equal(sim.selected_dose, repeated.selected_dose)
        np.testing.assert_array_equal(
            sim.toxicities, sim.responses if rho == 1 else sim.patients - sim.responses
        )
    with pytest.raises(ValueError):
        simulate_rare_disease_123(design, [0.2, 0.6], [0.2], trials=10)
