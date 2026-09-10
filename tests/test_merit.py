from itertools import product

import numpy as np
import pytest
from scipy.optimize import isotonic_regression
from scipy.stats import beta, binom

from mdanderson_stats import MERITDesign, merit_monitor, merit_sample_size, simulate_merit
from mdanderson_stats.merit_search import _union_table


def test_isotonic_decision_and_inclusive_boundaries():
    design = MERITDesign(10, 4, 5, doses=4)
    t = np.array(list(product(range(4), repeat=4)), dtype=float)
    e = 10 - t
    result = design.select(t, e)
    np.testing.assert_allclose(result.toxicity, [isotonic_regression(row).x for row in t])
    np.testing.assert_allclose(result.efficacy, [isotonic_regression(row).x for row in e])
    decision = design.select([6, 2, 4, 8], [8, 2, 5, 9])
    np.testing.assert_equal(decision.toxicity, [4, 4, 4, 8])
    np.testing.assert_equal(decision.efficacy, [5, 5, 5, 9])
    np.testing.assert_equal(decision.admissible, [True, True, True, False])
    assert not decision.admissible.flags.writeable
    with pytest.raises(ValueError):
        MERITDesign(10.0, 4, 5)


def test_all_boundary_probabilities_against_direct_enumeration():
    # Every possible count pair for three arms, n=2; equal-weight pooling
    # creates fractional counts, which must not be rounded before decisions.
    data = np.array(list(product(range(3), repeat=6)))
    transformed = MERITDesign(2, 1, 1, doses=3).select(data[:, :3], data[:, 3:])
    table = _union_table(transformed.toxicity, transformed.efficacy, 2)
    for mt, me in product(range(3), repeat=2):
        expected = ((transformed.toxicity <= mt) & (transformed.efficacy >= me)).any(1).mean()
        assert table[mt, me] == pytest.approx(expected, abs=1e-15)
    np.testing.assert_equal(_union_table(data[:, :0], data[:, :0], 2), np.zeros((3, 3)))


def test_monitor_and_independent_binary_operating_characteristics():
    monitor = merit_monitor(10, [1, 8], [1, 8], toxicity_target=0.2, efficacy_target=0.4)
    np.testing.assert_allclose(
        monitor.toxicity_probability, beta.sf(0.2, np.array([1, 8]) + 0.1, np.array([9, 2]) + 0.1)
    )
    np.testing.assert_allclose(
        monitor.futility_probability, beta.cdf(0.4, np.array([1, 8]) + 0.1, np.array([9, 2]) + 0.1)
    )
    np.testing.assert_equal(monitor.stop_toxicity, [False, True])
    np.testing.assert_equal(monitor.stop_futility, [True, False])
    design = MERITDesign(12, 3, 4, isotonic_toxicity=False, isotonic_efficacy=False)
    sim = simulate_merit(
        design,
        [0.15, 0.5],
        [0.55, 0.6],
        correlation=0,
        trials=50000,
        truly_admissible=[True, False],
        rng=160,
    )
    p = binom.cdf(3, 12, [0.15, 0.5]) * binom.sf(3, 12, [0.55, 0.6])
    np.testing.assert_allclose(sim.selection_probability, p, atol=0.006)
    assert sim.power_one == pytest.approx(p[0] * (1 - p[1]), abs=0.006)
    assert sim.power_two == pytest.approx(p[0], abs=0.006)
    correlated = simulate_merit(
        MERITDesign(1, 0, 1, isotonic_toxicity=False, isotonic_efficacy=False),
        [0.5, 0.5],
        [0.5, 0.5],
        correlation=0.5,
        trials=50000,
        rng=161,
    )
    # P(X>0,Y<=0)=1/4-arcsin(rho)/(2*pi) for standardized bivariate normals.
    np.testing.assert_allclose(correlated.selection_probability, 1 / 6, atol=0.006)
    deterministic = simulate_merit(design, [0, 1], [1, 0], correlation=-1, trials=10, rng=3)
    np.testing.assert_equal(deterministic.selection_probability, [1, 0])


def test_search_constraints_and_independent_validation():
    result = merit_sample_size(
        toxicity_null=0.7,
        toxicity_alternative=0.1,
        efficacy_null=0.1,
        efficacy_alternative=0.7,
        alpha=0.15,
        power=0.7,
        max_patients_per_arm=20,
        trials=4000,
        rng=160,
    )
    assert result.global_type_one_error <= 0.15
    assert result.global_power_two >= 0.7
    assert result.null_rates.shape == (6, 2, 2)
    assert result.alternative_rates.shape == (3, 2, 2)
    # Independent samples check the selected configuration; no reuse of search draws.
    for i, rates in enumerate(result.null_rates):
        sim = simulate_merit(result.design, rates[:, 0], rates[:, 1], trials=12000, rng=1000 + i)
        assert sim.any_selection_probability == pytest.approx(result.null_error[i], abs=0.025)
    for i, rates in enumerate(result.alternative_rates):
        truth = (rates[:, 0] == 0.1) & (rates[:, 1] == 0.7)
        sim = simulate_merit(
            result.design,
            rates[:, 0],
            rates[:, 1],
            trials=12000,
            truly_admissible=truth,
            rng=2000 + i,
        )
        assert sim.power_one == pytest.approx(result.alternative_power_one[i], abs=0.025)
        assert sim.power_two == pytest.approx(result.alternative_power_two[i], abs=0.025)
