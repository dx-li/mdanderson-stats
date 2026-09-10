"""Published boundaries, independent prior integrals and complete trial paths."""

from itertools import product

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad

from mdanderson_stats import bayes_factor_binary_design


def test_all_published_boundaries_and_report_scenarios():
    design = bayes_factor_binary_design(50)
    assert_array_equal(design.inferiority_max, [-1, 0, 1, 3, 4, 5, 6, 8, 9])
    assert_array_equal(design.superiority_min, [6, 8, 9, 11, 12, 13, 15, 16, 17])
    oc = design.operating_characteristics([0.2, 0.25, 0.3, 0.35, 0.4])
    # Guide's 1,000-replicate report, compared using Monte Carlo sampling error.
    reported = np.array([0.037, 0.168, 0.378, 0.657, 0.858])
    assert np.all(
        abs(oc.superiority - reported) < 4 * np.sqrt(oc.superiority * (1 - oc.superiority) / 1000)
    )
    assert_allclose(oc.superiority + oc.inferiority + oc.inconclusive, 1, atol=1e-14)
    state = design.monitor([9, 10, 16, 17], 50)
    assert_array_equal(
        state.decision, ["inferiority", "inconclusive", "inconclusive", "superiority"]
    )
    assert design.log_bayes_factor[0, 0] == 0


def test_marginal_likelihood_against_direct_normalized_imom_integral():
    for p0, mode in [(0.2, 0.3), (0.001, 0.004), (0.8, 0.95)]:
        design = bayes_factor_binary_design(
            12, null_rate=p0, alternative_mode=mode, min_subjects=2, cohort_size=2
        )
        tau = 1.5 * (mode - p0) ** 2
        normalizer = 2 * np.exp(tau / (1 - p0) ** 2)
        for n, m in [(1, 0), (1, 1), (6, 3), (12, 0), (12, 8), (12, 12)]:

            def integrand(theta):
                gap = theta - p0
                logprior = np.log(normalizer * tau) - 3 * np.log(gap) - tau / gap**2
                logratio = m * np.log(theta / p0) + (n - m) * np.log((1 - theta) / (1 - p0))
                return np.exp(logprior + logratio)

            bf, error = quad(integrand, p0, 1, points=[mode], epsabs=1e-12, epsrel=2e-11, limit=300)
            assert error < max(1e-11, bf * 1e-9)
            assert_allclose(design.log_bayes_factor[n, m], np.log(bf), atol=2e-9, rtol=0)


def test_exact_oc_against_all_paths_and_simulation():
    design = bayes_factor_binary_design(
        6,
        null_rate=0.2,
        alternative_mode=0.5,
        inferiority_cutoff=0.3,
        superiority_cutoff=0.7,
        min_subjects=2,
        cohort_size=2,
    )
    p = 0.35
    totals = np.zeros(3)
    sample_pmf = np.zeros(3)
    for path in product([0, 1], repeat=6):
        count = 0
        weight = p ** sum(path) * (1 - p) ** (6 - sum(path))
        for i, y in enumerate(path, 1):
            count += y
            if i % 2:
                continue
            posterior = 1 / (1 + np.exp(-design.log_bayes_factor[i, count]))
            decision = 0 if posterior < 0.3 else 1 if posterior > 0.7 else 2
            if decision != 2 or i == 6:
                totals[decision] += weight
                sample_pmf[i // 2 - 1] += weight
                break
    oc = design.operating_characteristics(p)
    assert_allclose([oc.inferiority, oc.superiority, oc.inconclusive], totals, atol=1e-14)
    assert_allclose(oc.sample_size_probability, sample_pmf, atol=1e-14)
    sim = design.simulate([p, 0, 1], n_trials=30000, rng=94)
    assert np.all(abs(sim.decision_probability[0] - totals) < 5 * sim.monte_carlo_se[0] + 1e-4)
    assert_allclose(sim.expected_sample_size[0], oc.expected_sample_size, atol=0.03)
    assert_array_equal(sim.events[1], 0)
    assert_array_equal(sim.events[2], sim.sample_size[2])
    replay = design.simulate(p, n_trials=30000, rng=94)
    # A fixed input and seed reproduces both stopping time and observed responses.
    replay2 = design.simulate(p, n_trials=30000, rng=94)
    assert_array_equal(replay.events, replay2.events)
    assert_array_equal(replay.sample_size, replay2.sample_size)


def test_disabled_stopping_and_path_decisions():
    design = bayes_factor_binary_design(
        6, min_subjects=1, cohort_size=2, inferiority_cutoff=0, superiority_cutoff=1
    )
    oc = design.operating_characteristics([0, 0.2, 1])
    assert_allclose(oc.inconclusive, 1, atol=1e-14)
    assert_allclose(oc.expected_sample_size, 6, atol=1e-14)
    assert_array_equal(oc.sample_size_quantile(0), 6)
    assert_array_equal(oc.sample_size_quantile(1), 6)
    assert_array_equal(
        design.monitor_outcomes([1] * 6).decision, ["continue"] * 5 + ["inconclusive"]
    )
    early = bayes_factor_binary_design(6, min_subjects=2, cohort_size=2, superiority_cutoff=0.55)
    assert_array_equal(
        early.monitor_outcomes([1, 1, 0, 0, 0, 0]).decision, ["continue"] + ["superiority"] * 5
    )


def test_rare_null_at_maximum_sample_size():
    p0, mode = 1e-6, 2e-6
    design = bayes_factor_binary_design(400, null_rate=p0, alternative_mode=mode)
    tau = 1.5 * (mode - p0) ** 2
    # All-success marginal likelihood, in the original theta coordinate.
    marginal, error = quad(
        lambda theta: (
            theta**400
            * 2
            * tau
            / (theta - p0) ** 3
            * np.exp(tau / (1 - p0) ** 2 - tau / (theta - p0) ** 2)
        ),
        p0,
        1,
        epsabs=1e-26,
        epsrel=1e-11,
        limit=300,
    )
    assert error < marginal * 1e-9
    assert_allclose(
        design.log_bayes_factor[400, 400], np.log(marginal) - 400 * np.log(p0), atol=2e-9, rtol=0
    )
