import itertools

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import (
    prt_conditional_toxicity,
    prt_decision,
    prt_final_selection,
    prt_interval_loglikelihood,
    prt_predictive_risk,
)


def test_predictive_dependence_beta_moments_and_exact_future_enumeration():
    total = np.array([0.5 - 1 / np.sqrt(12), 0.5 + 1 / np.sqrt(12)])
    future = np.repeat(total[:, None], 2, axis=1)
    result = prt_predictive_risk(
        total, future, target=0.5, negligible_cutoff=0.3, excessive_cutoff=0.7
    )
    assert_allclose([result.beta_alpha, result.beta_beta], [1, 1])
    assert_allclose(result.count_probability, [1 / 3] * 3)
    assert_allclose(result.updated_exceedance, [1 / 8, 1 / 2, 7 / 8])
    assert_allclose(
        [result.predictive_negligible, result.predictive_acceptable, result.predictive_excessive],
        [1 / 3] * 3,
    )
    # Integrating patient probabilities separately would incorrectly give .25,.5,.25.
    assert not np.allclose(result.count_probability, [0.25, 0.5, 0.25])
    assert not result.count_probability.flags.writeable
    rng = np.random.default_rng(6901)
    total = rng.beta(2, 7, 37)
    future = total[:, None] * rng.uniform(0.1, 1, size=(37, 5))
    expected = np.zeros(6)
    for outcomes in itertools.product((0, 1), repeat=5):
        expected[sum(outcomes)] += np.prod(np.where(outcomes, future, 1 - future), axis=1).mean()
    for chunk in (1, 16, 256):
        assert_allclose(
            prt_predictive_risk(total, future, chunk_size=chunk).count_probability,
            expected,
            atol=1e-14,
            rtol=0,
        )
    complete = prt_predictive_risk(total, np.empty((37, 0)))
    assert complete.decision_negligible == 1 and complete.decision_excessive == 0
    assert_allclose(complete.count_probability, [1])


def test_probit_model_and_published_decision_boundaries():
    b = np.zeros((3, 2))
    assert_allclose(prt_conditional_toxicity(b), [[7 / 8] * 2, [3 / 4] * 2, [1 / 2] * 2, [0] * 2])
    assert_allclose(
        prt_interval_loglikelihood(b, [[2, 1], [1, 0], [0, 0]], [[0, 1], [1, 0], [0, 0]]),
        6 * np.log(0.5),
    )
    assert np.isfinite(prt_interval_loglikelihood([[1000]], [[1]], [[1]]))
    assert prt_conditional_toxicity([[-10]])[0, 0] > 0  # Retain very small hazards.

    def decide(xi, pn, pe, m, k):
        return prt_decision(xi, pn, pe, m, k)

    assert decide([0.91, 0.95], [0, 0], [1, 1], [1, 1], 0).rule == "3"
    d = decide([0.2, 0.9, 0.95], [0] * 3, [0] * 3, [1] * 3, 2)
    assert (d.rule, d.dose) == ("4", 1)  # Upper cutoff equality remains acceptable.
    assert decide([0.3, 0.6], [0, 0], [0.05, 0], [1, 1], 0).rule == "5.1"
    assert decide([0.3, 0.6], [0, 0], [0.051, 0], [1, 1], 0).rule == "5.2"
    assert decide([0.2, 0.95], [0.95, 0], [0.05, 1], [1, 1], 0).action == "stay"
    assert decide([0.1, 0.2], [1, 0], [0, 0.06], [1, 1], 1).rule == "6.1"
    d = decide([0.2, 0.4], [0.95, 0], [0, 0.05], [1, 1], 0)
    assert (d.rule, d.dose) == ("6.2", 1)
    assert decide([0.2, 0.4], [0.95, 0], [0, 0.051], [1, 1], 0).rule == "6.3"
    assert decide([0.2, 0.4], [0.94, 0], [0, 0], [1, 1], 0).rule == "6.4"
    assert decide([0.2, 0.4], [0, 0], [1, 1], [0, 0], 0).rule == "6.2"
    assert prt_final_selection([0.2, 0.5, 0.95], [0.1, 0.29, 0.3]) == 1
    assert prt_final_selection([0.91, 0.99], [0.3, 0.6]) is None
