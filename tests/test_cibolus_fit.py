from dataclasses import replace

import numpy as np

from mdanderson_stats.cibolus import CiBolusObservation, CiBolusPrior
from mdanderson_stats.cibolus_decision import cibolus_decision
from mdanderson_stats.cibolus_fit import fit_cibolus


def _prior() -> CiBolusPrior:
    return CiBolusPrior(
        np.log([0.5, 0.7, 0.8, 0.08, 1.4, 1.6, 0.9, 0.12, 0.25, 0.2, 0.1]),
        [0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2],
    )


def _fit(observations: list[CiBolusObservation]):
    return fit_cibolus(
        observations,
        _prior(),
        [0.2, 0.4],
        [0.1, 0.2],
        [0.5, 1.0],
        utility=np.array([[100, 0], [80, 20], [60, 30], [0, 0]]),
        draws=8,
        warmup=2,
        chains=2,
        rng=np.random.default_rng(17),
    )


def test_direct_prior_and_posterior_sampler_are_bounded_and_reproducible() -> None:
    first = _fit([])
    second = _fit([])
    assert first.direct_prior
    assert first.log_parameters.shape == (2, 8, 11)
    assert first.joint.shape == (2, 8, 2, 2, 4, 2)
    assert first.likelihood_evaluations == 0
    np.testing.assert_array_equal(first.log_parameters, second.log_parameters)
    np.testing.assert_allclose(first.joint.sum(axis=(-2, -1)), 1, atol=1e-12)

    observed = _fit(
        [
            CiBolusObservation(0.4, 0.2, "exact", True, time=0.35),
            CiBolusObservation(0.2, 0.1, "interval", False, lower=0.2, upper=0.5),
            CiBolusObservation(0.4, 0.2, "failure", False),
        ]
    )
    assert not observed.direct_prior
    assert observed.likelihood_evaluations >= 2 * (1 + 2 + 8)
    assert np.all(np.isfinite(observed.log_likelihood))


def test_decision_uses_strict_tail_screens_and_reports_start_override() -> None:
    fit = _fit([])
    tox = np.broadcast_to(np.array([[0.1, 0.5], [0.3, 0.2]])[None, None, :, :], (2, 8, 2, 2)).copy()
    response = np.full_like(tox, 0.8)
    utility = np.broadcast_to(np.array([[1.0, 20.0], [3.0, 2.0]])[None, None], tox.shape).copy()
    fit = replace(
        fit,
        toxicity_at_one_response=tox,
        response_at_one=response,
        expected_utility=utility,
    )
    empty = cibolus_decision(
        fit,
        np.zeros((2, 2), dtype=int),
        toxicity_limit=0.4,
        toxicity_cutoff=0.5,
        efficacy_limit=0.5,
        efficacy_cutoff=0.1,
        starting=(1, 1),
    )
    assert empty.action == "start" and empty.pair == (1, 1)
    assert not empty.acceptable[0, 1]

    tox3 = np.full((2, 8, 3, 2), 0.1)
    response3 = np.full_like(tox3, 0.8)
    tox3[:, :, 1, 0] = 0.4
    response3[:, :, 1, 0] = 0.5
    tox3.reshape(16, 3, 2)[:8, 0, 1] = 0.5
    response3.reshape(16, 3, 2)[:8, 0, 1] = 0.4
    utility3 = np.broadcast_to(
        np.array([[1.0, 2.0], [3.0, 2.0], [4.0, 10.0]])[None, None], tox3.shape
    ).copy()
    fit = replace(
        fit,
        toxicity_at_one_response=tox3,
        response_at_one=response3,
        expected_utility=utility3,
    )
    interim = cibolus_decision(
        fit,
        [[1, 0], [0, 0], [0, 0]],
        toxicity_limit=0.4,
        toxicity_cutoff=0.5,
        efficacy_limit=0.5,
        efficacy_cutoff=0.5,
    )
    assert interim.action == "treat"
    assert interim.pair == (1, 0)
    assert interim.toxicity_probability[0, 1] == 0.5
    assert interim.acceptable[0, 1]  # Outer cutoff equality is allowed.
    assert interim.inefficacy_probability[0, 1] == 0.5
    assert interim.acceptable[1, 0]  # Inner event thresholds are strict.
    assert interim.eligible[1, 0] and interim.eligible[1, 1]
    assert not np.any(interim.eligible[2])

    final = cibolus_decision(
        fit,
        [[1, 0], [0, 0], [0, 0]],
        toxicity_limit=0.4,
        toxicity_cutoff=0.5,
        efficacy_limit=0.5,
        efficacy_cutoff=0.0,
        final=True,
    )
    assert final.pair == (2, 1)
    assert final.eligible[2, 1]
