from dataclasses import replace

import numpy as np
import pytest

from mdanderson_stats.uaroet import (
    uaroet_logits,
    uaroet_parameter_names,
    uaroet_probabilities,
)
from mdanderson_stats.uaroet_decision import uaroet_allocation
from mdanderson_stats.uaroet_fit import fit_uaroet


def test_continuation_logits_and_gaussian_copula_limits() -> None:
    names = uaroet_parameter_names(2, 3, 2)
    assert names == (
        "efficacy.baseline.1",
        "efficacy.baseline.2",
        "efficacy.increment.1.1",
        "efficacy.increment.1.2",
        "toxicity.baseline.1",
        "toxicity.increment.1.1",
    )
    e, t = uaroet_logits(
        [-1, 1, 0.5, 0.25, -2, 0.4], dose_count=2, efficacy_levels=3, toxicity_levels=2
    )
    assert np.all(e[1] >= e[0])
    assert t[1, 0] >= t[0, 0]
    independent = uaroet_probabilities(e, t, association=0)
    assert independent.joint == pytest.approx(
        independent.efficacy[:, :, None] * independent.toxicity[:, None, :]
    )
    assert independent.joint.sum(axis=(1, 2)) == pytest.approx([1, 1])

    positive = uaroet_probabilities(e, t, association=1)
    negative = uaroet_probabilities(e, t, association=-1)
    assert positive.joint.sum(axis=2) == pytest.approx(positive.efficacy)
    assert positive.joint.sum(axis=1) == pytest.approx(positive.toxicity)
    assert negative.joint.sum(axis=2) == pytest.approx(negative.efficacy)
    assert negative.joint.sum(axis=1) == pytest.approx(negative.toxicity)


def test_fixed_independence_posterior_and_allocation() -> None:
    counts = np.zeros((2, 2, 2), dtype=int)
    counts[0, 1, 0] = 2
    counts[1, 1, 1] = 1
    names = uaroet_parameter_names(2, 2, 2)
    fit = fit_uaroet(
        counts,
        prior_mean=np.array([-1.0, 0.5, -2.0, 0.3]),
        prior_sd=np.ones(len(names)),
        association=0.0,
        draws=8,
        warmup=2,
        chains=2,
        max_evaluations=1000,
        rng=np.random.default_rng(31),
    )
    assert fit.joint.shape == (2, 8, 2, 2, 2)
    assert fit.joint.sum(axis=(-2, -1)) == pytest.approx(np.ones((2, 8, 2)))
    allocation = uaroet_allocation(
        fit,
        [[0.0, 0.0], [1.0, 0.5]],
        [3, 0],
        toxicity_limit=0.5,
        p_L=0,
        p_U=1,
        utility_tolerance=1,
        good_utility_cutoff=0.75,
    )
    assert allocation.probabilities.sum() == pytest.approx(1)
    assert allocation.eligible.tolist() == [True, True]
    assert not fit.joint.flags.writeable


def test_no_data_uses_prior_draws_and_final_rule_is_explicit() -> None:
    names = uaroet_parameter_names(3, 2, 2)
    fit = fit_uaroet(
        np.zeros((3, 2, 2), dtype=int),
        prior_mean=np.zeros(len(names)),
        prior_sd=np.ones(len(names)),
        association=0,
        draws=8,
        warmup=99,
        chains=2,
        rng=np.random.default_rng(8),
    )
    assert fit.direct_prior
    assert fit.warmup == 0
    joint = np.zeros((2, 2, 3, 2, 2))
    outcomes = [
        ((1, 0), (1, 0), (1, 1)),
        ((1, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 0), (1, 0)),
        ((0, 0), (0, 0), (1, 0)),
    ]
    for draw, row in enumerate(outcomes):
        chain, index = divmod(draw, 2)
        for dose, cell in enumerate(row):
            joint[chain, index, dose, cell[0], cell[1]] = 1
    fit = replace(fit, joint=joint)
    utility = [[0.0, 0.0], [1.0, 0.5]]
    allocation = uaroet_allocation(
        fit,
        utility,
        [1, 1, 1],
        toxicity_limit=0.1,
        p_L=0,
        p_U=1,
        utility_tolerance=1,
        good_utility_cutoff=0.9,
    )
    assert allocation.probability_best == pytest.approx([0.25, 0.5, 0.5])
    assert allocation.probability_best.sum() == pytest.approx(1.25)
    assert allocation.mean_utility == pytest.approx([0.375, 0.5, 0.75])
    assert allocation.probabilities == pytest.approx([0.2, 0.4, 0.4])
    acceptable_final = uaroet_allocation(
        fit,
        utility,
        [1, 1, 1],
        toxicity_limit=0.1,
        p_L=0,
        p_U=0.25,
        utility_tolerance=1,
        good_utility_cutoff=0.9,
        final=True,
    )
    paper_final = uaroet_allocation(
        fit,
        utility,
        [1, 1, 1],
        toxicity_limit=0.1,
        p_L=0,
        p_U=0.25,
        utility_tolerance=1,
        good_utility_cutoff=0.9,
        final=True,
        final_rule="paper_global",
    )
    assert acceptable_final.best_dose == 1
    assert paper_final.best_dose == 2
    start = uaroet_allocation(fit, utility, [0, 0, 0], toxicity_limit=0.3, starting_dose=0)
    assert start.action == "start"
    with pytest.raises(ValueError, match="requires at least one treated"):
        uaroet_allocation(fit, utility, [0, 0, 0], toxicity_limit=0.3, final=True)
