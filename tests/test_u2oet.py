from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import logsumexp

from mdanderson_stats import U2OETMarginal, u2oet_probabilities, u2oet_standardize


def test_independent_r_ordinal_models_joint_likelihood_and_utility():
    reference = np.loadtxt(
        Path(__file__).parent / "fixtures/u2oet-reference.csv", skiprows=1, delimiter=","
    )
    for case, levels in enumerate((4, 3, 2, 4, 4), 1):
        alpha = np.array([-0.8, 0.1, -0.3])[: levels - 1]
        beta = np.array([[0.7, 1.2], [0.4, 0.8], [0.3, 1.1]])[: levels - 1]
        powers = (1, 1) if case == 2 else (0.4, 2)
        interaction = (0, 0.6, -0.4, 0, 0)[case - 1]
        result = u2oet_probabilities(
            [4, 5, 6],
            [40, 60, 80],
            efficacy=U2OETMarginal(alpha, beta, powers, 0.7, interaction),
            toxicity=U2OETMarginal(-alpha, beta[:, ::-1], powers[::-1], 1.8, -interaction),
            association=(0.6, -0.8, 1, -1, 0)[case - 1],
            model=("pds", "cmi", "pds+cmi", "pds", "pds")[case - 1],
            centering="linear" if case == 4 else "log",
        )
        expected = reference[reference[:, 0] == case, -1].reshape(3, 3, levels, levels)
        assert_allclose(result.joint, expected, atol=8e-16, rtol=2e-12)
        assert_allclose(logsumexp(result.log_joint, axis=(-2, -1)), 0, atol=1e-15)
        assert_allclose(logsumexp(result.log_joint, axis=-1), result.log_efficacy, atol=2e-15)
        assert_allclose(logsumexp(result.log_joint, axis=-2), result.log_toxicity, atol=2e-15)
        counts = np.arange(expected.size).reshape(expected.shape) % 4
        assert_allclose(result.loglikelihood(counts), np.sum(counts * np.log(expected)), rtol=1e-12)
        utility = np.arange(levels**2).reshape(levels, levels)
        assert_allclose(result.expected_utility(utility), np.sum(expected * utility, axis=(-2, -1)))
        assert not result.log_joint.flags.writeable
        if case == 5:
            # Positive dose effects make every upper category tail nondecreasing.
            tail = np.exp(np.logaddexp.accumulate(result.log_efficacy[..., ::-1], axis=-1))
            assert np.all(np.diff(tail, axis=0) >= -1e-15)
            assert np.all(np.diff(tail, axis=1) >= -1e-15)


def test_extreme_logistic_tails_copula_boundaries_and_dose_units():
    # Two binary outcomes with logits -1000 at the middle (mean) dose pair.
    marginal = U2OETMarginal([-1000], [[1, 1]])
    for rho in (-1, 0, 1):
        result = u2oet_probabilities(
            [1, 2, 3], [1, 2, 3], efficacy=marginal, toxicity=marginal, association=rho
        )
        assert np.all(np.isfinite(result.log_joint))
        # At rho=-1 the rare/rare cell is p^2*(2p-p^2), p=expit(-1000).
        expected = -3000 + np.log(2) if rho == -1 else -2000 + np.log1p(rho)
        assert_allclose(result.log_joint[1, 1, 1, 1], expected, atol=2e-12)
        counts = np.zeros((3, 3, 2, 2))
        counts[1, 1, 1, 1] = 1
        assert_allclose(result.loglikelihood(counts), expected, atol=2e-12)
    baseline = u2oet_standardize([4, 5, 6], 0.4)
    assert_allclose(u2oet_standardize(np.array([4, 5, 6]) * 1e300, 0.4), baseline)
    assert_allclose(u2oet_standardize(np.array([4, 5, 6]) * 1e-300, 0.4), baseline)
    assert_allclose(u2oet_standardize([4, 5, 6]), np.array([4, 5, 6]) / 5)
    for phi in (1e-300, 1e300):
        m = U2OETMarginal([0], [[1, 1]], link=phi)
        p = u2oet_probabilities([1, 2, 3], [1, 2, 3], efficacy=m, toxicity=m)
        assert_allclose(logsumexp(p.log_joint, axis=(-2, -1)), 0, atol=2e-15)
        assert np.all(np.isfinite(p.log_joint))
        if phi < 1:
            assert_allclose(p.log_efficacy[1, 1, 0], -1, atol=1e-15)
    with pytest.raises(ValueError, match="strictly increasing"):
        u2oet_standardize([2, 1])
    with pytest.raises(ValueError, match="positive"):
        U2OETMarginal([0], [[-1, 1]])
    with pytest.raises(ValueError, match="CMI"):
        u2oet_probabilities(
            [1, 2],
            [1, 2],
            efficacy=U2OETMarginal([0], [[1, 1]], (2, 1)),
            toxicity=marginal,
            model="cmi",
        )
