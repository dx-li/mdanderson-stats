"""Fixed-design KSBIN1 operating characteristics and explicit path enumeration."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import KStageBinomial, ksbin1_operating_characteristics

CASES = json.loads((Path(__file__).parent / "fixtures/ksbin1.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_transition_and_expectation(case):
    c = case
    r = ksbin1_operating_characteristics(
        c["totals"], c["critical"], c["quit"], c["probability"], alternative=c["alternative"]
    )
    for name in ["rejection", "quitting", "continuation"]:
        assert_allclose(getattr(r, name), c[name], rtol=5e-13, atol=1e-15)
    assert_allclose(
        [r.expected_sample_size, r.expected_given_rejection, r.expected_given_quitting],
        c["expected"],
        rtol=5e-13,
    )


@pytest.mark.parametrize("p", [0, 0.06, 0.2, 0.7, 1])
@pytest.mark.parametrize("alternative", ["less", "greater"])
def test_exhaustive_paths(p, alternative):
    totals = [2, 4, 6]
    cuts = [0, 1, 2] if alternative == "less" else [2, 3, 4]
    quits = [2, 3] if alternative == "less" else [0, 1]
    r = ksbin1_operating_characteristics(totals, cuts, quits, p, alternative=alternative)
    rej, qt, cont = np.zeros(3), np.zeros(3), np.zeros(3)
    for sequence in itertools.product((0, 1), repeat=6):
        weight = p ** sum(sequence) * (1 - p) ** (6 - sum(sequence))
        for i, n in enumerate(totals):
            k = sum(sequence[:n])
            reject = k <= cuts[i] if alternative == "less" else k >= cuts[i]
            stop = i == 2 or (k >= quits[i] if alternative == "less" else k <= quits[i])
            if reject:
                rej[i] += weight
                break
            if stop:
                qt[i] += weight
                break
            cont[i] += weight
    assert_allclose([r.rejection, r.quitting, r.continuation], [rej, qt, cont], atol=1e-14)
    assert_allclose(r.expected_sample_size, np.sum(np.array(totals) * (rej + qt)), atol=1e-13)
    assert_allclose(np.cumsum(r.rejection + r.quitting) + r.continuation, 1, atol=1e-14)
    if rej.sum() == 0:
        assert np.isnan(r.expected_given_rejection)
    if qt.sum() == 0:
        assert np.isnan(r.expected_given_quitting)


def test_broadcast_probabilities_and_stage_arrival_mass():
    p = np.array([[0.06, 0.2], [0.5, 0.9]])
    result = ksbin1_operating_characteristics([14, 28, 42], [0, 1, 3], [3, 4], p)
    assert result.rejection.shape == (2, 2, 3)
    for i in range(3):
        mass = result.design.stage_distribution(i + 1, p)
        expected = np.ones(p.shape) if i == 0 else result.continuation[..., i - 1]
        assert_allclose(mass.sum(axis=-1), expected, atol=1e-14)
    scalar = ksbin1_operating_characteristics([14, 28, 42], [0, 1, 3], [3, 4], 0.2)
    assert_allclose(result.expected_sample_size[0, 1], scalar.expected_sample_size)


def test_disabled_final_rejection_and_single_stage():
    r = ksbin1_operating_characteristics([5, 10], [-1, -1], [-1], [0, 0.5, 1])
    assert_allclose(r.rejection_probability, 0)
    assert_allclose(r.quitting, [[0, 1]] * 3)
    assert_allclose(r.expected_sample_size, 10)
    assert np.isnan(r.expected_given_rejection).all()
    r = ksbin1_operating_characteristics([10], [3], [], 0.5)
    assert r.expected_sample_size == pytest.approx(10)
    assert r.continuation[0] == 0


@pytest.mark.parametrize(
    "totals,critical,quit,p,alternative",
    [
        ([5], [6], [], 0.5, "less"),
        ([5], [], [], 0.5, "less"),
        ([5], [2.5], [], 0.5, "less"),
        ([5], [2], [], np.nan, "less"),
        ([5], [2], [], 1.1, "less"),
        ([5], [2], [], 0.5, "wrong"),
        ([5, 10], [3, 5], [2], 0.5, "less"),
        ([5, 10], [5, 5], [-1], 0.5, "less"),
        ([5, 10], [1, 5], [], 0.5, "less"),
        ([], [], [], 0.5, "less"),
    ],
)
def test_invalid_design(totals, critical, quit, p, alternative):
    with pytest.raises(ValueError):
        ksbin1_operating_characteristics(totals, critical, quit, p, alternative=alternative)


@pytest.mark.parametrize("stage,p", [(0, 0.5), (True, 0.5), (2, 0.5), (1, -0.1), (1, np.inf)])
def test_invalid_stage_distribution(stage, p):
    with pytest.raises(ValueError):
        KStageBinomial([2], [], []).stage_distribution(stage, p)
