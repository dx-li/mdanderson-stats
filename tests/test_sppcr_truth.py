"""Truth designs preserve native units and normalize without overflowing."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import (
    RandlibGenerator,
    sppcr_bootstrap,
    sppcr_generate,
    sppcr_generate_legacy,
    sppcr_truth,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/sppcr_truth.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_truth_generation(case):
    truth = sppcr_truth(
        **{k: case[k] for k in ("dna", "wells", "weights", "calibration")}, progenitor=(0, 0)
    )
    assert_allclose(truth.frequency, case["frequency"], rtol=2e-16)
    samples = sppcr_generate_legacy(
        truth.probability, truth.wells, rng=RandlibGenerator(), replicates=3
    )
    expected = np.array(case["seen"]).reshape(samples.seen.shape)
    assert_array_equal(samples.seen, expected)


@pytest.mark.parametrize(
    "weights,calibration",
    [([1e308, 1e308, 1e308], 3), ([1e-300, 2e-300, 3e-300], 1.7), ([0, 2, 5], 2)],
)
def test_normalization_against_decimal(weights, calibration):
    truth = sppcr_truth([0.25, 1], 100, weights, calibration, progenitor=(0, 2))
    with localcontext() as ctx:
        ctx.prec = 100
        w = [Decimal.from_float(float(x)) for x in weights]
        expected = [float(x / sum(w)) for x in w]
        probability = [
            [
                float(
                    1
                    - (
                        -Decimal.from_float(d) * Decimal.from_float(float(calibration)) * x / sum(w)
                    ).exp()
                )
                for x in w
            ]
            for d in [0.25, 1.0]
        ]
    assert_allclose(truth.frequency, expected, rtol=3e-16)
    assert_allclose(truth.probability, probability, rtol=5e-16)


@pytest.mark.parametrize(
    "override",
    [
        {"dna": []},
        {"dna": [0]},
        {"dna": [[1]]},
        {"dna": [np.inf]},
        {"weights": []},
        {"weights": [0, 0]},
        {"weights": [-1, 2]},
        {"weights": [[1, 2]]},
        {"weights": [np.nan, 2]},
        {"calibration": 0},
        {"calibration": -1},
        {"calibration": [1]},
        {"calibration": np.inf},
        {"wells": 0},
        {"wells": 1.5},
        {"wells": [1, 2, 3]},
        {"progenitor": (0, 2)},
        {"progenitor": (-1, 0)},
        {"progenitor": (0,)},
        {"progenitor": (0, 0.5)},
    ],
)
def test_invalid_design(override):
    args = dict(dna=[1, 2], wells=20, weights=[2, 3], calibration=1, progenitor=(0, 1))
    args.update(override)
    with pytest.raises(ValueError):
        sppcr_truth(**args)


@pytest.mark.parametrize("weights,calibration", [([1e308, 1e-300], 1), ([1, 1], 5e-324)])
def test_positive_truth_underflow_is_explicit(weights, calibration):
    with pytest.raises(ArithmeticError, match="underflows"):
        sppcr_truth([1], 20, weights, calibration, progenitor=(0, 1))


def test_owned_immutable_design_and_variable_wells():
    dna, weights, wells = np.array([1.0, 2.0]), np.array([1.0, 3.0]), np.array([20.0, 40.0])
    truth = sppcr_truth(dna, wells, weights, 2, progenitor=(1, 1))
    dna[:] = weights[:] = wells[:] = 0
    assert_array_equal(truth.dna, [1, 2])
    assert_array_equal(truth.wells, [20, 40])
    assert_array_equal(truth.mu, [0.5, 1.5])
    for a in (truth.dna, truth.wells, truth.frequency, truth.mu, truth.probability):
        with pytest.raises(ValueError):
            a.setflags(write=True)


def test_truth_sample_then_bootstrap_from_truth_or_observation():
    truth = sppcr_truth([0.25, 1, 2], 100, [4, 5, 1], 1.5, progenitor=(0, 1))
    first = sppcr_generate(truth.probability, truth.wells, rng=np.random.default_rng(42))
    common = dict(progenitor=truth.progenitor, replicates=20)
    from_truth = sppcr_bootstrap(
        truth.dna,
        first.seen[0],
        truth.wells,
        probability=truth.probability,
        rng=np.random.default_rng(7),
        **common,
    )
    from_observed = sppcr_bootstrap(
        truth.dna, first.seen[0], truth.wells, rng=np.random.default_rng(7), **common
    )
    assert_array_equal(from_truth.samples.probability, truth.probability)
    assert_array_equal(from_observed.samples.probability, first.seen[0] / 100)
    assert_array_equal(from_truth.observed_fit.dna, truth.dna)
    assert_allclose(from_truth.summary.frequency.values.sum(axis=-1), 1)
