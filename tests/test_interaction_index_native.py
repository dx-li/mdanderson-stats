from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    fit_median_effect,
    interaction_index_monte_carlo,
    interaction_index_pooled_error,
    interaction_index_ray,
)

FIXTURES = Path(__file__).parent / "fixtures/interaction-index-native"


@pytest.mark.parametrize("case", [1, 2])
def test_unchanged_author_functions_with_unequal_residual_degrees_of_freedom(case):
    inputs = np.genfromtxt(FIXTURES / f"inputs-{case}.csv", delimiter=",", names=True)
    reference = np.genfromtxt(FIXTURES / f"reference-{case}.csv", delimiter=",", names=True)
    observed = np.genfromtxt(FIXTURES / f"observed-{case}.csv", delimiter=",", names=True)
    rows = [inputs[inputs["curve"] == j] for j in (1, 2, 3)]
    fits = [fit_median_effect(row["dose"], row["effect"]) for row in rows]
    ratio = reference["ratio"][0]
    # Original CI.delta omits inverse-mixture-dose squared in its mixture term.
    assert np.max(abs(reference["delta_lower"] - reference["corrected_lower"])) > 0.001
    delta = interaction_index_ray(fits[:2], fits[2], [1, ratio], reference["effect"])
    np.testing.assert_allclose(delta.index, reference["index"], atol=1e-13)
    np.testing.assert_allclose(
        delta.interval,
        np.column_stack((reference["corrected_lower"], reference["corrected_upper"])),
        atol=1e-12,
        rtol=1e-12,
    )
    mixture = rows[2]["dose"][:, None] * np.array([1, ratio])[None, :] / (1 + ratio)
    known = interaction_index_pooled_error(fits[:2], mixture, rows[2]["effect"])
    np.testing.assert_allclose(known.index, observed["index"], atol=1e-13)
    np.testing.assert_allclose(
        known.interval,
        np.column_stack((observed["lower"], observed["upper"])),
        atol=1e-12,
        rtol=1e-12,
    )
    tape = np.loadtxt(FIXTURES / f"draws-{case}.csv", delimiter=",", skiprows=1).reshape(31, 3, 2)
    mc = interaction_index_monte_carlo(
        fits[:2],
        fits[2],
        [1, ratio],
        reference["effect"],
        samples=31,
        coefficient_draws=tape,
    )
    np.testing.assert_allclose(mc.standard_error, reference["mc_sd"], atol=1e-13, rtol=1e-12)
    interval = mc.interval.copy()
    interval[:, 0] = np.maximum(interval[:, 0], 0.0001)
    np.testing.assert_allclose(
        interval,
        np.column_stack((reference["mc_lower"], reference["mc_upper"])),
        atol=1e-12,
        rtol=1e-12,
    )
    tape[:] = 0
    assert np.any(mc.coefficient_draws != 0)
    assert not mc.coefficient_draws.flags.writeable


@pytest.mark.parametrize("bad", [True, "x", complex(1), np.nan, np.inf])
def test_external_draw_tape_validation(bad):
    fits = [fit_median_effect([1, 2, 3, 4], [0.2, 0.3, 0.5, 0.6])] * 3
    with pytest.raises(ValueError, match="coefficient_draws"):
        interaction_index_monte_carlo(
            fits[:2], fits[2], [1, 1], 0.5, samples=2, coefficient_draws=np.full((2, 3, 2), bad)
        )


def test_external_draw_tape_cannot_consume_rng_and_rejects_zero_slopes():
    fits = [fit_median_effect([1, 2, 3, 4], [0.2, 0.3, 0.5, 0.6])] * 3
    tape = np.ones((2, 3, 2))
    generator = np.random.default_rng(12)
    before = generator.bit_generator.state
    with pytest.raises(ValueError, match="cannot be combined"):
        interaction_index_monte_carlo(
            fits[:2], fits[2], [1, 1], 0.5, samples=2, coefficient_draws=tape, rng=generator
        )
    assert before == generator.bit_generator.state
    tape[..., 1] = 0
    with pytest.raises(ArithmeticError, match="zero slope"):
        interaction_index_monte_carlo(
            fits[:2], fits[2], [1, 1], 0.5, samples=2, coefficient_draws=tape
        )
