from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

from mdanderson_stats.bard_response import bard_response_probabilities
from mdanderson_stats.bard_response_scenarios import (
    BARDResponseScenario,
    bard_response_scenario,
)


def test_published_scenario_tables_are_captured_as_immutable_inputs() -> None:
    expected = {
        "five-dose-1": (
            (0.12, 0.25, 0.42, 0.49, 0.55),
            (0.181, 0.349, 0.439, 0.519, 0.596),
            (-2.197, -1.099, -0.619, -0.201, 0.201),
        ),
        "five-dose-2": (
            (0.04, 0.12, 0.25, 0.43, 0.63),
            (0.152, 0.181, 0.349, 0.439, 0.519),
            (-2.442, -2.197, -1.099, -0.619, -0.201),
        ),
        "five-dose-3": (
            (0.02, 0.06, 0.10, 0.25, 0.40),
            (0.103, 0.152, 0.181, 0.349, 0.439),
            (-2.944, -2.442, -2.197, -1.099, -0.619),
        ),
        "five-dose-4": (
            (0.02, 0.05, 0.08, 0.11, 0.25),
            (0.046, 0.103, 0.152, 0.181, 0.349),
            (-3.892, -2.944, -2.442, -2.197, -1.099),
        ),
        "five-dose-5": (
            (0.12, 0.25, 0.42, 0.49, 0.55),
            (0.349, 0.349, 0.359, 0.359, 0.359),
            (-1.099, -1.099, -1.046, -1.046, -1.046),
        ),
        "five-dose-6": (
            (0.04, 0.12, 0.25, 0.43, 0.63),
            (0.181, 0.349, 0.349, 0.359, 0.359),
            (-2.197, -1.099, -1.099, -1.046, -1.046),
        ),
        "five-dose-7": (
            (0.02, 0.06, 0.10, 0.25, 0.40),
            (0.152, 0.181, 0.349, 0.349, 0.359),
            (-2.442, -2.197, -1.099, -1.099, -1.046),
        ),
        "five-dose-8": (
            (0.02, 0.05, 0.08, 0.11, 0.25),
            (0.103, 0.152, 0.181, 0.349, 0.349),
            (-2.944, -2.442, -2.197, -1.099, -1.099),
        ),
        "three-dose-1": (
            (0.12, 0.25, 0.40),
            (0.181, 0.349, 0.439),
            (-2.197, -1.099, -0.619),
        ),
        "three-dose-2": (
            (0.04, 0.12, 0.25),
            (0.152, 0.181, 0.349),
            (-2.442, -2.197, -1.099),
        ),
        "three-dose-3": (
            (0.12, 0.25, 0.42),
            (0.349, 0.349, 0.359),
            (-1.099, -1.099, -1.046),
        ),
        "three-dose-4": (
            (0.04, 0.12, 0.25),
            (0.181, 0.349, 0.349),
            (-2.197, -1.099, -1.099),
        ),
    }
    assert len(expected) == 12
    for scenario_id, (toxicity, response, intercepts) in expected.items():
        scenario = bard_response_scenario(scenario_id)
        assert scenario.toxicity_probabilities == toxicity
        assert scenario.published_response_probabilities == response
        assert scenario.response_intercepts == intercepts
        assert scenario.response_coefficients == (1.7, -1.5, 0.4)
        assert scenario.factor_levels == (1, 2)
        assert scenario.factor_level_probabilities == (0.5, 0.5)
        assert scenario.factor_joint_distribution == "not specified by source"
        assert scenario.dose_levels == tuple(range(1, len(toxicity) + 1))

    five = bard_response_scenario("five-dose-2")
    with pytest.raises(FrozenInstanceError):
        five.scenario_id = "changed"  # type: ignore[misc]


def test_factory_rejects_unknown_ids_and_mutable_vectors() -> None:
    with pytest.raises(ValueError, match="unknown"):
        bard_response_scenario("five-dose-9")
    with pytest.raises(TypeError, match="immutable tuples"):
        BARDResponseScenario(
            scenario_id="test",
            dose_levels=[1, 2, 3],  # type: ignore[arg-type]
            toxicity_probabilities=(0.1, 0.2, 0.3),
            published_response_probabilities=(0.1, 0.2, 0.3),
            response_intercepts=(-2.0, -1.0, -0.5),
        )


@pytest.mark.parametrize("scenario_id", ("five-dose-1", "three-dose-4"))
def test_intercepts_approximately_reproduce_printed_margins_under_product_profiles(
    scenario_id: str,
) -> None:
    scenario = bard_response_scenario(scenario_id)
    profiles = np.asarray(tuple(product(scenario.factor_levels, repeat=3)), dtype=np.int64)
    odds_ratios = np.column_stack(
        (np.ones(3), np.exp(np.asarray(scenario.response_coefficients, dtype=np.float64)))
    )
    conditional = bard_response_probabilities(scenario.response_intercepts, profiles, odds_ratios)
    equally_weighted_margins = conditional.mean(axis=1)
    # Intercepts and published marginal rates are each rounded to 0.001.
    # The 0.000625 bound covers 0.0005 printed-rate rounding plus at most
    # 0.000125 from intercept rounding (0.0005 / 4, the logistic slope bound).
    # Product weights are this check's explicit convention; joint factor
    # dependence is not specified in the paper's scenario description.
    np.testing.assert_allclose(
        equally_weighted_margins,
        scenario.published_response_probabilities,
        rtol=0,
        atol=6.25e-4,
    )
