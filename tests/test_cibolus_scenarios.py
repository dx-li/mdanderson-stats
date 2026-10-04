"""Focused source-contract tests for CiBolus interpolated scenario truths."""

import numpy as np
import pytest

from mdanderson_stats.cibolus_scenarios import cibolus_interpolated_truth


def test_power_profile_builds_bolus_intervals_failure_and_conditional_toxicity() -> None:
    truth = cibolus_interpolated_truth(
        [0.2],
        [0.1],
        [0.25, 0.5, 1.0],
        response_zero=0.1,
        response_one=0.7,
        toxicity_zero=0.2,
        toxicity_one=0.6,
        toxicity_failure=0.8,
        response_curve="below_linear",
    )
    masses = np.array([0.1, 0.0375, 0.1125, 0.45, 0.3])
    toxicities = np.array([0.2, 0.3, 0.4, 0.6, 0.8])
    expected = np.stack((masses * (1 - toxicities), masses * toxicities), axis=-1)
    np.testing.assert_allclose(truth[0, 0], expected, rtol=0, atol=2e-16)
    assert not truth.flags.writeable


def test_s_shaped_profiles_and_regimen_broadcasting() -> None:
    truth = cibolus_interpolated_truth(
        [0.1, 0.2],
        [0.0, 0.5],
        [0.25, 0.5, 0.75, 1.0],
        response_zero=[np.array([0.05]), np.array([0.1])],
        response_one=[0.65, 0.85],
        toxicity_zero=0.1,
        toxicity_one=[[0.5], [0.7]],
        toxicity_failure=0.9,
        response_curve="s_shaped",
        toxicity_curve="s_shaped",
    )
    assert truth.shape == (2, 2, 6, 2)
    np.testing.assert_allclose(truth.sum(axis=(-2, -1)), 1.0, rtol=0, atol=3e-16)
    # The source S profile is quadratic to .5 and square-root thereafter.
    cumulative = np.cumsum(truth[0, 0, 1:-1].sum(axis=-1))
    np.testing.assert_allclose(
        cumulative,
        (0.65 - 0.05) * np.array([0.125, 0.5, 0.5 + np.sqrt(0.5) / 2, 1.0]),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response_zero": 0.6, "response_one": 0.5},
        {"response_zero": -0.1, "response_one": 0.5},
        {"response_zero": 0.1, "response_one": 0.5, "toxicity_failure": 1.1},
        {"response_zero": 0.1, "response_one": 0.5, "response_curve": "cubic"},
    ],
)
def test_invalid_probability_order_domain_or_curve_is_rejected(kwargs: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "response_zero": 0.1,
        "response_one": 0.5,
        "toxicity_zero": 0.2,
        "toxicity_one": 0.3,
        "toxicity_failure": 0.4,
        **kwargs,
    }
    with pytest.raises(ValueError):
        cibolus_interpolated_truth([0.2], [0.1], [0.5, 1.0], **arguments)  # type: ignore[arg-type]


def test_zero_width_response_has_only_failure_and_bolus_atoms() -> None:
    truth = cibolus_interpolated_truth(
        [0.2],
        [0.1],
        [0.25, 1.0],
        response_zero=0.3,
        response_one=0.3,
        toxicity_zero=0.0,
        toxicity_one=1.0,
        toxicity_failure=0.5,
    )
    np.testing.assert_array_equal(truth[0, 0, 1:-1], 0.0)
    np.testing.assert_allclose(truth[0, 0, 0], [0.3, 0.0])
    np.testing.assert_allclose(truth[0, 0, -1], [0.35, 0.35])


def test_interpolation_preserves_rare_endpoint_cells() -> None:
    truth = cibolus_interpolated_truth(
        [0.2],
        [0.1],
        [1.0],
        response_zero=0.0,
        response_one=0.5,
        toxicity_zero=0.5,
        toxicity_one=1e-300,
        toxicity_failure=1.0 - 1e-14,
    )
    assert truth[0, 0, 1, 1] > 0.0
    assert truth[0, 0, -1, 0] > 0.0
