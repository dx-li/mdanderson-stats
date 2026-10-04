import numpy as np
import pytest

from mdanderson_stats.asypow import asypow_information
from mdanderson_stats.asypow_design import asypow_design_information
from mdanderson_stats.asypow_groups import asypow_group_information
from mdanderson_stats.asypow_multinomial import asypow_multinomial_information
from mdanderson_stats.asypow_ordinal import (
    asypow_ordinal_information,
    asypow_ordinal_regression_information,
)
from mdanderson_stats.asypow_regression import asypow_regression_information
from mdanderson_stats.asypow_smo import asypow_smo_binomial
from mdanderson_stats.asypow_workflow import AsyPowCalculationRequest, asypow_calculate


@pytest.mark.parametrize("missing", ["significance", "power", "sample_size"])
def test_group_lr_workflow_constructs_model_and_calculates_each_target(missing):
    settings = {
        "parameters": np.array([0.25, 0.48]),
        "contrasts": np.array([1.0, -1.0]),
    }
    targets = {
        "significance": np.array([0.04, 0.05]),
        "power": np.array([0.8, 0.9]),
        "sample_size": np.array([60.0, 90.0]),
    }
    supplied = {key: value for key, value in targets.items() if key != missing}
    result = asypow_calculate(AsyPowCalculationRequest("lr_groups", settings, **supplied))

    fit = asypow_information(
        settings["parameters"],
        asypow_group_information(settings["parameters"]),
        settings["contrasts"],
    )
    expected = getattr(fit, missing)(
        targets["sample_size"] if missing != "sample_size" else targets["power"],
        targets["power"] if missing == "significance" else targets["significance"],
    )
    np.testing.assert_allclose(getattr(result, missing), expected)
    assert result.calculated == missing
    assert result.method == "LR"
    assert not result.effective_settings[0][1].flags.writeable
    report = result.report()
    assert "Effective settings (defaults included)" in report
    assert "Significance\tPower\tSample size" in report
    assert "group_size = 1" in report


def test_smo_route_captures_effective_defaults_and_preserves_scalar_result():
    result = asypow_calculate(
        AsyPowCalculationRequest(
            "asypow_smo_binomial",
            {"probabilities": [0.25, 0.45], "group_size": [2, 1]},
            power=0.8,
            significance=0.05,
        )
    )
    direct = asypow_smo_binomial([0.25, 0.45], group_size=[2, 1])
    assert isinstance(result.sample_size, float)
    assert result.sample_size == pytest.approx(direct.sample_size(0.8, 0.05))
    assert "subtract_df = True" in result.report()
    assert "group_size = (2, 1)" in result.report()


@pytest.mark.parametrize(
    ("procedure", "settings", "theta", "information", "contrast"),
    [
        (
            "lr_regression",
            {"parameters": [[-1.0, 0.4], [-0.3, 0.7]], "covariates": [-1, 0, 1]},
            np.array([-1.0, 0.4, -0.3, 0.7]),
            asypow_regression_information([[-1.0, 0.4], [-0.3, 0.7]], [-1, 0, 1]),
            [1, 0, -1, 0],
        ),
        (
            "lr_ordinal",
            {"cumulative": [0.2, 0.55]},
            np.array([0.2, 0.55]),
            asypow_ordinal_information([0.2, 0.55]),
            [1, -1],
        ),
        (
            "lr_ordinal_regression",
            {"parameters": [[-1.0, 0.4], [-0.3, 0.7]], "covariates": [-1, 0, 1]},
            np.array([-1.0, 0.4, -0.3, 0.7]),
            asypow_ordinal_regression_information([[-1.0, 0.4], [-0.3, 0.7]], [-1, 0, 1]),
            [1, 0, -1, 0],
        ),
        (
            "lr_multinomial",
            {"probabilities": [[0.2, 0.3], [0.3, 0.4]]},
            np.array([0.2, 0.3, 0.3, 0.4]),
            asypow_multinomial_information([[0.2, 0.3], [0.3, 0.4]]),
            [1, 0, -1, 0],
        ),
        (
            "lr_design",
            {"coefficients": [0.3, 0.1], "design": [[1, -1], [1, 0], [1, 1]]},
            np.array([0.3, 0.1]),
            asypow_design_information([0.3, 0.1], [[1, -1], [1, 0], [1, 1]]),
            [0, 1],
        ),
    ],
)
def test_typed_lr_routes_preserve_existing_parameter_and_information_order(
    procedure, settings, theta, information, contrast
):
    workflow_settings = {**settings, "contrasts": contrast}
    result = asypow_calculate(
        AsyPowCalculationRequest(
            procedure,
            workflow_settings,
            power=0.8,
            significance=0.05,
        )
    )
    direct = asypow_information(theta, information, contrast)
    assert result.model_result.degrees_of_freedom == direct.degrees_of_freedom
    assert result.model_result.noncentrality_per_observation == pytest.approx(
        direct.noncentrality_per_observation
    )
    np.testing.assert_allclose(result.model_result.null_parameters, direct.null_parameters)


def test_target_validation_and_callback_report_limitations_are_explicit():
    with pytest.raises(ValueError, match="exactly two"):
        asypow_calculate(
            AsyPowCalculationRequest("lr_groups", {"parameters": [0.2, 0.4], "contrasts": [1, -1]})
        )
    with pytest.raises(TypeError):
        asypow_calculate(
            AsyPowCalculationRequest(
                "lr_groups",
                {"parameters": [0.2, 0.4], "contrasts": [1, -1], "unknown": 1},
                power=0.8,
                significance=0.05,
            )
        )

    def expected_log_likelihood(alternative, candidate):
        return -float(np.sum((candidate - alternative) ** 2))

    callback_result = asypow_calculate(
        AsyPowCalculationRequest(
            "asypow_smo_generic",
            {
                "parameters": [0.2],
                "expected_log_likelihood": expected_log_likelihood,
                "lower": [0.0],
                "upper": [1.0],
                "constraints": [[1, 1, 0.5]],
            },
            power=0.8,
            significance=0.05,
        )
    )
    assert "closure state not captured" in callback_result.report()
