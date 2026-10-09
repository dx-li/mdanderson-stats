import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import rmc_compass_reported_prediction

_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/rmc-compass-public-reference.json").read_text()
)


@pytest.mark.parametrize("case", _REFERENCE["cases"])
def test_live_app_predictions_are_consistent_with_reported_parameter_precision(case):
    result = rmc_compass_reported_prediction(
        case["ecog"],
        case["sites"],
        neutrophil_lymphocyte_ratio=case["nlr"],
        corrected_calcium_mg_dl=case["corrected_calcium_mg_dl"],
        times_months=case["times_months"],
    )
    # Native outputs are themselves rounded. This is consistency with public
    # precision, not equality to an unavailable full-precision fitted model.
    native = np.array(case["survival_probabilities"])
    envelope = result.survival_parameter_rounding_envelope
    assert np.all(envelope[:, 0] <= native + 0.0005)
    assert np.all(envelope[:, 1] >= native - 0.0005)
    median_low, median_high = result.median_parameter_rounding_envelope
    assert median_low <= case["median_months"] + 0.05
    assert median_high >= case["median_months"] - 0.05
    assert result.risk_group == case["risk_group"]


def test_times_zero_order_tail_complements_and_immutable_results():
    result = rmc_compass_reported_prediction(
        1,
        0,
        neutrophil_lymphocyte_ratio=5 / 1.5,
        corrected_calcium_mg_dl=9.5,
        times_months=[36, 0, 12, 12],
    )
    assert result.survival_probability[1] == 1
    assert result.death_probability[1] == 0
    assert result.survival_probability[2] == result.survival_probability[3]
    assert result.survival_probability[0] < result.survival_probability[2]
    np.testing.assert_allclose(result.survival_probability + result.death_probability, 1)
    assert not result.times_months.flags.writeable
    assert not result.survival_probability.flags.writeable
    assert not result.survival_parameter_rounding_envelope.flags.writeable
    assert result.risk_group_stable_under_parameter_rounding


def test_clamping_and_extrapolation_remain_visible():
    result = rmc_compass_reported_prediction(
        4,
        11,
        neutrophil_lymphocyte_ratio=0,
        corrected_calcium_mg_dl=1e308,
    )
    assert result.effective_neutrophil_lymphocyte_ratio == 0.84
    assert result.effective_corrected_calcium_mg_dl == 11.2
    assert result.extrapolated_predictors == ("ecog_performance_status", "disease_sites")
    assert result.clamped_predictors == (
        "neutrophil_lymphocyte_ratio",
        "corrected_calcium_mg_dl",
    )
    assert result.neutrophil_lymphocyte_ratio == 0
    assert result.corrected_calcium_mg_dl == 1e308


def test_rounding_can_make_the_24_month_risk_classification_unstable():
    # The public formula crosses the Low/Intermediate cutoff at this calcium.
    result = rmc_compass_reported_prediction(
        1,
        0,
        neutrophil_lymphocyte_ratio=5 / 1.5,
        corrected_calcium_mg_dl=8.9835,
    )
    assert not result.risk_group_stable_under_parameter_rounding


@pytest.mark.parametrize(
    "field,value",
    [
        ("ecog_performance_status", True),
        ("ecog_performance_status", 1.5),
        ("ecog_performance_status", 5),
        ("disease_sites", -1),
        ("disease_sites", 12),
        ("neutrophil_lymphocyte_ratio", np.inf),
        ("neutrophil_lymphocyte_ratio", -1),
        ("corrected_calcium_mg_dl", "9.5"),
        ("corrected_calcium_mg_dl", [9.5]),
        ("times_months", []),
        ("times_months", [-1]),
        ("times_months", [37]),
        ("times_months", [np.nan]),
        ("times_months", [[6, 12]]),
    ],
)
def test_invalid_covariates_and_unsupported_horizons_are_rejected(field, value):
    arguments = dict(
        ecog_performance_status=1,
        disease_sites=0,
        neutrophil_lymphocyte_ratio=3,
        corrected_calcium_mg_dl=9.5,
    )
    arguments[field] = value
    with pytest.raises(ValueError):
        rmc_compass_reported_prediction(**arguments)


def test_scalar_time_and_output_limit():
    args = dict(neutrophil_lymphocyte_ratio=3, corrected_calcium_mg_dl=9.5)
    assert rmc_compass_reported_prediction(
        1, 0, times_months=np.array(12), **args
    ).times_months.shape == (1,)
    with pytest.raises(ValueError, match="exceeds"):
        rmc_compass_reported_prediction(1, 0, times_months=np.zeros(100_001), **args)
