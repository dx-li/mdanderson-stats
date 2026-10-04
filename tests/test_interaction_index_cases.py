import json
import math
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    fit_median_effect,
    interaction_index_pooled_error,
    interaction_index_ray,
)

CASES = Path(__file__).parent / "fixtures" / "interaction_index_cases.json"


def test_published_case_data_reproduce_rounded_median_effect_summaries():
    document = json.loads(CASES.read_text(encoding="utf-8"))

    for case in document["cases"]:
        dose_groups = [*case["single_agent_doses"], case["combination_total_doses"]]
        response_groups = [*case["single_agent_responses"], case["combination_responses"]]
        assert len(dose_groups) == len(response_groups) == len(case["printed_fits"]) == 3
        fits = []
        for dose, response, printed, reference in zip(
            dose_groups,
            response_groups,
            case["printed_fits"],
            case["reference_fits"],
            strict=True,
        ):
            fit = fit_median_effect(dose, response)
            fits.append(fit)
            assert fit.intercept == pytest.approx(reference["intercept"], abs=1e-11)
            assert fit.slope == pytest.approx(reference["slope"], abs=1e-11)
            assert math.exp(-fit.intercept / fit.slope) == pytest.approx(
                reference["median_dose"], abs=1e-11
            )
            assert fit.residual_variance**0.5 == pytest.approx(reference["residual_sd"], abs=1e-11)
            assert math.sqrt(fit.covariance[0, 0]) == pytest.approx(
                reference["intercept_se"], abs=1e-11
            )
            assert math.sqrt(fit.covariance[1, 1]) == pytest.approx(
                reference["slope_se"], abs=1e-11
            )
            assert fit.intercept == pytest.approx(printed["intercept"], abs=0.0005)
            assert fit.slope == pytest.approx(printed["slope"], abs=0.0005)
            assert math.exp(-fit.intercept / fit.slope) == pytest.approx(
                printed["median_dose"], abs=0.0005
            )
            assert fit.residual_variance**0.5 == pytest.approx(printed["residual_sd"], abs=0.0005)

        proportions = np.asarray(case["combination_proportions"], dtype=float)
        total = np.asarray(case["combination_total_doses"], dtype=float)
        component_doses = total[:, None] * proportions / proportions.sum()
        observed = interaction_index_pooled_error(
            fits[:2], component_doses, case["combination_responses"]
        )
        reference_observed = case["reference_observed"]
        np.testing.assert_allclose(observed.log_index, reference_observed["log_index"], atol=1e-11)
        np.testing.assert_allclose(
            observed.log_standard_error, reference_observed["log_standard_error"], atol=1e-11
        )
        np.testing.assert_allclose(
            observed.interval,
            np.asarray(reference_observed["interval"]),
            atol=1e-11,
        )
        if case["printed_observed_indices"] is not None:
            np.testing.assert_allclose(observed.index, case["printed_observed_indices"], atol=0.001)
            np.testing.assert_allclose(
                observed.interval, case["printed_observed_intervals"], atol=0.001
            )

        reference_ray = case["reference_ray"]
        ray = interaction_index_ray(fits[:2], fits[2], proportions, reference_ray["effect"])
        np.testing.assert_allclose(ray.log_index, reference_ray["log_index"], atol=1e-11)
        np.testing.assert_allclose(
            ray.log_standard_error, reference_ray["log_standard_error"], atol=1e-11
        )
        np.testing.assert_allclose(ray.interval, np.asarray(reference_ray["interval"]), atol=1e-11)
