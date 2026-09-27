from dataclasses import replace

import numpy as np
import pytest

from mdanderson_stats.dose_schedule import (
    DoseSchedulePatient,
    dose_schedule_cumulative_hazard,
    dose_schedule_hazard,
    dose_schedule_parameter_names,
    dose_schedule_patient_loglikelihood,
)
from mdanderson_stats.dose_schedule_decision import dose_schedule_decision
from mdanderson_stats.dose_schedule_fit import fit_dose_schedule
from mdanderson_stats.dose_schedule_prior import DoseSchedulePrior, dose_schedule_moment_prior


def test_triangular_hazard_and_variable_dose_patient_history() -> None:
    times = np.array([-1.0, 0.0, 1.0, 2.0, 4.0, 5.0])
    hazard = dose_schedule_hazard(times, 3.0, 2.0, 3.0)
    cumulative = dose_schedule_cumulative_hazard(times, 3.0, 2.0, 3.0)
    np.testing.assert_allclose(hazard, [0, 0, 0.6, 1.2, 0.4, 0])
    np.testing.assert_allclose(cumulative, [0, 0, 0.3, 1.2, 2.8, 3])

    patient = DoseSchedulePatient(3.0, True, [0.0, 1.0], [0, 1])
    loglikelihood = dose_schedule_patient_loglikelihood(patient, [1.0, 2.0], [1.0, 2.0], [2.0, 2.0])
    expected = -2.0
    assert loglikelihood == pytest.approx(expected)


def test_moment_prior_matches_published_parameterization() -> None:
    prior = dose_schedule_moment_prior([0.2, 0.25, 0.3], 5, [18, 14, 10], [10, 14, 18])
    np.testing.assert_allclose(prior.sd**2, np.log(3))
    np.testing.assert_allclose(
        prior.mean, [-3.66, 2.34, 1.75, -4.90, 2.09, 2.09, -4.83, 1.75, 2.34], atol=0.02
    )
    assert dose_schedule_parameter_names(1) == ("log_area_increment.1", "log_peak.1", "log_tail.1")


def test_prior_only_fit_and_safety_screened_final_decision() -> None:
    prior = DoseSchedulePrior(
        [-2.0, 0.0, 0.0, -1.5, 0.0, 0.0],
        [0.3, 0.0, 0.0, 0.3, 0.0, 0.0],
        2,
    )
    fit = fit_dose_schedule(
        [],
        prior,
        [[0.0], [0.0, 1.0]],
        5.0,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(4),
    )
    assert fit.direct_prior
    assert fit.log_parameters.shape == (2, 8, 6)
    assert fit.regimen_risk.shape == (2, 8, 2, 2)
    assert fit.likelihood_evaluations == 0

    synthetic = np.broadcast_to(
        np.array([[0.20, 0.30], [0.40, 0.50]])[None, None, :, :],
        fit.regimen_risk.shape,
    ).copy()
    fit = replace(fit, regimen_risk=synthetic)
    treated = np.array([[1, 0], [0, 0]])
    interim = dose_schedule_decision(
        fit,
        treated,
        toxicity_limit=0.45,
        upper_probability=0.1,
        target=0.45,
        current=(0, 0),
    )
    assert interim.action == "treat"
    assert interim.pair == (1, 0)
    final = dose_schedule_decision(
        fit,
        treated,
        toxicity_limit=0.45,
        upper_probability=0.1,
        target=0.45,
        final=True,
    )
    assert final.action == "select"
    assert final.pair == (1, 0)
    assert not final.eligible[1, 1]

    observed = fit_dose_schedule(
        [
            DoseSchedulePatient(1.0, True, [0.0], [0]),
            DoseSchedulePatient(2.0, False, [0.0, 1.0], [0, 1]),
        ],
        prior,
        [[0.0], [0.0, 1.0]],
        5.0,
        draws=8,
        warmup=2,
        chains=2,
        rng=np.random.default_rng(9),
    )
    assert not observed.direct_prior
    assert observed.likelihood_evaluations > 2 * (8 + 2 + 1)
    assert observed.regimen_risk.shape == (2, 8, 2, 2)
    assert np.all(np.isfinite(observed.log_likelihood))


def test_schedule_lengths_must_grow() -> None:
    prior = DoseSchedulePrior([-1.0, 0.0, 0.0], [0.1, 0.0, 0.0], 1)
    with pytest.raises(ValueError, match="lengths must increase"):
        fit_dose_schedule([], prior, [[0.0, 2.0], [0.0, 2.0]], 5.0, rng=np.random.default_rng(1))
