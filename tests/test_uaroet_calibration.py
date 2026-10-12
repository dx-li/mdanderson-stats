"""Pseudo-trial contracts, native parameter ordering, and fail-closed calibration."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mdanderson_stats.uaroet_calibration as calibration
from mdanderson_stats import UAROETDesktopScenario, read_uaroet_desktop_study
from mdanderson_stats.uaroet_desktop_study import uaroet_desktop_prior_coordinates


def inputs():
    s = read_uaroet_desktop_study(
        (Path(__file__).parent / "fixtures/uaroet-python-study.json").read_text()
    )
    s = replace(s, simulation_parameters={"NumChains": 2, "BurnInSamples": 0})
    scenario = UAROETDesktopScenario(
        3, np.array([[0.7, 0.3], [0.5, 0.5]]), np.array([[0.9, 0.1], [0.7, 0.3]]), 0.2
    )
    return s, scenario


def test_pseudo_trial_design_and_average(monkeypatch):
    s, scenario = inputs()
    seen = []
    callbacks = []

    def fit(pseudo, **kwargs):
        counts, partial, assigned = pseudo.outcome_counts()
        np.testing.assert_array_equal(assigned, 100)
        np.testing.assert_array_equal(partial, 0)
        np.testing.assert_array_equal(pseudo.prior_mean, 0)
        np.testing.assert_array_equal(pseudo.prior_sd, 100)
        assert kwargs["initial_association"] == 0
        assert np.all(kwargs["initial_parameters"][[1, 3]] > 0)
        seen.append(counts)
        return SimpleNamespace(
            precision_met=True,
            utility_split_rhat=np.array([1.0, 1.0]),
            mean_parameters=np.full(4, len(seen), dtype=float),
            mean_association=0.1 * len(seen),
            utility_mcse_ratio=np.full((2, 2), 0.02),
            monitored_doses=(0, 1),
            draws_per_chain=1000,
        )

    monkeypatch.setattr(calibration, "fit_uaroet_desktop", fit)
    result = calibration.calibrate_uaroet_prior(
        s,
        scenario,
        repetitions=3,
        rng=np.random.default_rng(83),
        on_trial=lambda i, fit: callbacks.append(i),
    )
    assert callbacks == [1, 2, 3]
    np.testing.assert_array_equal(result.prior_mean, 2)
    assert not result.meets_guide_replication_minimum
    assert not np.array_equal(seen[0], seen[1])
    applied = result.apply(s)
    np.testing.assert_array_equal(applied.prior_sd, s.prior_sd)
    np.testing.assert_array_equal(applied.prior_mean, 2)
    assert applied.native_association_mean == pytest.approx(0.2)
    values = [float(x) for x in calibration.uaroet_calibrated_prior_text(result).split()]
    theta, rho = uaroet_desktop_prior_coordinates(
        values, dose_count=2, efficacy_levels=2, toxicity_levels=2
    )
    np.testing.assert_array_equal(theta, result.prior_mean)
    assert rho == pytest.approx(0.2)


@pytest.mark.parametrize("E", [2, 3, 4])
@pytest.mark.parametrize("T", [2, 3, 4])
@pytest.mark.parametrize("D", [2, 3, 5])
def test_native_calibrated_parameter_order(D, E, T):
    width = D * (E + T - 2)
    values = np.arange(width, dtype=float) + 0.125
    result = calibration.UAROETCalibration(
        values[None], np.array([0.25]), np.array([0.02]), np.array([1.0]), (1000,), 1, D, E, T
    )
    text = calibration.uaroet_calibrated_prior_text(result)
    restored, rho = uaroet_desktop_prior_coordinates(
        [float(x) for x in text.split()], dose_count=D, efficacy_levels=E, toxicity_levels=T
    )
    np.testing.assert_array_equal(restored, values)
    assert rho == 0.25


@pytest.mark.parametrize(
    "options",
    [
        {"repetitions": 0},
        {"repetitions": True},
        {"target": True},
        {"target": 0},
        {"maximum_rhat": True},
        {"maximum_rhat": 0.9},
        {"maximum_rhat": float("nan")},
        {"chunk_draws": 7},
        {"max_draws": 8, "chunk_draws": 16},
        {"max_evaluations_per_trial": 0},
        {"on_trial": 3},
    ],
)
def test_invalid_calibration_does_not_advance_rng(options):
    s, scenario = inputs()
    rng = np.random.default_rng(3)
    before = rng.bit_generator.state
    kwargs = {"repetitions": 1, **options}
    with pytest.raises(ValueError):
        calibration.calibrate_uaroet_prior(s, scenario, rng=rng, **kwargs)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("precision,rhat", [(False, 1.0), (True, 1.2), (True, float("nan"))])
def test_failed_trial_never_returns_partial_prior(monkeypatch, precision, rhat):
    s, scenario = inputs()
    monkeypatch.setattr(
        calibration,
        "fit_uaroet_desktop",
        lambda *args, **kwargs: SimpleNamespace(
            precision_met=precision, utility_split_rhat=np.array([rhat])
        ),
    )
    with pytest.raises(RuntimeError, match="pseudo-trial 1"):
        calibration.calibrate_uaroet_prior(s, scenario, repetitions=2, rng=np.random.default_rng(1))


def test_actual_short_diffuse_fit_fails_precision_gate():
    s, scenario = inputs()
    with pytest.raises(RuntimeError, match="pseudo-trial 1"):
        calibration.calibrate_uaroet_prior(
            s,
            scenario,
            repetitions=1,
            rng=np.random.default_rng(12),
            target=1e-6,
            chunk_draws=8,
            max_draws=8,
        )


def test_replication_minimum_is_not_a_clinical_validity_flag():
    result = calibration.UAROETCalibration(
        np.ones((1000, 4)),
        np.zeros(1000),
        np.full(1000, 0.02),
        np.ones(1000),
        (1000,) * 1000,
        1,
        2,
        2,
        2,
    )
    assert result.meets_guide_replication_minimum
