"""Saved conduct connects inference, explicit profiles and computational holds."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mdanderson_stats.uaroet_desktop_conduct as conduct
from mdanderson_stats import read_uaroet_desktop_study
from mdanderson_stats.uaroet_desktop_rules import UAROETCohortState


def study():
    value = read_uaroet_desktop_study(
        (Path(__file__).parent / "fixtures/uaroet-python-study.json").read_text()
    )
    return replace(value, simulation_parameters={"NumChains": 2, "BurnInSamples": 0})


def posterior(precision=True, rhat=1.0, toxicity=0.0):
    joint = np.zeros((2, 2, 2))
    joint[0, 0, 0] = 1
    joint[1, 1, 0] = 1
    return SimpleNamespace(
        precision_met=precision,
        utility_samples=np.zeros((2, 8, 2)),
        utility_split_rhat=np.array([rhat, rhat]),
        mean_joint=joint,
        toxicity_exceedance=np.full(2, toxicity),
        efficacy_shortfall=np.zeros(2),
        probability_best=np.array([0.1, 0.9]),
    )


def test_initial_assignment_needs_no_posterior(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("unexpected fit")

    monkeypatch.setattr(conduct, "fit_uaroet_desktop", unexpected)
    result = conduct.recommend_uaroet_desktop(
        replace(study(), patients=()), rng=np.random.default_rng(1)
    )
    assert result.action == "treat" and result.selected == 0 and result.posterior is None


def test_legacy_first_cohort_bypass_is_explicit(monkeypatch):
    s = replace(study(), patients=((1, 0, 0, 1),))
    monkeypatch.setattr(
        conduct, "fit_uaroet_desktop", lambda *args, **kwargs: posterior(toxicity=0.9)
    )
    safe = conduct.recommend_uaroet_desktop(s, rng=np.random.default_rng(3))
    legacy = conduct.recommend_uaroet_desktop(s, rng=np.random.default_rng(3), profile="legacy_1_8")
    assert safe.action == "stop" and safe.selected is None
    assert legacy.action == "treat" and legacy.selected == 0 and legacy.posterior is None


@pytest.mark.parametrize(
    "precision,rhat,reason",
    [
        (False, 1.0, "posterior_precision_not_met"),
        (True, 1.2, "posterior_chains_disagree"),
        (True, float("nan"), "posterior_chains_disagree"),
    ],
)
def test_numerical_failures_hold_rather_than_clinically_stop(monkeypatch, precision, rhat, reason):
    monkeypatch.setattr(
        conduct, "fit_uaroet_desktop", lambda *args, **kwargs: posterior(precision, rhat)
    )
    result = conduct.recommend_uaroet_desktop(study(), rng=np.random.default_rng(1))
    assert result.action == "hold" and result.selected is None and result.reason == reason


def test_final_selection_uses_highest_eligible_mean(monkeypatch):
    monkeypatch.setattr(conduct, "fit_uaroet_desktop", lambda *args, **kwargs: posterior())
    result = conduct.recommend_uaroet_desktop(study(), rng=np.random.default_rng(1), final=True)
    assert result.action == "select" and result.selected == 1 and result.status == (1, 1)
    np.testing.assert_array_equal(result.allocation_weights, [0, 1])


def test_state_must_correspond_to_saved_assignments():
    s = study()
    with pytest.raises(ValueError, match="correspond"):
        conduct.recommend_uaroet_desktop(
            s, rng=np.random.default_rng(1), cohort_state=UAROETCohortState((2, 1), 3)
        )


def test_maximum_accrual_requires_final_request():
    s = study()
    s = replace(s, patients=tuple((i + 1, 0, 1, 0) for i in range(30)))
    with pytest.raises(ValueError, match="maximum accrual"):
        conduct.recommend_uaroet_desktop(s, rng=np.random.default_rng(1))


def test_final_request_requires_patients():
    with pytest.raises(ValueError, match="assigned patients"):
        conduct.recommend_uaroet_desktop(
            replace(study(), patients=()), rng=np.random.default_rng(1), final=True
        )


def test_actual_saved_study_conduct_path_with_explicit_demo_tolerances():
    result = conduct.recommend_uaroet_desktop(
        study(),
        rng=np.random.default_rng(92),
        target=0.99,
        maximum_rhat=1000,
        chunk_draws=32,
        max_draws=32,
    )
    assert result.posterior is not None
    if result.selected is not None:
        assert result.status[result.selected] == 1
    assert result.action in ("treat", "hold", "stop")
