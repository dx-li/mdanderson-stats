"""Saved inputs: explicit repairs, coordinate mapping, and missing-data accounting."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.uaroet_desktop_study import (
    UAROETDesktopScenario,
    read_uaroet_desktop_bundle,
    read_uaroet_desktop_study,
    uaroet_desktop_prior_coordinates,
)


@pytest.fixture
def bundle():
    trial = dict(
        NumPatients=30,
        CohortSize_First=3,
        CohortSize_NewDose=1,
        CohortSize_OldDose=1,
        DoseStart=1,
        LowestUnacceptableToxLevel=1,
        MaxProbTox=0.2,
        UpperAcceptablePiToxCutoff=0.8,
        LowestAcceptableEffLevel=1,
        MinProbEff=0.2,
        UpperAcceptablePiEffCutoff=1,
        MinProbUtilityIsHighest=0.1,
        MinUtilityGoodSet=50,
        UseHybridGreedyArandAssignment=1,
        NumPatientSurplusBeforeRandomize=1,
        RandomizeByMeanUtility=1,
        InitialUtilityCutoffToRandomize=100,
        FinalUtilityCutoffToRandomize=0,
        NumTimesToShrinkUtilityCutoff=10,
        MinDaysToEfficacy=1,
        MaxDaysToEfficacy=42,
        MinDaysToToxicity=1,
        MaxDaysToToxicity=42,
        MeanDaysBetweenArrivals=45,
    )
    return {
        "UAROET_ModelParams.txt": "NumDoses 2\nNumEffLevels 2\nNumToxLevels 2",
        "UAROET_PriorMeans.txt": "0 1 2 3 0.2",
        "UAROET_PriorStdDev.txt": "1",
        "UAROET_DefaultUtility.txt": "1",
        "UAROET_Utility_1.txt": "0 0 30\n0 1 0\n1 0 100\n1 1 50",
        "UAROET_TrialParams.txt": "\n".join(f"{k} {v}" for k, v in trial.items()),
        "UAROET_SimulationParams.txt": "NumChains 4\nBurnInSamples 1000",
        "UAROET_Scenario_1.txt": "0\n1 .8 .2 .9 .1\n2 .5 .5 .7 .3",
    }


def test_bundle_roundtrip_and_pending_counts(bundle):
    study = read_uaroet_desktop_bundle(
        bundle, patient_text="1 1 1 0\n2 2 -1 1\n3 2 1 -1\n4 1 -1 -1"
    )
    assert study.native_association_mean == 0.2
    joint, partial, treated = study.outcome_counts()
    assert joint.sum() == 1 and joint[0, 1, 0] == 1
    assert partial.sum() == 1 and partial[1, 1] == 1
    np.testing.assert_array_equal(treated, [2, 2])
    restored = read_uaroet_desktop_study(study.to_json())
    assert restored.to_json() == study.to_json()
    scenario = study.scenarios[0]
    np.testing.assert_allclose(
        scenario.joint(), scenario.efficacy[:, :, None] * scenario.toxicity[:, None, :]
    )
    assert not study.prior_mean.flags.writeable
    with pytest.raises(TypeError):
        study.trial_parameters["NumPatients"] = 5


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("e", [2, 3, 4])
@pytest.mark.parametrize("t", [2, 3, 4])
def test_prior_coordinates_threshold_baselines_then_dose_increments(d, e, t):
    # Distinct values identify every endpoint/threshold/dose independently.
    blocks = [
        np.array([[1000 * j + 100 * k + i for i in range(d)] for k in range(levels - 1)])
        for j, levels in enumerate((e, t))
    ]
    expected = []
    for block in blocks:
        expected.extend(block[:, 0])
        for dose in range(1, d):
            expected.extend(block[:, dose])
    actual, rho = uaroet_desktop_prior_coordinates(
        np.r_[blocks[0].ravel(), blocks[1].ravel(), 0.25],
        dose_count=d,
        efficacy_levels=e,
        toxicity_levels=t,
    )
    np.testing.assert_array_equal(actual, expected)
    assert rho == 0.25


def test_repairs_require_explicit_choice_and_survive_reopen(bundle):
    bundle["UAROET_TrialParams.txt"] = bundle["UAROET_TrialParams.txt"].replace(
        "ArandAssignment", "ArandAlgorithm"
    )
    with pytest.raises(ValueError, match="repair_example_typo"):
        read_uaroet_desktop_bundle(bundle)
    bundle["UAROET_DefaultUtility.txt"] = "2"
    with pytest.raises(ValueError, match="Utility_2"):
        read_uaroet_desktop_bundle(bundle, repair_example_typo=True)
    study = read_uaroet_desktop_bundle(bundle, repair_example_typo=True, utility_identifier=1)
    assert len(study.input_notes) == 2
    assert read_uaroet_desktop_study(study.to_json()).input_notes == study.input_notes


@pytest.mark.parametrize("value", [True, complex(1), float("nan"), float("inf"), [1], "1"])
@pytest.mark.parametrize("setting", ["NumPatients", "MaxProbTox", "NumChains", "TargetMCSERatio"])
def test_nonreal_settings_rejected_before_comparison(bundle, setting, value):
    study = read_uaroet_desktop_bundle(bundle)
    field = (
        "simulation_parameters"
        if setting in ("NumChains", "TargetMCSERatio")
        else "trial_parameters"
    )
    settings = dict(getattr(study, field))
    settings[setting] = value
    with pytest.raises(ValueError):
        replace(study, **{field: settings})


@pytest.mark.parametrize(
    "text", ["2 1 0 0", "1 0 0 0", "1 1 -2 0", "1 1 0 2", "1 1 0 0\n1 2 1 1", "1 1 0", "1 1.5 0 0"]
)
def test_bad_patient_history_rejected(bundle, text):
    with pytest.raises(ValueError):
        read_uaroet_desktop_bundle(bundle, patient_text=text)


@pytest.mark.parametrize(
    "change",
    [
        {"NumPatients": 0},
        {"NumPatients": 2},
        {"DoseStart": 3},
        {"MaxProbTox": 1.1},
        {"MinDaysToToxicity": 43},
        {"FinalUtilityCutoffToRandomize": 101},
        {"CohortSize_NewDose": 31},
        {"UnknownModelOption": 1},
    ],
)
def test_invalid_design_rejected(bundle, change):
    study = read_uaroet_desktop_bundle(bundle)
    with pytest.raises(ValueError):
        replace(study, trial_parameters=dict(study.trial_parameters) | change)


def test_duplicate_json_and_bool_version_rejected(bundle):
    text = read_uaroet_desktop_bundle(bundle).to_json()
    with pytest.raises(ValueError, match="duplicate"):
        read_uaroet_desktop_study(text.replace('"version": 1', '"version": 1, "version": 1'))
    data = json.loads(text)
    data["version"] = True
    with pytest.raises(ValueError, match="schema"):
        read_uaroet_desktop_study(json.dumps(data))


def test_scenario_inputs_are_copied_and_association_bounded():
    e = np.array([[0.5, 0.5], [0.4, 0.6]])
    s = UAROETDesktopScenario(1, e, e, 0)
    e[:] = 0
    np.testing.assert_array_equal(s.efficacy.sum(1), [1, 1])
    with pytest.raises(ValueError):
        replace(s, association=1)


def test_per_coordinate_sd_file_and_unsupported_model(bundle):
    bundle["UAROET_PriorStdDeviations.txt"] = "1 2 3 4 5"
    np.testing.assert_array_equal(read_uaroet_desktop_bundle(bundle).prior_sd, [1, 2, 3, 4])
    bundle["UAROET_ModelParams.txt"] += "\nUndocumentedModel 1"
    with pytest.raises(ValueError, match="model"):
        read_uaroet_desktop_bundle(bundle)


NATIVE_COUNTS = json.loads(
    (Path(__file__).parent / "fixtures/uaroet-desktop-counts.json").read_text()
)


@pytest.mark.parametrize("case", NATIVE_COUNTS["cases"])
def test_study_count_bookkeeping_matches_original_instructions(bundle, case):
    d, e, t = case["doses"], case["efficacy_levels"], case["toxicity_levels"]
    study = read_uaroet_desktop_bundle(bundle)
    n = d * (e + t - 2)
    settings = dict(study.trial_parameters) | {"NumPatients": 500}
    study = replace(
        study,
        dose_count=d,
        efficacy_levels=e,
        toxicity_levels=t,
        prior_mean=np.zeros(n),
        prior_sd=np.ones(n),
        utility=np.zeros((e, t)),
        trial_parameters=settings,
        scenarios=(),
        patients=case["patients"],
    )
    actual = study.counts()
    expected = case["result"]
    for field, native_field in [
        ("assigned", "assigned"),
        ("complete", "complete"),
        ("likelihood", "likelihood"),
        ("complete_efficacy", "efficacy"),
        ("complete_toxicity", "toxicity"),
    ]:
        np.testing.assert_array_equal(getattr(actual, field), expected[native_field])
        assert not getattr(actual, field).flags.writeable
    assert actual.highest_assigned == expected["highest_assigned"]
    assert actual.legacy_highest_allowed == expected["highest_allowed"]


def test_both_native_distribution_spellings_canonicalize_identically(bundle):
    bundle["UAROET_SimulationParams.txt"] += "\nDistributionSave 1"
    a = read_uaroet_desktop_bundle(bundle)
    bundle["UAROET_SimulationParams.txt"] = bundle["UAROET_SimulationParams.txt"].replace(
        "DistributionSave 1", "DistributionSaveOption 1"
    )
    b = read_uaroet_desktop_bundle(bundle)
    assert a.to_json() == b.to_json()
    assert a.simulation_parameters["DistributionSaveOption"] == 1
    bundle["UAROET_SimulationParams.txt"] += "\nDistributionSave 1"
    with pytest.raises(ValueError, match="duplicate distribution"):
        read_uaroet_desktop_bundle(bundle)
