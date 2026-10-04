import json

import numpy as np
import pytest

from mdanderson_stats.bard_bf_boin_trial import BARDStageTwoDesign
from mdanderson_stats.bard_study import BARDStudySpecification
from mdanderson_stats.bf_boin import BFBOINDesign


def _study(*, trials=3, seed=731, max_patient_work=100_000, inputs=None):
    if inputs is None:
        profiles = [[1, 1], [1, 2], [2, 1], [2, 2]]
        weights = [0.25] * 4
        odds = [[1, 1.5], [1, 0.8]]
        toxicity = [0.08, 0.18, 0.30]
        response = [0.20, 0.32, 0.45]
    else:
        profiles, weights, odds, toxicity, response = inputs
    stage_two = BARDStageTwoDesign(
        total_target=14,
        eligible_profiles=[True, True, True, False],
        prior=[0.25] * 4,
        safety_weights=[1, 1],
        toxicity_limit=0.30,
        efficacy_limit=0.20,
        safety_cutoff=0.95,
        efficacy_cutoff=0.95,
        utilities=[0, 30, 50, 100],
        margin=0.05,
        tie_arm=1,
        stage_two_accrual_rate=2.0,
        dose_pair=(1, 2),
        balanced_factors=(0, 1),
    )
    return BARDStudySpecification(
        "reference scenario",
        BFBOINDesign(target=0.25, n_cap=6, elimination_probability=0.99),
        toxicity,
        response,
        profiles,
        weights,
        odds,
        stage_two,
        true_obd_noninferiority=2,
        true_obd_utility=2,
        seed=seed,
        trials=trials,
        cohorts=3,
        cohort_size=3,
        max_patient_work=max_patient_work,
    )


def test_saved_inputs_are_immutable_and_replay_the_same_serial_oc_study(tmp_path):
    profiles = np.array([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=np.int64)
    weights = np.full(4, 0.25)
    odds = np.array([[1, 1.5], [1, 0.8]])
    toxicity = np.array([0.08, 0.18, 0.30])
    response = np.array([0.20, 0.32, 0.45])
    study = _study(inputs=(profiles, weights, odds, toxicity, response))
    profiles[0, 0] = 99
    weights[:] = 0
    odds[:] = 99
    toxicity[:] = 1
    response[:] = 1
    assert study.patient_work_bound == 3 * (3 * 3 + 3 * 6 + 14)
    assert study.stage_two.balanced_factors == (0, 1)
    assert study.factor_profiles[0] == (1, 1)
    assert study.profile_probabilities == (0.25,) * 4
    with pytest.raises(ValueError):
        study.stage_two.prior.flags.writeable = True

    encoded = study.to_json()
    restored = BARDStudySpecification.from_json(encoded)
    path = tmp_path / "bard-study.json"
    study.write_json(path)
    disk_copy = BARDStudySpecification.read_json(path)
    assert restored.to_json() == encoded
    assert disk_copy.to_json() == encoded

    original, replay = study.run(), restored.run()
    assert original.trials == replay.trials == 3
    assert original.scenario_label == replay.scenario_label is None
    assert original.status_frequency == replay.status_frequency
    assert original.noninferiority_selection_by_dose == replay.noninferiority_selection_by_dose
    assert original.utility_selection_by_dose == replay.utility_selection_by_dose
    assert original.mean_total_enrollment == replay.mean_total_enrollment
    assert original.mean_duration == replay.mean_duration


def test_seed_and_resource_preflight_rejects_invalid_studies_before_running():
    with pytest.raises(ValueError, match="explicit nonnegative 64-bit integer"):
        _study(seed=np.random.default_rng(4))
    with pytest.raises(ValueError, match="patient_work_bound"):
        _study(max_patient_work=10)
    with pytest.raises(ValueError, match="unknown or missing fields"):
        BARDStudySpecification.from_json(
            json.dumps({**_study().to_dict(), "unrecorded_setting": 1})
        )
