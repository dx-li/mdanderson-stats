"""Patient-file workflows retain native identifiers, cohort gates and metadata."""

import pytest
from test_boin_desktop import document

from mdanderson_stats.boin_desktop import read_boin_desktop_project
from mdanderson_stats.boin_desktop_controller import recommend_boin_desktop
from mdanderson_stats.boin_desktop_patients import (
    add_boin_desktop_patient,
    clear_boin_desktop_patients,
    delete_last_boin_desktop_patient,
    update_boin_desktop_patient,
)


def study():
    xml = document().replace(
        "</TrialDesign>",
        """<AcceptableToxicity_LowerBound>.15</AcceptableToxicity_LowerBound>
<AcceptableToxicity_UpperBound>.35</AcceptableToxicity_UpperBound><EliminateDoseThreshold>.95</EliminateDoseThreshold>
<ExtraSafe>false</ExtraSafe><Offset>.05</Offset><EarlyStop>12</EarlyStop></TrialDesign>""",
    )
    return read_boin_desktop_project(xml)


def test_pending_add_update_delete_and_reopen():
    saved = study()
    for i in range(3):
        saved = add_boin_desktop_patient(saved, patient_id=i + 1, date_entered="2026-10-12", seed=3)
    assert recommend_boin_desktop(saved, seed=3).status == "wait"
    with pytest.raises(ValueError, match="requires GO"):
        add_boin_desktop_patient(saved, patient_id=4, date_entered="2026-10-13", seed=3)
    original = saved.patients
    for i in range(3):
        saved = update_boin_desktop_patient(saved, i + 1, date_updated="2026-10-13", outcome=0)
    assert recommend_boin_desktop(saved, seed=3).status == "go"
    assert [p.fingerprint for p in saved.patients] == [p.fingerprint for p in original]
    saved = add_boin_desktop_patient(saved, patient_id=4, date_entered="2026-10-14", seed=3)
    assert saved.patients[-1].cohort == 2
    assert read_boin_desktop_project(saved.to_html()) == saved
    saved = delete_last_boin_desktop_patient(saved)
    assert len(saved.patients) == 3
    assert not clear_boin_desktop_patients(saved).patients


def test_capacity_and_duplicate_identity_cannot_be_bypassed():
    saved = read_boin_desktop_project(
        study().xml.replace("<MaximumSampleSize>12", "<MaximumSampleSize>2")
    )
    saved = add_boin_desktop_patient(saved, patient_id=1, date_entered="2026-10-12", seed=1)
    with pytest.raises(ValueError, match="already exists"):
        add_boin_desktop_patient(saved, patient_id=1, date_entered="2026-10-12", seed=1)
    saved = add_boin_desktop_patient(saved, patient_id=2, date_entered="2026-10-12", seed=1)
    with pytest.raises(ValueError, match="capacity"):
        add_boin_desktop_patient(saved, patient_id=3, date_entered="2026-10-12", seed=1)
    with pytest.raises(ValueError, match="duplicate"):
        update_boin_desktop_patient(saved, 2, new_patient_id=1, date_updated="2026-10-13")


def test_new_dates_and_outcomes_are_explicit():
    with pytest.raises(ValueError, match="ISO"):
        add_boin_desktop_patient(study(), patient_id=1, date_entered="tomorrow", seed=1)
    with pytest.raises(ValueError, match="outcome"):
        add_boin_desktop_patient(
            study(), patient_id=1, date_entered="2026-10-12", seed=1, outcome=-1
        )
    with pytest.raises(ValueError, match="no patient"):
        delete_last_boin_desktop_patient(study())


def test_contour_saved_caps_and_safe_invalid_native_branch():
    import numpy as np

    from mdanderson_stats.boin_desktop_contour import desktop_contour_decision
    from mdanderson_stats.boin_desktop_decision import BOINDesktopCombDesign

    xml = (
        study()
        .xml.replace("<FindContours>false", "<FindContours>true")
        .replace(
            "</TrialDesign>",
            "<MaximumSampleSizeForSubTrials><int>12</int><int>6</int></MaximumSampleSizeForSubTrials></TrialDesign>",
        )
    )
    saved = read_boin_desktop_project(xml)
    assert saved.subtrial_maximums == (12, 6)
    assert read_boin_desktop_project(saved.to_html()) == saved
    design = BOINDesktopCombDesign(target=0.25, extra_safe=True, early_stop_patients=12)
    n = np.array([[3, 0, 0], [0, 0, 0]])
    y = np.array([[2, 0, 0], [0, 0, 0]])
    assert desktop_contour_decision(design, n, y, (1, 1), [12, 6]).action == "stop_safety"
    assert (
        desktop_contour_decision(
            design, n, y, (1, 1), [12, 6], compatibility_profile="desktop_legacy"
        ).action
        == "error"
    )
