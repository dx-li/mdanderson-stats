import numpy as np
import pytest
from test_efftox_desktop import document
from test_efftox_desktop_controller import decision

from mdanderson_stats.efftox_desktop import read_efftox_desktop_xml
from mdanderson_stats.efftox_desktop_patients import (
    add_efftox_desktop_patient,
    update_efftox_desktop_patient,
)


def test_edit_and_append_preserve_metadata_cohort_and_actual_dose():
    study = read_efftox_desktop_xml(document())
    updated = update_efftox_desktop_patient(
        study, 1, outcome=3, administered_dose_index=0, begin_date="<day 1>"
    )
    assert updated.patients[0].binary_outcome == (1, 0)
    assert updated.patients[0].begin_date == "<day 1>"
    assert "<Unknown>kept</Unknown>" in updated.xml
    added = add_efftox_desktop_patient(
        updated, lambda _: decision(0), patient_id=99, begin_date="day2", administered_dose_index=1
    )
    assert 'TypeOfTrial="TrialConduct"' in added.xml
    assert added.patients[-1].cohort == 2
    assert added.patients[-1].calculated_dose_index == 0
    assert added.patients[-1].administered_dose_index == 1
    assert added.patients[-1].binary_outcome is None
    np.testing.assert_array_equal(added.observed_counts(), updated.observed_counts())
    assert read_efftox_desktop_xml(added.xml) == added
    with pytest.raises(ValueError, match="capacity"):
        add_efftox_desktop_patient(added, lambda _: decision(0), patient_id=100, begin_date="x")


def test_append_cannot_bypass_wait_shutdown_or_duplicate_id():
    study = read_efftox_desktop_xml(document())
    for text, service, identifier, match in [
        (study.xml, lambda c: decision(0 if c.sum() == 2 else 1), 99, "wait"),
        (
            study.xml.replace("<TrialIsShutDown>0", "<TrialIsShutDown>1"),
            lambda _: decision(0),
            99,
            "shut down",
        ),
        (study.xml, lambda _: decision(0), 1, "already exists"),
    ]:
        with pytest.raises(ValueError, match=match):
            add_efftox_desktop_patient(
                read_efftox_desktop_xml(text), service, patient_id=identifier, begin_date="x"
            )


def test_edit_can_clear_an_outcome_and_rejects_unknown_patient():
    study = read_efftox_desktop_xml(document())
    edited = update_efftox_desktop_patient(study, 5, outcome=-1)
    assert edited.patients[-1].binary_outcome is None
    assert edited.observed_counts().sum() == study.observed_counts().sum() - 1
    with pytest.raises(ValueError, match="exactly one"):
        update_efftox_desktop_patient(study, 999, outcome=1)


def test_delete_last_and_clear_history_preserve_design_without_recalculating_doses():
    from mdanderson_stats.efftox_desktop_patients import (
        clear_efftox_desktop_patients,
        delete_last_efftox_desktop_patient,
    )

    study = read_efftox_desktop_xml(document())
    one = delete_last_efftox_desktop_patient(study)
    assert one.patients == study.patients[:-1]
    two = delete_last_efftox_desktop_patient(one)
    assert two.patients == study.patients[:3]
    cleared = clear_efftox_desktop_patients(study)
    assert not cleared.patients and cleared.doses == study.doses
    assert cleared.prior_means == study.prior_means and cleared.scenarios == study.scenarios
    with pytest.raises(ValueError, match="no recorded patient"):
        delete_last_efftox_desktop_patient(cleared)
    restarted = add_efftox_desktop_patient(
        cleared, lambda _: decision(0), patient_id=1, begin_date="new start"
    )
    assert restarted.patients[0].cohort == 1
    empty_again = delete_last_efftox_desktop_patient(restarted)
    assert 'TypeOfTrial="Simulation"' in empty_again.xml
    assert 'TypeOfTrial="Simulation"' in clear_efftox_desktop_patients(restarted).xml


def test_edit_patient_id_retains_calculated_dose_and_rejects_duplicate_id():
    study = read_efftox_desktop_xml(document())
    renamed = update_efftox_desktop_patient(study, 1, new_patient_id=99)
    assert renamed.patients[0].patient_id == 99
    assert renamed.patients[0].calculated_dose_index == study.patients[0].calculated_dose_index
    with pytest.raises(ValueError, match="unique"):
        update_efftox_desktop_patient(study, 1, new_patient_id=5)


def test_prior_calibration_cannot_change_an_accruing_trial_design():
    from mdanderson_stats.efftox_desktop_calibration import calibrate_efftox_desktop

    with pytest.raises(ValueError, match="after patient accrual"):
        calibrate_efftox_desktop(read_efftox_desktop_xml(document()), monotone_toxicity=False)


@pytest.mark.parametrize("zero", ["\n  0\n ", "0.0", " 0e0 "])
def test_valid_numeric_xml_shutdown_whitespace_cannot_disable_patient_workflows(zero):
    from mdanderson_stats.efftox_desktop_controller import recommend_efftox_desktop

    saved = document().replace("<TrialIsShutDown>0<", f"<TrialIsShutDown>{zero}<")
    study = read_efftox_desktop_xml(saved)
    assert recommend_efftox_desktop(study, lambda _: decision(0)).status == "go"
    updated = add_efftox_desktop_patient(
        study, lambda _: decision(0), patient_id=99, begin_date="literal date"
    )
    assert len(updated.patients) == 6


def test_clearing_a_conduct_file_without_scenarios_preserves_an_editable_empty_study():
    import xml.etree.ElementTree as ET

    from mdanderson_stats.efftox_desktop_batches import simulate_efftox_desktop_scenarios
    from mdanderson_stats.efftox_desktop_patients import clear_efftox_desktop_patients

    root = ET.fromstring(document())
    root.set("TypeOfTrial", "TrialConduct")
    simulation = root.find("SimulationParams")
    for scenario in simulation.findall("ScenarioParams"):
        simulation.remove(scenario)
    saved = read_efftox_desktop_xml(ET.tostring(root, encoding="unicode"))
    empty = clear_efftox_desktop_patients(saved)
    assert not empty.patients and not empty.scenarios
    assert 'TypeOfTrial="Simulation"' in empty.xml
    with pytest.raises(ValueError, match="no simulation scenarios"):
        simulate_efftox_desktop_scenarios(empty, lambda _: decision(0), trials=1, seed=1)
