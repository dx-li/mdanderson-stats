"""Current saved schema; synthetic inputs, not redistributed native trial output."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mdanderson_stats.efftox_desktop import read_efftox_desktop_xml


def document():
    root = ET.Element(
        "EffToxTrial", ProgID="C893BCF6-6184-419a-8173-354C4BDE8728", TypeOfTrial="Simulation"
    )
    design = ET.SubElement(root, "DoseFinderParams")
    for key, value in {
        "Model": 1,
        "StartingDoseIndex": 0,
        "ResponseCutoff": 0.5,
        "ToxicityCutoff": 0.3,
        "AcceptableProbOfLowTox": 0.1,
        "AcceptableProbOfHighRes": 0.1,
        "BivariateBinaryContourPoints": "[.5,.7,.25,.65]",
    }.items():
        ET.SubElement(design, key).text = str(value)
    for parent, child, values in [
        ("Doses", "DVal", [1, 2]),
        ("PriorMeans", "MVal", [-2, 1, 0, 1, 0, 0]),
        ("PriorStdDevs", "SVal", [1, 1, 1, 1, 0.2, 1]),
    ]:
        container = ET.SubElement(design, parent)
        for v in values:
            ET.SubElement(container, child).text = str(v)
    for key, value in {"MaxQtyPatients": 6, "CohortSize": 3, "TrialIsShutDown": 0}.items():
        ET.SubElement(root, key).text = str(value)
    cohorts = ET.SubElement(root, "Cohorts")
    for cohort_codes in ((-1, 0, 1), (2, 3)):
        cohort = ET.SubElement(cohorts, "Cohort")
        for code in cohort_codes:
            patient = ET.SubElement(cohort, "Patient")
            for key, value in {
                "ID": code + 2,
                "BeginDate": "literal date",
                "DoseCalcedIndex": 0,
                "DoseAdminIndex": 1,
                "Outcome": code,
                "Unknown": "kept",
            }.items():
                ET.SubElement(patient, key).text = str(value)
    sim = ET.SubElement(root, "SimulationParams")
    ET.SubElement(sim, "NewestHyperParams", value="output")
    ET.SubElement(sim, "ProbsAreDependent").text = "0"
    scenario = ET.SubElement(sim, "ScenarioParams")
    for parent, child, values in [
        ("TrueToxProbs", "TVal", [0.2, 0.4]),
        ("TrueResProbs", "RVal", [0.3, 0.6]),
        ("TrueResGivenNoToxProbs", "RNVal", [0.99, 0.99]),
    ]:
        container = ET.SubElement(scenario, parent)
        for v in values:
            ET.SubElement(container, child).text = str(v)
    return ET.tostring(root, encoding="unicode")


def test_current_patient_codes_administered_doses_and_metadata_roundtrip():
    study = read_efftox_desktop_xml(document())
    assert read_efftox_desktop_xml(study.to_xml()) == study
    assert [p.binary_outcome for p in study.patients] == [None, (0, 1), (0, 0), (1, 1), (1, 0)]
    np.testing.assert_array_equal(study.observed_counts(), [[[0, 0], [0, 0]], [[1, 1], [1, 1]]])
    assert study.patients[-1].cohort == 2
    assert "literal date" in study.xml and "<Unknown>kept</Unknown>" in study.xml
    np.testing.assert_array_equal(study.saved_prior().mean, [-2, 1, 0, 1, 0, 0])
    assert study.contour().utility(0.7, 0.25) == pytest.approx(0, abs=1e-12)


def test_independent_truth_retains_inactive_conditional_values():
    study = read_efftox_desktop_xml(document())
    truth = study.scenarios[0].joint_probabilities()
    np.testing.assert_allclose(truth.sum(axis=2)[:, 1], [0.3, 0.6])
    np.testing.assert_allclose(truth.sum(axis=1)[:, 1], [0.2, 0.4])
    assert study.scenarios[0].efficacy_given_no_toxicity == (0.99, 0.99)
    with pytest.raises(ValueError, match="impossible joint"):
        read_efftox_desktop_xml(document().replace("<ProbsAreDependent>0", "<ProbsAreDependent>1"))
    source = (
        document().replace("<ProbsAreDependent>0", "<ProbsAreDependent>1").replace(".99", ".35")
    )
    truth = read_efftox_desktop_xml(source).scenarios[0].joint_probabilities()
    np.testing.assert_allclose(truth[:, 1, 0], [0.28, 0.21])


def test_stale_saved_prior_cannot_override_newer_elicitation():
    root = ET.fromstring(document())
    sim = root.find("SimulationParams")
    sim.find("NewestHyperParams").set("value", "input")
    for parent, child in [("ElicitedToxProbs", "TVal"), ("ElicitedResProbs", "RVal")]:
        container = ET.SubElement(sim, parent)
        for _ in range(2):
            ET.SubElement(container, child).text = ".2"
    ET.SubElement(sim, "TargetESS").text = ".9"
    study = read_efftox_desktop_xml(ET.tostring(root, encoding="unicode"))
    with pytest.raises(ValueError, match="requires prior calibration"):
        study.saved_prior()


@pytest.mark.parametrize(
    "before,after",
    [
        ("<Model>1</Model>", "<Model>0</Model>"),
        ("<Model>1</Model>", "<Model>1</Model><Model>1</Model>"),
        ("<MVal>-2</MVal>", ""),
        ("<DVal>2</DVal>", "<DVal>1</DVal>"),
        ("<Outcome>3</Outcome>", "<Outcome>4</Outcome>"),
        ("<DoseAdminIndex>1</DoseAdminIndex>", "<DoseAdminIndex>2</DoseAdminIndex>"),
        ("<ID>5</ID>", "<ID>1</ID>"),
        ("<MaxQtyPatients>6", "<MaxQtyPatients>4"),
        ("<EffToxTrial", "<!DOCTYPE x><EffToxTrial"),
        ("C893BCF6-6184-419a-8173-354C4BDE8728", "historical-guid"),
    ],
)
def test_reject_unsupported_ambiguous_or_invalid_saved_study(before, after):
    with pytest.raises(ValueError):
        read_efftox_desktop_xml(document().replace(before, after))
