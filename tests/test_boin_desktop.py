"""Independently authored project inputs and original patient serialization."""

from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.boin_desktop import read_boin_desktop_project

GUID = "E45C3723-2D8F-40F8-8A47-AEF4DFD489E7"


def document(patient="", extra=""):
    return f'''<Model ProgID="{GUID}" future="retain"><SimDat><TrialDesign>
<SingleDrugTrial>false</SingleDrugTrial><TITE_BOINTrial>false</TITE_BOINTrial>
<FindContours>false</FindContours><NumberOfDoses_DrugA>2</NumberOfDoses_DrugA>
<NumberOfDoses_DrugB>3</NumberOfDoses_DrugB><CohortSize>3</CohortSize>
<MaximumSampleSize>12</MaximumSampleSize><StartingDoseLevel_DrugA>1</StartingDoseLevel_DrugA>
<StartingDoseLevel_DrugB>1</StartingDoseLevel_DrugB><TargetToxicityProbability>.25</TargetToxicityProbability>
<UseThreePlusThreeRunIn>true</UseThreePlusThreeRunIn><UseAcceleratedTitration>false</UseAcceleratedTitration>
<AcceleratedTitrationCap_DrugA>2</AcceleratedTitrationCap_DrugA></TrialDesign>
<Patients>{patient}</Patients><Scenarios><Scenario><Name>Independent -- example</Name>
<RowsPerScenario>2</RowsPerScenario><TrueProbabilities><double>.05</double><double>.1</double>
<double>.2</double><double>.15</double><double>.25</double><double>.4</double>
</TrueProbabilities></Scenario></Scenarios><Seed>7</Seed><NumberOfSimulations>10</NumberOfSimulations>
{extra}</SimDat></Model>'''


def native_patient():
    text = (Path(__file__).parents[1] / "research/fixtures/boin-desktop/patient.xml").read_text()
    return text[text.index("<Patient ") :]


def test_project_roundtrip_retains_metadata_and_comment_safe_names():
    study = read_boin_desktop_project(document(extra="<Unknown>retained</Unknown>"))
    assert study.mode == "combination" and study.shape == (2, 3) and study.run_in
    assert study.scenarios[0].probabilities == ((0.05, 0.1, 0.2), (0.15, 0.25, 0.4))
    assert read_boin_desktop_project(study.to_html()) == study
    assert 'future="retain"' in study.xml and "<Unknown>retained</Unknown>" in study.xml


@pytest.mark.parametrize("outcome,code", [("tslNO", 0), ("tslYES", 1), ("tslUNKNOWN", 2)])
def test_original_patient_serializer_contract(outcome, code):
    study = read_boin_desktop_project(document(native_patient().replace("tslUNKNOWN", outcome)))
    patient = study.patients[0]
    assert (patient.patient_id, patient.fingerprint, patient.dose, patient.outcome) == (
        41,
        101,
        (2, 3),
        code,
    )
    assert patient.date_entered == "10/12/2026"
    n, y, pending = study.counts()
    assert n.sum() == 1 and n[1, 2] == 1
    assert y.sum() == (code == 1) and pending.sum() == (code == 2)
    assert not n.flags.writeable
    assert read_boin_desktop_project(study.to_html()) == study


@pytest.mark.parametrize(
    "old,new",
    [
        ("<CohortSize>3", "<CohortSize>nan"),
        ("<double>.4", "<double>1.4"),
        ("<RowsPerScenario>2", "<RowsPerScenario>3"),
        ("<SingleDrugTrial>false", "<SingleDrugTrial>False"),
        ("<Seed>7", "<Seed>7.5"),
        (GUID, "foreign"),
    ],
)
def test_invalid_saved_contracts(old, new):
    with pytest.raises(ValueError):
        read_boin_desktop_project(document().replace(old, new))


def test_pending_is_not_a_completed_nontoxic_outcome():
    study = read_boin_desktop_project(document(native_patient()))
    n, y, pending = study.counts()
    np.testing.assert_array_equal(n - pending, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="OutcomeCode"):
        read_boin_desktop_project(document(native_patient().replace("tslUNKNOWN", "2")))
    with pytest.raises(ValueError, match="inconsistent"):
        read_boin_desktop_project(
            document(native_patient().replace("<DoseLevelB>3", "<DoseLevelB>2"))
        )


def test_duplicate_ids_entities_and_markers_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        read_boin_desktop_project(document(native_patient() * 2))
    with pytest.raises(ValueError, match="entities"):
        read_boin_desktop_project('<!DOCTYPE x [<!ENTITY e "x">]>' + document())
    html = read_boin_desktop_project(document()).to_html()
    with pytest.raises(ValueError, match="marker"):
        read_boin_desktop_project(html + f"begin{GUID}")


@pytest.mark.parametrize(
    "target,size,expected", [(0.25, 3, True), (0.2, 3, False), (0.25, 1, False)]
)
def test_native_runin_eligibility_gate(target, size, expected):
    xml = (
        document()
        .replace(">.25</TargetToxicityProbability>", f">{target}</TargetToxicityProbability>")
        .replace("<CohortSize>3", f"<CohortSize>{size}")
    )
    assert read_boin_desktop_project(xml).run_in == expected
