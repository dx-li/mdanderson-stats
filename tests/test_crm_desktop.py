"""Saved native schema and bounded full design/report workflow checks."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    dacrm_trimester_prior,
    read_crm_desktop_html,
    simulate_crm_desktop_study,
)

GUID = "A18959FB-1E57-4CE0-AAB1-B038A7B78423"


def document(da=False):
    # Synthetic settings in the recovered native schema; no upstream document
    # or generated output is redistributed as an alleged native trial result.
    flags = "".join(f"<boolean>{v}</boolean>" for v in ["false", "true"])
    xml = f'''<CRMSimApp ProgID="{GUID}" FileVersion="1.0.0">
    <DoseDescriptions><string>A &amp; B</string><string>Dose2</string></DoseDescriptions>
    <Patients/><PreservedSetting>untouched</PreservedSetting><CRMSimulatorData>
    <NumRepetitions>2</NumRepetitions><Seed>15</Seed>
    <MeanInterArrivalTimeSpan>2</MeanInterArrivalTimeSpan><FractToxFirstHalfT>.3</FractToxFirstHalfT>
    <Scenarios><CRMScenario><Name>&lt;script&gt;</Name><Probs><double>.1</double><double>.4</double><double>.9</double></Probs></CRMScenario></Scenarios>
    <Model><DAandSkelsUsed_ForSerializationOnly>
    <SkeletonsBeingUsedWithOutDA_ForSerializationOnly>{flags}</SkeletonsBeingUsedWithOutDA_ForSerializationOnly>
    <SkeletonsBeingUsedWithDA_ForSerializationOnly>{flags}</SkeletonsBeingUsedWithDA_ForSerializationOnly>
    <DataAugmentationIsEnabled>{str(da).lower()}</DataAugmentationIsEnabled>
    </DAandSkelsUsed_ForSerializationOnly><NumDoses>2</NumDoses><NumSkeletons>2</NumSkeletons>
    <Probabilities><double>.05</double><double>.2</double><double>.1</double><double>.3</double><double>.95</double></Probabilities>
    <PriorsAre_P>1</PriorsAre_P><DoseEscalationRule>eStrict</DoseEscalationRule>
    <MinPatientsToDeclareMTD>3</MinPatientsToDeclareMTD><CohortSize>1</CohortSize>
    <MaxPatients>2</MaxPatients><EvalTime>1</EvalTime><StartingDoseIndex>0</StartingDoseIndex>
    <TargetToxicity>.25</TargetToxicity><UpperLimitOnProbToxAtLowestDose>.9</UpperLimitOnProbToxAtLowestDose>
    <MinimumNumberRequiredToEscalate>1</MinimumNumberRequiredToEscalate>
    <ToxicityOccurrencePriorProbabilities><double>.05</double><double>.15</double><double>.8</double></ToxicityOccurrencePriorProbabilities>
    </Model></CRMSimulatorData></CRMSimApp>'''
    return f"<html><script>alert(1)</script></html><!--begin{GUID}\n{xml}\nend{GUID}-->"


@pytest.mark.parametrize("da", [False, True])
def test_complete_saved_study_simulation_report_round_trip(da):
    study = read_crm_desktop_html(document(da))
    assert study.skeletons == ((0.1, 0.3),)
    assert study.scenarios == ((0.1, 0.4),)
    assert read_crm_desktop_html(study.to_html()) == study
    result = simulate_crm_desktop_study(study, trials=2, seed=71, draws=8, warmup=0)
    report = result.to_html()
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert "native-report months" in report
    restored = read_crm_desktop_html(report)
    assert restored.skeletons == study.skeletons
    root = ET.fromstring(restored.xml)
    assert root.findtext("PreservedSetting") == "untouched"
    assert len(root.findall("CRMSimulatorData/Model/Probabilities/double")) == 5
    assert root.findtext("CRMSimulatorData/NumRepetitions") == "2"
    assert root.findtext("CRMSimulatorData/Seed") == "71"
    run = result.simulations[0]
    assert run.scheduling == "native_desktop"
    assert run.patients.sum() <= 4
    assert run.no_selection_probability + run.selection_probability.sum() == pytest.approx(1)


def test_native_hazard_prior_instruction_fixtures():
    cases = json.loads((Path(__file__).parent / "fixtures/crm-native-prior.json").read_text())
    for case in cases:
        prior = dacrm_trimester_prior(
            case["window"], case["probabilities"], dispersion=2, endpoint_policy="native_desktop"
        )
        np.testing.assert_allclose(
            prior.shape, [v["shape"] for v in case["native_prior"]], rtol=2e-12
        )
        np.testing.assert_array_equal(prior.rate, [0.5] * 6)


@pytest.mark.parametrize(
    "before,after",
    [
        ('FileVersion="1.0.0"', 'FileVersion="99"'),
        ("<NumDoses>2</NumDoses>", "<NumDoses>2</NumDoses><NumDoses>3</NumDoses>"),
        ("<double>.1</double>", "<double>nan</double>"),
        ("<CRMSimApp ", '<!DOCTYPE x [<!ENTITY y "z">]><CRMSimApp '),
    ],
)
def test_reject_unsupported_or_ambiguous_native_document(before, after):
    with pytest.raises(ValueError):
        read_crm_desktop_html(document().replace(before, after))


def test_unsupported_policy_and_budget_fail_explicitly():
    study = read_crm_desktop_html(document().replace("eStrict", "eRelaxed"))
    with pytest.raises(ValueError, match="DoseEscalationRule"):
        simulate_crm_desktop_study(study, trials=1, seed=1)
    study = read_crm_desktop_html(document())
    with pytest.raises((RuntimeError, ValueError), match="evaluation"):
        simulate_crm_desktop_study(study, trials=1, seed=1, max_total_evaluations=1)


def test_fractional_native_window_is_not_silently_reinterpreted():
    study = read_crm_desktop_html(
        document().replace("<EvalTime>1</EvalTime>", "<EvalTime>1.5</EvalTime>")
    )
    with pytest.raises(ValueError, match="EvalTime"):
        simulate_crm_desktop_study(study, trials=1, seed=1)
