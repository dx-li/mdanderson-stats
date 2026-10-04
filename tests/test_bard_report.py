import pytest

from mdanderson_stats.bard_bf_boin_trial import BARDStageTwoDesign
from mdanderson_stats.bard_report import bard_design_report
from mdanderson_stats.bard_study import BARDStudySpecification
from mdanderson_stats.bf_boin import BFBOINDesign


def _specification(label: str, seed: int, *, trials: int = 3) -> BARDStudySpecification:
    stage_two = BARDStageTwoDesign(
        total_target=6,
        eligible_profiles=[True, True],
        prior=[0.25, 0.25, 0.25, 0.25],
        safety_weights=[1.0, 1.0],
        toxicity_limit=0.3,
        efficacy_limit=0.2,
        safety_cutoff=0.99,
        efficacy_cutoff=0.99,
        utilities=[0.0, 30.0, 50.0, 100.0],
        margin=0.05,
        tie_arm=1,
        stage_two_accrual_rate=1.0,
        dose_pair=(1, 2),
    )
    return BARDStudySpecification(
        label=label,
        design=BFBOINDesign(target=0.3, n_cap=6, elimination_probability=0.99),
        true_toxicity=[0.1, 0.25],
        population_response=[0.3, 0.6],
        factor_profiles=[[1], [2]],
        profile_probabilities=[0.5, 0.5],
        response_odds_ratios=[[1.0, 1.5]],
        stage_two=stage_two,
        true_obd_noninferiority=2,
        true_obd_utility=2,
        seed=seed,
        trials=trials,
        cohorts=2,
        cohort_size=3,
        stage_one_accrual_rate=1.0,
    )


def test_report_replays_each_scenario_independently_of_order() -> None:
    first = _specification("scenario A", 401)
    second = _specification("scenario B", 812)
    forward = bard_design_report([first, second])
    reverse = bard_design_report([second, first])

    forward_by_label = dict(zip((item.label for item in forward.specifications), forward.summaries))
    reverse_by_label = dict(zip((item.label for item in reverse.specifications), reverse.summaries))
    assert forward_by_label == reverse_by_label
    assert forward.patient_work_bound == first.patient_work_bound + second.patient_work_bound


def test_html_escapes_labels_and_contains_captured_settings_and_oc_fields() -> None:
    specification = _specification("<Scenario & one>", 71)
    report = bard_design_report([specification])
    content = report.to_html()

    assert "&lt;Scenario &amp; one&gt;" in content
    assert "Complete captured study inputs (versioned JSON)" in content
    assert "&quot;response_model&quot;" in content
    assert "Balanced factor columns" in content
    assert "No selection" in content
    assert "correct/all trials" in content
    assert "Absolute arm-count difference" in content
    assert "Per-factor level-1 allocation imbalance" in content
    assert "BF-BLRM" in content
    assert "<Scenario & one>" not in content


def test_aggregate_budget_rejects_before_running_any_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _specification("scenario A", 4, trials=2)
    second = _specification("scenario B", 5, trials=2)
    calls: list[str] = []

    def run_must_not_start(self: BARDStudySpecification) -> None:
        calls.append(self.label)
        raise AssertionError("simulation started before aggregate preflight")

    monkeypatch.setattr(BARDStudySpecification, "run", run_must_not_start)
    budget = first.patient_work_bound + second.patient_work_bound - 1
    with pytest.raises(ValueError, match="aggregate patient-work bound"):
        bard_design_report([first, second], max_patient_work=budget)
    assert calls == []
