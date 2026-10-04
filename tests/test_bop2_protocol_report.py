import importlib

import numpy as np
import pytest

from mdanderson_stats.bop2_binary import _success, bop2_binary_design
from mdanderson_stats.bop2_efftox import bop2_efftox_design
from mdanderson_stats.bop2_paired import _cells, bop2_paired_design
from mdanderson_stats.bop2_protocol_report import (
    BOP2ProtocolScenario,
    bop2_binary_efficacy_report,
    bop2_binary_toxicity_report,
    bop2_efftox_report,
    bop2_multiple_report,
    bop2_ordinal_report,
    bop2_survival_report,
)
from mdanderson_stats.bop2_survival import bop2_survival_design
from mdanderson_stats.bop2_survival_trial import simulate_bop2_survival


def test_binary_endpoint_reports_use_existing_oc_and_keep_toxicity_adverse_rate():
    cases = (
        (bop2_binary_efficacy_report, "efficacy", 0.2, 0.4),
        (bop2_binary_toxicity_report, "toxicity", 0.3, 0.1),
    )
    for factory, endpoint, null, alternative in cases:
        report = factory(
            4,
            null,
            (BOP2ProtocolScenario("null", null), BOP2ProtocolScenario("alt", alternative)),
            cutoff_scale=0.8,
            gamma=0.5,
            looks=[2, 4],
            min_subjects=2,
            cohort_size=2,
        )
        design = bop2_binary_design(
            4,
            null,
            endpoint=endpoint,
            cutoff_scale=0.8,
            gamma=0.5,
            looks=[2, 4],
            min_subjects=2,
            cohort_size=2,
        )
        oc = design.operating_characteristics(alternative)
        assert report.scenarios[1].success_probability == pytest.approx(float(_success(design, oc)))
        assert report.scenarios[1].sample_size_probability == pytest.approx(
            tuple(oc.sample_size_probability)
        )
        assert sum(report.scenarios[1].stop_probability_by_look) + report.scenarios[
            1
        ].success_probability == pytest.approx(1.0)
        if endpoint == "toxicity":
            assert report.scenarios[1].truth_values == (alternative,)
            assert "complete-negative" in report.to_html()


def test_ordinal_multiple_and_efftox_reports_retain_category_order_and_exact_oc():
    ordinal_scenario = BOP2ProtocolScenario("ordinal", [0.2, 0.6])
    ordinal = bop2_ordinal_report(
        4,
        [0.2, 0.5],
        [ordinal_scenario],
        cutoff_scale=0.8,
        gamma=0.5,
        looks=[2, 4],
        min_subjects=2,
        cohort_size=2,
    )
    ordinal_design = bop2_paired_design(
        4,
        [0.2, 0.5],
        cutoff_scale=0.8,
        gamma=0.5,
        endpoint="ordinal",
        looks=[2, 4],
        min_subjects=2,
        cohort_size=2,
    )
    ordinal_cells = _cells((0.2, 0.6), None, "ordinal")
    assert ordinal.scenarios[0].category_probabilities == pytest.approx(tuple(ordinal_cells))
    assert ordinal.scenarios[0].success_probability == pytest.approx(
        float(ordinal_design.operating_characteristics(ordinal_cells).success_probability)
    )
    assert sum(ordinal.scenarios[0].stop_probability_by_look) + ordinal.scenarios[
        0
    ].success_probability == pytest.approx(1.0)
    assert "both marginal criteria fail" in ordinal.decision_rule

    multiple_scenario = BOP2ProtocolScenario("multiple", [0.5, 0.4], 0.25)
    multiple = bop2_multiple_report(
        4,
        [0.2, 0.3],
        [multiple_scenario],
        cutoff_scale=0.8,
        gamma=0.5,
        null_joint_rate=0.1,
        looks=[2, 4],
        min_subjects=2,
        cohort_size=2,
    )
    assert multiple.scenarios[0].category_probabilities == pytest.approx((0.25, 0.25, 0.15, 0.35))

    efftox = bop2_efftox_report(
        4,
        [0.2, 0.3],
        [0.5, 0.1],
        cutoff_scales=[0.8, 0.8],
        gamma=0.5,
        efficacy_looks=[2, 4],
        toxicity_looks=[2, 4],
        min_subjects=2,
        cohort_size=2,
    )
    assert tuple(s.label for s in efftox.scenarios) == ("H00", "H01", "H10", "H11")
    efftox_design = bop2_efftox_design(
        4,
        [0.2, 0.3],
        cutoff_scales=[0.8, 0.8],
        gamma=0.5,
        efficacy_looks=[2, 4],
        toxicity_looks=[2, 4],
        min_subjects=2,
        cohort_size=2,
    )
    h11_cells = _cells((0.5, 0.1), 0.05, "multiple")
    assert efftox.scenarios[-1].category_probabilities == pytest.approx(tuple(h11_cells))
    assert efftox.scenarios[-1].success_probability == pytest.approx(
        float(efftox_design.operating_characteristics(h11_cells).success_probability)
    )
    assert sum(efftox.scenarios[-1].stop_probability_by_look) + efftox.scenarios[
        -1
    ].success_probability == pytest.approx(1.0)
    assert "either assessed failure" in efftox.decision_rule


def test_survival_report_replays_seeded_null_then_alternative_simulation():
    kwargs = dict(
        max_subjects=4,
        null_median=3.0,
        alternative_median=5.0,
        cutoff_scale=0.8,
        gamma=0.5,
        n_trials=60,
        seed=821,
        accrual_rate=2.0,
        final_followup=1.0,
        arrival="poisson",
        looks=[2, 4],
        min_subjects=2,
        cohort_size=2,
    )
    report = bop2_survival_report(**kwargs)
    design = bop2_survival_design(
        4,
        3.0,
        cutoff_scale=0.8,
        gamma=0.5,
        looks=[2, 4],
        min_subjects=2,
        cohort_size=2,
    )
    rng = np.random.default_rng(821)
    expected = simulate_bop2_survival(
        design,
        3.0,
        accrual_rate=2.0,
        final_followup=1.0,
        n_trials=60,
        arrival="poisson",
        rng=rng,
    )
    assert report.scenarios[0].success_probability == expected.success_probability
    assert report.scenarios[0].success_mcse == expected.success_mcse
    assert report.scenarios[0].expected_sample_size == expected.expected_sample_size
    assert dict(report.settings)["n_trials"] == "60"
    assert dict(report.settings)["seed"] == "821"
    assert report.boundary_rows
    assert sum(report.scenarios[0].stop_probability_by_look) + report.scenarios[
        0
    ].success_probability == pytest.approx(1.0)


def test_report_html_escapes_labels_and_exact_work_fails_before_design(monkeypatch, tmp_path):
    report = bop2_binary_efficacy_report(
        4,
        0.2,
        [BOP2ProtocolScenario("<unsafe>", 0.2)],
        cutoff_scale=0.8,
        gamma=0.5,
        looks=[2, 4],
        min_subjects=2,
        cohort_size=2,
    )
    assert "&lt;unsafe&gt;" in report.to_html()
    assert "<unsafe>" not in report.to_html()

    module = importlib.import_module("mdanderson_stats.bop2_protocol_report")
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("design factory should not run after work preflight rejection")

    monkeypatch.setattr(module, "bop2_paired_design", fail_if_called)
    many = tuple(BOP2ProtocolScenario(f"s{i}", [0.2, 0.5]) for i in range(20))
    with pytest.raises(ValueError, match="exact-work budget"):
        bop2_ordinal_report(
            200,
            [0.2, 0.5],
            many,
            cutoff_scale=0.8,
            gamma=0.5,
        )
    assert not called

    destination = tmp_path / "report.html"
    report.write_html(destination)
    assert destination.read_text(encoding="utf-8") == report.to_html()
