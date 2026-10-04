import pytest

from mdanderson_stats.multc_study import (
    MultcStudySimulationSettings,
    MultcStudySpecification,
)


def _spec(*, simulation=None):
    return MultcStudySpecification(
        3,
        (1, 1),
        (1, 1),
        0.5,
        0.5,
        [[0.1, 0.2, 0.3, 0.4]],
        scenario_names=("paired truth",),
        response_cutoff=1,
        toxicity_cutoff=1,
        pretrial_check=False,
        simulation=simulation,
    )


def test_saved_study_round_trips_inputs_and_reports_exact_scenario_results(tmp_path):
    original = _spec().run()
    destination = tmp_path / "multc.json"
    original.write_specification(destination)
    replayed = MultcStudySpecification.from_json(destination.read_text()).run()

    assert replayed.specification == original.specification
    assert replayed.operating_characteristics[0].expected_sample_size == pytest.approx(3)
    report = replayed.report(digits=17)
    assert "Full stopping boundaries" in report
    assert "Exact joint operating characteristics" in report
    assert "paired truth" in report
    assert "not native Multc Lean files" in report
    report_path = tmp_path / "multc.tsv"
    replayed.write_report(report_path, digits=17)
    assert report_path.read_text() == report


def test_optional_duration_simulation_is_captured_and_labeled_separately():
    settings = MultcStudySimulationSettings(
        response_window=1,
        toxicity_delay=0,
        response_timing="clip_at_window",
        accrual_rate=2,
        trials=2,
        seed=4,
    )
    study = MultcStudySpecification.from_json(_spec(simulation=settings).to_json()).run()

    assert len(study.simulations) == 1
    assert len(study.simulation_seeds) == 1
    assert study.simulations[0].trials == 2
    report = study.report()
    assert "Calendar simulation (Monte Carlo estimates; separate from exact OCs)" in report
    assert f"\t{study.simulation_seeds[0]}\t2\t" in report
    assert "Decision probabilities" in report


def test_simulation_budget_and_serialization_input_guards():
    settings = MultcStudySimulationSettings(
        response_window=1,
        toxicity_delay=0,
        response_timing="clip_at_window",
        accrual_rate=2,
        trials=2,
        seed=4,
        max_total_work=1,
    )
    with pytest.raises(ValueError, match="simulation scenarios exceed max_total_work"):
        _spec(simulation=settings).run()
    storage_limited = MultcStudySimulationSettings(
        response_window=1,
        toxicity_delay=0,
        response_timing="clip_at_window",
        accrual_rate=2,
        trials=1,
        seed=4,
        max_total_storage_bytes=1,
    )
    with pytest.raises(ValueError, match="simulation scenarios exceed max_total_storage_bytes"):
        _spec(simulation=storage_limited).run()
    with pytest.raises(ValueError, match="duplicate Multc study JSON key"):
        MultcStudySpecification.from_json('{"max_subjects": 3, "max_subjects": 4}')
    with pytest.raises(ValueError, match="non-finite JSON number"):
        MultcStudySpecification.from_json('{"scenario_probabilities": NaN}')
    with pytest.raises(ValueError, match="1 MiB"):
        MultcStudySpecification.from_json(" " * 1_048_577)


def test_specification_rejects_unrecognized_fields_and_malformed_scenario_labels():
    with pytest.raises(ValueError, match="unknown Multc study fields"):
        MultcStudySpecification.from_json('{"max_subjects": 3, "invented": true}')
    with pytest.raises(ValueError, match="scenario_names must be unique nonempty"):
        MultcStudySpecification(
            3,
            (1, 1),
            (1, 1),
            0.5,
            0.5,
            [[0, 0, 0, 1]],
            scenario_names=("bad\tlabel",),
        ).run()
