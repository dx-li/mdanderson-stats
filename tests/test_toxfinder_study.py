"""Saved physician-input mappings and complete independent simulation workflow."""

from dataclasses import replace

import numpy as np
import pytest

from mdanderson_stats import ToxFinderPrior, read_toxfinder_tmml
from mdanderson_stats.toxfinder_study import elicit_toxfinder_study_prior, simulate_toxfinder_study

TEXT = """win-drug-names 'Drug A' 'Drug B'
pi_star .3
lambda 1
tl-d_star 1200 600
tl-nD 3
tl-D1 120 60 600 300 1200 600
nD-start 1
sim-num-points 2 2
sim-surface 0 .2 .3 .5
num-stage1 2
num-stage2 2
cohort 1
tl-alloc 1
mcmc-trans 200
mcmc-maxiter 2000
mcmc-reeval 30
pr-negtox 600 350
pr-hightox 1400 700
pr-toxic 2000 800
pr-alpha3 1
pr-beta3 1
pr-alpha3-var 3
pr-beta3-var .9
toxic-minprob .05
toxic-maxprob .6
"""


def test_saved_physical_inputs_calibrate_with_literal_variances():
    study = read_toxfinder_tmml(TEXT)
    result = elicit_toxfinder_study_prior(study, seed=2026)
    np.testing.assert_allclose(
        result.prior.mean, [0.3 / 0.7, 8.126872809767802, 0.3 / 0.7, 8.126872809767802, 1, 1]
    )
    np.testing.assert_array_equal(result.prior.variance[-2:], [3, 0.9])
    # Saved .9 is variance, not the displayed SD and not .9 squared.
    run = simulate_toxfinder_study(study, prior=result, trials=2, seed=55, draws=8, warmup=0)
    assert all(t.final_policy == "refit" for t in run.simulation.trials)
    assert all(t.posterior_subjects == 4 for t in run.simulation.trials)
    report = run.to_html()
    assert "Independent prior-calibration assessment" in report
    assert "estimated cutoff met" in report
    assert "Split R-hat" in report and "mcmc-trans 200" in report
    assert run.draws == 8 and run.warmup == 0
    restored = ToxFinderPrior(result.prior.mean.tolist(), result.prior.variance.tolist())
    np.testing.assert_array_equal(restored.variance, result.prior.variance)


def test_legacy_final_fit_requires_explicit_study_option():
    study = read_toxfinder_tmml(TEXT)
    result = simulate_toxfinder_study(
        study,
        prior=ToxFinderPrior(np.ones(6), np.full(6, 0.25)),
        trials=2,
        seed=55,
        draws=8,
        warmup=0,
        final_policy="legacy",
    )
    assert all(t.final_policy == "legacy" for t in result.simulation.trials)
    assert all(t.posterior_subjects == 2 for t in result.simulation.trials)
    assert "Final posterior policy: legacy" in result.to_html()


@pytest.mark.parametrize(
    "edit",
    [
        lambda t: t.replace("pr-negtox 600 350\n", ""),
        lambda t: t.replace("pr-alpha3-var 3", "pr-alpha3-var -3"),
        lambda t: t.replace("toxic-minprob .05", "toxic-minprob .9"),
    ],
)
def test_missing_or_invalid_physician_inputs_fail_without_inventing_prior(edit):
    with pytest.raises(ValueError):
        elicit_toxfinder_study_prior(read_toxfinder_tmml(edit(TEXT)), seed=2026)


def test_forged_derived_study_fields_cannot_override_saved_settings():
    study = replace(read_toxfinder_tmml(TEXT), target=0.9)
    result = elicit_toxfinder_study_prior(study, seed=2026)
    assert result.prior.mean[0] == pytest.approx(0.3 / 0.7)


def test_patient_csv_edit_roundtrip_and_saved_study_recommendation_report():
    from mdanderson_stats import read_toxfinder_subject_csv
    from mdanderson_stats.toxfinder_study import (
        recommend_toxfinder_study,
        write_toxfinder_subject_csv,
    )

    study = read_toxfinder_tmml(TEXT)
    csv = "p1,id1,2026,10,12,10:00,1,120,60,0\np2,id2,2026,10,12,11:00,2,600,300,2\n"
    records = read_toxfinder_subject_csv(csv, reference_doses=study.reference_doses)
    saved = write_toxfinder_subject_csv(records, reference_doses=study.reference_doses)
    restored = read_toxfinder_subject_csv(saved, reference_doses=study.reference_doses)
    assert [r.toxicity for r in restored] == [0, 2]
    np.testing.assert_array_equal(restored[1].standardized_dose, [0.5, 0.5])
    result = recommend_toxfinder_study(
        study,
        restored,
        prior=ToxFinderPrior(np.ones(6), np.full(6, 0.25)),
        seed=17,
        draws=8,
        warmup=0,
    )
    assert result.recommendation.observed_patients == 1
    assert result.recommendation.pending_patients == 1
    assert result.recommendation.reason == "stage_one"
    assert "physical dose pair" in result.to_html()
    assert "p2,id2,2026,10,12,11:00,2,600.0,300.0,2" in result.to_html()
    edited = (restored[0], replace(restored[1], toxicity=1))
    completed = read_toxfinder_subject_csv(
        write_toxfinder_subject_csv(edited, reference_doses=study.reference_doses),
        reference_doses=study.reference_doses,
    )
    assert completed[1].toxicity == 1
    with pytest.raises(ValueError):
        write_toxfinder_subject_csv((replace(records[0], toxicity=3),), reference_doses=[1200, 600])


def test_csv_writer_rejects_comment_ids_and_preserves_quoted_identifiers():
    from mdanderson_stats import read_toxfinder_subject_csv
    from mdanderson_stats.toxfinder_study import write_toxfinder_subject_csv

    row = read_toxfinder_subject_csv(
        "p1,id1,2026,10,12,10:00,1,120,60,0\n", reference_doses=[1200, 600]
    )[0]
    with pytest.raises(ValueError, match="comments"):
        write_toxfinder_subject_csv(
            (replace(row, patient_id="# hidden"),), reference_doses=[1200, 600]
        )
    row = replace(row, identifier='name, with "quotes"')
    text = write_toxfinder_subject_csv((row,), reference_doses=[1200, 600])
    assert (
        read_toxfinder_subject_csv(text, reference_doses=[1200, 600])[0].identifier
        == row.identifier
    )


def test_ordinary_design_impossible_budget_fails_before_sampling(monkeypatch):
    import mdanderson_stats.toxfinder_native_trial as module

    study = read_toxfinder_tmml(
        TEXT.replace("num-stage1 2\n", "num-stage1 20\n")
        .replace("num-stage2 2\n", "num-stage2 40\n")
        .replace("cohort 1\n", "cohort 2\n")
    )

    def forbidden(*args, **kwargs):
        pytest.fail("preflight must happen before fitting")

    monkeypatch.setattr(module, "fit_toxfinder", forbidden)
    prior = ToxFinderPrior(np.ones(6), np.full(6, 0.25))
    with pytest.raises(ValueError, match="deterministic minimum 28622124"):
        simulate_toxfinder_study(study, prior=prior, trials=2, seed=55)
    with pytest.raises(ValueError, match="retained final fits"):
        simulate_toxfinder_study(
            study, prior=prior, trials=50, seed=55, max_sampler_evaluations=200_000_000
        )


def test_exhausted_map_passes_survive_simulation_and_patient_reports(monkeypatch):
    import importlib

    module = importlib.import_module("mdanderson_stats.toxfinder_native_information")
    simplex = module._simplex
    monkeypatch.setattr(
        module, "_simplex", lambda start, step, objective: simplex(start, step, objective, budget=1)
    )
    prior = ToxFinderPrior(np.ones(6), np.full(6, 0.25))
    study = read_toxfinder_tmml(TEXT)
    result = simulate_toxfinder_study(study, prior=prior, trials=2, seed=55, draws=8, warmup=0)
    for trial in result.simulation.trials:
        assert trial.cohorts[0].map_fit is None
        for cohort in trial.cohorts[2:]:
            assert cohort.map_fit.pass_converged == (False, False)
            assert cohort.map_fit.pass_iterations == (1, 1)
    assert "(False, False)" in result.to_html()
    from mdanderson_stats.toxfinder_native_io import read_toxfinder_subject_csv
    from mdanderson_stats.toxfinder_study import recommend_toxfinder_study

    records = read_toxfinder_subject_csv(
        "1,x,2026,1,1,0,0,120,60,0\n2,x,2026,1,1,0,1,600,300,1\n",
        reference_doses=study.reference_doses,
    )
    recommendation = recommend_toxfinder_study(
        study, records, prior=prior, seed=5, draws=8, warmup=0
    )
    assert recommendation.recommendation.map_fit.pass_converged == (False, False)
    assert "(False, False)" in recommendation.to_html()
