"""Original executable fixtures, controller boundaries, and end-to-end workflows."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.toxfinder_model import ToxFinderPrior, toxfinder_probabilities
from mdanderson_stats.toxfinder_native_decision import (
    _allocation,
    _stage1,
    toxfinder_native_allocation,
    toxfinder_native_stage1,
)
from mdanderson_stats.toxfinder_native_information import (
    _simplex,
    fit_toxfinder_native_map,
    toxfinder_native_hessian,
    toxfinder_native_information,
)
from mdanderson_stats.toxfinder_native_io import (
    read_toxfinder_simulations_csv,
    read_toxfinder_subject_csv,
    recommend_toxfinder_native,
)
from mdanderson_stats.toxfinder_native_trial import (
    simulate_toxfinder_native_trial,
    toxfinder_native_surface,
)

FIXTURES = Path(__file__).parent / "fixtures"
PUBLIC = [0, 3, 1, 4, 2, 5]


def fixture(name):
    return json.loads((FIXTURES / f"toxfinder-native-{name}.json").read_text())


def prior_for(row):
    shape, rate = np.asarray(row["shapes"]), np.asarray(row["rates"])
    return ToxFinderPrior((shape / rate)[PUBLIC], (shape / rate**2)[PUBLIC])


@pytest.mark.parametrize("row", fixture("simplex"))
def test_original_simplex_budget_and_ties(row):
    f = (lambda x: 4.0) if row.get("constant") else lambda x: (x[0] - 3) ** 2 + (x[1] + 1) ** 2
    point, converged, iterations = _simplex(row["start"], row["step"], f, row["budget"])
    np.testing.assert_allclose(point, row["point"], rtol=0, atol=1e-12)
    assert iterations == row["iterations"]


@pytest.mark.parametrize("row", fixture("map"))
def test_original_complete_map(row):
    fit = fit_toxfinder_native_map(row["doses"], row["outcomes"], [1, 1], prior=prior_for(row))
    np.testing.assert_allclose(fit.parameters, np.asarray(row["point"])[PUBLIC], rtol=0, atol=1e-10)
    assert fit.evaluations > 14
    with pytest.raises(ValueError):
        fit.parameters.flags.writeable = True


@pytest.mark.parametrize("row", fixture("information"))
def test_original_complete_information_chain(row):
    prior = prior_for(row)
    info = toxfinder_native_information(row["doses"], row["outcomes"], [1, 1], prior=prior)
    samples = np.asarray(row["samples"])[:, PUBLIC]
    probability = float(toxfinder_probabilities([row["candidate"]], samples)[..., 1].mean())
    assert info.score(row["candidate"], probability) == pytest.approx(
        row["native_information"], rel=2e-10
    )
    parameters = np.asarray(row["fitted_modes"][0])[PUBLIC]
    for outcome in (0, 1):
        h = toxfinder_native_hessian(
            row["doses"] + [row["candidate"]],
            row["outcomes"] + [outcome],
            [1, 1, 1],
            parameters,
            prior=prior,
        )
        native = np.asarray(row["raw_hessians"][outcome])[np.ix_(PUBLIC, PUBLIC)]
        np.testing.assert_allclose(h, native, rtol=1e-12, atol=2e-12)


@pytest.mark.parametrize("row", fixture("stage1"))
def test_original_stage_one(row):
    actual = _stage1(
        np.asarray(row["levels"]),
        np.asarray(row["history"]),
        np.asarray(row["outcomes"]),
        row["target"],
        lambda x, y: x * x,
        0,
    )
    np.testing.assert_allclose(actual, row["dose"], rtol=0, atol=5e-17)


def test_safe_lowest_boundary_and_empty_start():
    levels = np.array([[0.1, 0.1], [0.3, 0.3], [0.7, 0.7]])
    assert np.array_equal(
        _stage1(levels, np.array([[0.3, 0.3]]), np.array([1]), 0.01, lambda x, y: 0.9, 0), levels[0]
    )
    assert np.array_equal(
        toxfinder_native_stage1(levels, np.ones(6), [], [], target=0.3, start_index=1), levels[1]
    )


@pytest.mark.parametrize("index,row", list(enumerate(fixture("allocation30"))))
def test_original_production_grid(index, row):
    levels = np.asarray(row["levels"] or [[0.2, 0.2], [0.8, 0.8]])
    # The original fixture declares its constant or 2x+y callback.
    if row["objective"] == "constant":

        def objective(x, y):
            return 1.0
    else:

        def objective(x, y):
            return 2 * x + y

    actual = _allocation(
        levels,
        row["goal"],
        row["ckp"],
        "fi" if row["mode"] == 4 else "fi_ckp",
        lambda x, y: 0.1 + 0.4 * x + 0.2 * y,
        objective,
        row["allocate"],
    )
    np.testing.assert_allclose(actual, np.asarray([row["x"], row["y"]]).T, rtol=0, atol=2e-16)


@pytest.mark.parametrize("row", fixture("surface"))
def test_original_bilinear_surface(row):
    grid = [[0, 0.1, 0.4], [0.2, 0.5, 0.7], [0.3, 0.6, 0.9]]
    assert toxfinder_native_surface(grid, [row["dose"]])[0] == pytest.approx(
        row["native"], abs=1e-16
    )


@pytest.mark.parametrize("policy", ["legacy", "refit"])
def test_full_paired_cohort_trial_and_report(policy):
    prior = ToxFinderPrior(np.ones(6), np.full(6, 0.25))
    result = simulate_toxfinder_native_trial(
        [[0.1, 0.1], [0.3, 0.2], [0.6, 0.7], [0.9, 0.95]],
        [[0, 0.3], [0.2, 0.5]],
        prior=prior,
        target=0.3,
        stage1_patients=2,
        stage2_patients=4,
        cohort_size=1,
        seed=12,
        draws=8,
        warmup=0,
        final_policy=policy,
    )
    assert result.subjects == 6
    assert [c.selected_index for c in result.cohorts] == [0, 0, 2, 1, 2, 1]
    assert [c.posterior_update for c in result.cohorts] == [1, 2, 3, 3, 4, 4]
    assert result.posterior_subjects == (4 if policy == "legacy" else 6)
    assert result.toxicities == sum(c.toxicities for c in result.cohorts)
    assert np.array_equal(result.cohorts[2].recommended_doses, result.cohorts[3].recommended_doses)
    output = result.to_html(reference_doses=[1200, 600], title="<script>")
    assert "&lt;script&gt;" in output and "<script>" not in output
    assert "Posterior SD" in output


def test_patient_pending_exclusion_and_capacity():
    records = read_toxfinder_subject_csv(
        "# header\na,A,2026,10,12,10:00,0,120,60,1\nb,B,2026,10,12,10:00,0,120,60,2\n",
        reference_doses=[1200, 600],
    )
    result = recommend_toxfinder_native(
        records,
        [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]],
        prior=ToxFinderPrior(np.ones(6), np.full(6, 0.25)),
        target=0.3,
        stage1_patients=2,
        stage2_patients=0,
        cohort_size=1,
        seed=4,
        draws=8,
        warmup=0,
    )
    assert result.observed_patients == result.pending_patients == 1
    assert result.fit.subjects.sum() == 1
    assert result.dose is None and result.reason == "capacity_reached"
    assert result.records[1].toxicity == 2


def test_archived_csv_native_parameter_order():
    row = "0,120,60,240,90,90,120,13,15,2,6,.3,1,2,3,4,5,6,"
    result = read_toxfinder_simulations_csv(row, reference_doses=[1200, 600], retained_draws=1)[0]
    np.testing.assert_array_equal(result.parameters, [[1, 4, 2, 5, 3, 6]])
    assert result.subjects == 6 and result.toxicities == 2
    np.testing.assert_allclose(result.standardized_doses[0], [0.1, 0.1])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_invalid_potency(bad):
    with pytest.raises(ValueError):
        toxfinder_native_allocation(
            [[0.1, 0.1], [0.9, 0.9]], np.ones(6), target=0.3, potency=bad, allocation=False
        )


def test_resource_bounds_before_trial():
    prior = ToxFinderPrior(np.ones(6), np.full(6, 0.25))
    with pytest.raises(ValueError, match="budget"):
        simulate_toxfinder_native_trial(
            [[0.1, 0.1], [0.9, 0.9]],
            [[0, 0.2], [0.3, 0.5]],
            prior=prior,
            target=0.3,
            stage1_patients=2,
            stage2_patients=2,
            cohort_size=1,
            seed=1,
            draws=8,
            warmup=0,
            max_sampler_evaluations=1,
        )


def test_archived_report_independent_60_digit_reference():
    row = fixture("archived-summary")
    archived = read_toxfinder_simulations_csv(
        row["csv_row"], reference_doses=[1200, 600], retained_draws=60
    )[0]
    probabilities = toxfinder_probabilities(archived.standardized_doses, archived.parameters)[
        ..., 1
    ]
    truth = toxfinder_native_surface(row["true_surface"], archived.standardized_doses)
    for i, expected in enumerate(row["recommendations"]):
        assert probabilities[:, i].mean() == pytest.approx(
            float(expected["posterior_mean"]), abs=2e-15
        )
        assert probabilities[:, i].std(ddof=1) == pytest.approx(
            float(expected["posterior_sd_sample"]), abs=2e-15
        )
        assert truth[i] == pytest.approx(float(expected["scenario_true_bilinear"]), abs=2e-15)
    assert archived.toxicities == 19 and archived.subjects == 60


def test_tmml_roundtrip_original_endpoint_and_simulation_summary():
    from mdanderson_stats import read_toxfinder_tmml, simulate_toxfinder_native

    text = """# Native format, synthetic bounded study
win-drug-names 'Drug A' 'Drug B'
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
"""
    study = read_toxfinder_tmml(text)
    assert study.settings == read_toxfinder_tmml(study.to_tmml()).settings
    np.testing.assert_array_equal(study.dose_levels[-1], [1, 1])
    result = simulate_toxfinder_native(
        study.dose_levels,
        study.true_surface,
        prior=ToxFinderPrior(np.ones(6), np.full(6, 0.25)),
        target=study.target,
        stage1_patients=study.stage1_patients,
        stage2_patients=study.stage2_patients,
        cohort_size=study.cohort_size,
        trials=2,
        seed=55,
        draws=8,
        warmup=0,
    )
    counts = np.array([t.toxicities for t in result.trials])
    assert result.mean_toxicities == counts.mean()
    assert result.mcse_mean_toxicities == pytest.approx(counts.std(ddof=1) / np.sqrt(2))
    assert "Mean absolute error" in result.to_html(reference_doses=study.reference_doses)
    with pytest.raises(ValueError, match="duplicate"):
        read_toxfinder_tmml(text + "pi_star .2\n")


def test_pending_patients_do_not_advance_observed_stage():
    rows = "\n".join(f"{i},P,2026,10,12,10:00,{i},120,60,{y}" for i, y in enumerate([0, 2, 2]))
    records = read_toxfinder_subject_csv(rows, reference_doses=[1200, 600])
    result = recommend_toxfinder_native(
        records,
        [[0.1, 0.1], [0.5, 0.5], [1, 1]],
        prior=ToxFinderPrior(np.ones(6), np.full(6, 0.25)),
        target=0.3,
        stage1_patients=2,
        stage2_patients=4,
        cohort_size=1,
        seed=4,
        draws=8,
        warmup=0,
    )
    assert result.reason == "stage_one" and result.selected_index == 0
    assert result.observed_patients == 1 and result.pending_patients == 2
