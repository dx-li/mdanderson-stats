import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.crm_prior_ess import _posterior_moments as complete_moments
from mdanderson_stats.tite_crm_prior_ess import (
    _source_second_derivative,
    simulate_tite_crm_prior_ess,
)


def test_tite_posterior_reduces_to_complete_crm_at_full_followup():
    result = simulate_tite_crm_prior_ess(
        [0.2, 0.4],
        [0.3, 0.5],
        0.3,
        max_patients=2,
        replications=1,
        obswin=1,
        rate=1,
        accrual="fixed",
        criterion="followup",
        assessment_delay=0,
        outcome_uniforms=[[0.9, 0.9]],
        event_time_uniforms=[[0.2, 0.2]],
        beta_sd=1.0,
    )
    expected = complete_moments(
        result.dose_indices[0, :1],
        result.latent_outcomes[0, :1],
        result.skeleton,
        result.beta_sd,
        "full",
    )[0]
    assert result.beta_mean_before_patient[0, 1] == pytest.approx(expected, abs=2e-6)
    assert result.observed_outcomes.tolist() == [[0, 0]]
    assert result.assessment_weights.tolist() == [[1.0, 0.0]]


def test_followup_criterion_does_not_leak_a_pending_late_dlt():
    result = simulate_tite_crm_prior_ess(
        [0.2],
        [0.3],
        0.25,
        max_patients=2,
        replications=1,
        obswin=10,
        rate=20,
        accrual="fixed",
        criterion="followup",
        assessment_delay=0,
        outcome_uniforms=[[0.0, 0.0]],
        event_time_uniforms=[[0.9, 0.9]],
        beta_sd=1.0,
    )
    # Fixed arrivals are .5 and 1.0; assessment is at the last arrival.
    # Both latent DLTs occur .9 time units after their enrollment, so neither
    # has occurred at this cutoff. A latent-outcome leak would report [1, 1].
    assert result.latent_outcomes.tolist() == [[1, 1]]
    assert result.observed_outcomes.tolist() == [[0, 0]]
    assert result.assessment_weights.tolist() == [[0.05, 0.0]]


def test_native_arrival_criterion_keeps_source_ess_weight_distinct():
    result = simulate_tite_crm_prior_ess(
        [0.9],
        [0.3],
        0.25,
        max_patients=2,
        replications=1,
        obswin=1,
        rate=1,
        accrual="fixed",
        criterion="native_arrival",
        outcome_uniforms=[[0.0, 0.0]],
        event_time_uniforms=[[0.8, 0.8]],
        beta_sd=1.0,
    )
    assert result.latent_outcomes.tolist() == [[1, 1]]
    assert result.observed_outcomes.tolist() == [[1, 1]]
    assert result.assessment_weights.tolist() == [[1.0, 1.0]]
    assert result.criterion == "native_arrival"


def test_poisson_uniform_transform_and_signed_source_curvature():
    result = simulate_tite_crm_prior_ess(
        [0.2],
        [0.3],
        0.25,
        max_patients=2,
        replications=1,
        obswin=2,
        rate=1,
        accrual="poisson",
        criterion="followup",
        assessment_delay=0,
        arrival_uniforms=[[np.exp(-0.5), np.exp(-1.25)]],
        outcome_uniforms=[[0.9, 0.9]],
        event_time_uniforms=[[0.2, 0.2]],
        beta_sd=1.0,
    )
    np.testing.assert_allclose(result.arrivals[0], [1.0, 3.5], rtol=0, atol=1e-15)
    positive = _source_second_derivative(0.8, 0.01, 0)
    assert positive > 0.0
    assert _source_second_derivative(0.3, 0.4, 1) < 0.0
    reference = _fixture_rows("tite-crm-prior-ess-curvature-sign-example.csv")[0]
    assert positive == pytest.approx(float(reference["getDiff"]), abs=2e-17)


def test_followup_requires_explicit_delay_and_work_is_preflighted():
    with pytest.raises(ValueError, match="assessment_delay is required"):
        simulate_tite_crm_prior_ess(
            [0.2],
            [0.3],
            0.25,
            max_patients=2,
            replications=1,
            obswin=1,
            rate=1,
            accrual="fixed",
            criterion="followup",
        )
    with pytest.raises(ValueError, match="work exceeds"):
        simulate_tite_crm_prior_ess(
            [0.2],
            [0.3],
            0.25,
            max_patients=2,
            replications=1,
            obswin=1,
            rate=1,
            accrual="fixed",
            criterion="native_arrival",
            outcome_uniforms=[[0.5, 0.5]],
            event_time_uniforms=[[0.5, 0.5]],
            max_work=1,
        )


def _fixture_rows(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_scalar_start_tite_path_matches_pinned_source_and_asof_ledger():
    path_rows = _fixture_rows("tite-crm-prior-ess-native-onetite-paths.csv")
    ledger_rows = _fixture_rows("tite-crm-prior-ess-followup-ledger.csv")
    curvature_rows = _fixture_rows("tite-crm-prior-ess-curvature.csv")
    summaries = _fixture_rows("tite-crm-prior-ess-native-onetite-summary.csv")
    for case in ("fixed", "poisson"):
        path = [row for row in path_rows if row["case"] == case]
        ledger = [row for row in ledger_rows if row["case"] == case]
        curv = [row for row in curvature_rows if row["case"] == case]
        summary = next(row for row in summaries if row["case"] == case)
        arrivals = np.array([float(row["arrival"]) for row in path])
        outcome_tape = np.array([[float(row["outcome_uniform"]) for row in path]])
        event_tape = np.array(
            [[float(row["toxicity_delay"]) / 10 if row["tox"] == "1" else 0.5 for row in path]]
        )
        accrual = "fixed" if case == "fixed" else "poisson"
        shared = dict(
            true_toxicity=[0.05, 0.15, 0.30, 0.45],
            skeleton=[0.05, 0.10, 0.20, 0.35],
            target=0.20,
            max_patients=8,
            replications=1,
            obswin=10,
            rate=2,
            accrual=accrual,
            outcome_uniforms=outcome_tape,
            event_time_uniforms=event_tape,
            beta_sd=np.sqrt(1.34),
        )
        if accrual == "poisson":
            intervals = np.diff(np.r_[0.0, arrivals])
            shared["arrival_uniforms"] = np.exp(-intervals[None, :] / 5.0)
        native = simulate_tite_crm_prior_ess(**shared, criterion="native_arrival")
        assert native.dose_indices[0].tolist() == [int(row["dose"]) for row in path]
        assert native.latent_outcomes[0].tolist() == [int(row["tox"]) for row in path]
        np.testing.assert_allclose(native.arrivals[0], arrivals, rtol=0, atol=2e-14)
        np.testing.assert_allclose(
            native.beta_mean_before_patient[0],
            [float(row["beta_before"]) for row in path],
            rtol=0,
            atol=3e-9,
        )
        assert native.final_beta_mean[0] == pytest.approx(
            float(summary["final_beta_mean"]), abs=3e-9
        )
        assert native.selected_dose[0] == int(summary["final_mtd"])
        assert native.replication_second_derivative[0] == pytest.approx(
            float(summary["source_curvature_sum"]), abs=2e-12
        )

        followup = simulate_tite_crm_prior_ess(**shared, criterion="followup", assessment_delay=3)
        assert followup.observed_outcomes[0].tolist() == [
            int(row["observed_tox"]) for row in ledger
        ]
        np.testing.assert_allclose(
            followup.assessment_weights[0],
            [float(row["valid_linear_weight"]) for row in ledger],
            rtol=0,
            atol=2e-15,
        )
        expected_followup_curvature = sum(float(row["observed_followup_getDiff"]) for row in curv)
        assert followup.replication_second_derivative[0] == pytest.approx(
            expected_followup_curvature, abs=2e-12
        )


def test_complete_non_dlt_curvature_is_stable_near_probability_one():
    from decimal import Decimal, localcontext

    probability = np.nextafter(1.0, 0.0)
    with localcontext() as context:
        context.prec = 90
        p = Decimal.from_float(float(probability))
        t = -p.ln()
        ratio = p * t / (Decimal(1) - p)
        expected = float(ratio * (Decimal(1) - t - ratio))
    actual = _source_second_derivative(float(probability), 1.0, 0)
    assert actual == pytest.approx(expected, rel=3e-15, abs=0.0)
