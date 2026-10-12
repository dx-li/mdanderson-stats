import csv
import io
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ndtri

from mdanderson_stats import PRTPatient, PRTStudy, read_prt_study
from mdanderson_stats.hierarchical_binomial import summarize_chains
from mdanderson_stats.prt import prt_conditional_toxicity
from mdanderson_stats.prt_conduct import analyze_prt_conduct
from mdanderson_stats.prt_fit import PRTModelFit


def study():
    return PRTStudy(
        3,
        3,
        patients=(PRTPatient(0, 3, True), PRTPatient(1, 3, True)),
        draws=2048,
        warmup=100,
        chains=4,
    )


def synthetic_fit(shift=0):
    # Ordered, separated dose risks; no projection pooling, nondegenerate moments.
    rng = np.random.default_rng(69421)
    beta = ndtri(np.array([0.01, 0.05, 0.2])) + 0.08 * rng.normal(size=(4, 2048, 3, 3))
    beta[0] += shift
    risks = prt_conditional_toxicity(beta)
    return PRTModelFit(
        beta,
        risks,
        summarize_chains(beta),
        summarize_chains(risks[..., :-1, :]),
        np.full(3, -14.0),
        np.full(3, 28.0),
        1000,
        100,
    )


@pytest.mark.parametrize("profile,rule", [("paper_2008", "6.1"), ("legacy_1_0", "6.1001")])
def test_saved_history_reopen_edit_and_report(monkeypatch, profile, rule):
    fit = synthetic_fit()
    monkeypatch.setattr("mdanderson_stats.prt_conduct.fit_prt_model", lambda *a, **kw: fit)
    saved = read_prt_study(study().to_json())
    result = analyze_prt_conduct(saved, rng=np.random.default_rng(6), profile=profile)
    assert (result.status, result.action, result.dose, result.rule) == ("computed", "stay", 1, rule)
    assert result.chain_decisions_agree
    data = json.loads(result.to_json())
    assert read_prt_study(json.dumps(data["study"])) == saved
    assert data["rng_state"]["bit_generator"] == "PCG64"
    rows = list(csv.DictReader(io.StringIO(result.to_csv())))
    assert len(rows) == 3 and rows[1]["assigned"] == "1"
    assert data["posterior_exceedance"] == [0.0, 0.0, 1.0]
    np.testing.assert_allclose(
        result.posterior_mean, fit.conditional_toxicity[:, :, 0, :].mean(axis=(0, 1))
    )
    edited = replace(saved, patients=(*saved.patients, PRTPatient(0, 3, True)))
    reanalysis = analyze_prt_conduct(
        read_prt_study(edited.to_json()), rng=np.random.default_rng(6), profile=profile
    )
    assert reanalysis.assigned == (2, 1, 0)
    assert saved.patients[-1].dose == 1 and edited.patients[-1].dose == 0


def test_legacy_missing_output_is_not_converted_into_recommendation(monkeypatch):
    monkeypatch.setattr(
        "mdanderson_stats.prt_conduct.fit_prt_model", lambda *a, **kw: synthetic_fit()
    )
    inputs = replace(study(), patients=(PRTPatient(0, 3, True),))
    result = analyze_prt_conduct(inputs, rng=np.random.default_rng(1), profile="legacy_1_0")
    assert result.rule == "6.0"
    assert result.provisional_action == "legacy_no_output"
    assert result.action == "hold" and result.dose is None
    assert "no matching" in " ".join(result.warnings)


def test_poor_mixing_keeps_provisional_decision_on_hold(monkeypatch):
    fit = synthetic_fit()
    fit = replace(fit, beta_summary=replace(fit.beta_summary, split_rhat=np.full((3, 3), 1.2)))
    monkeypatch.setattr("mdanderson_stats.prt_conduct.fit_prt_model", lambda *a, **kw: fit)
    result = analyze_prt_conduct(study(), rng=np.random.default_rng(1))
    assert result.max_split_rhat > 1.05
    assert result.provisional_action == "stay"
    assert result.status == "numerical_hold" and result.dose is None


def test_invalid_projection_and_fit_failures_have_json_reports(monkeypatch):
    def failure(*a, **kw):
        raise ArithmeticError("projection outside probability range")

    monkeypatch.setattr("mdanderson_stats.prt_conduct.fit_prt_model", failure)
    result = analyze_prt_conduct(study(), rng=np.random.default_rng(1))
    assert result.status == "numerical_hold" and result.posterior_mean == ()
    assert len(list(csv.DictReader(io.StringIO(result.to_csv())))) == 3
    assert json.loads(result.to_json())["max_split_rhat"] is None
    assert any("outside probability" in s for s in result.warnings)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile": "native"},
        {"max_rhat": 1},
        {"max_rhat": 1.2},
        {"max_mcse": 0},
        {"max_mcse": 0.1},
        {"max_likelihood_evaluations": False},
    ],
)
def test_invalid_configuration_does_not_consume_randomness(kwargs):
    rng = np.random.default_rng(1)
    before = repr(rng.bit_generator.state)
    with pytest.raises(ValueError):
        analyze_prt_conduct(study(), rng=rng, **kwargs)
    assert repr(rng.bit_generator.state) == before


def test_work_bound_and_empty_history_rejected_before_rng():
    for inputs in (
        replace(study(), patients=()),
        replace(study(), draws=100000),
        replace(study(), patients=tuple(PRTPatient(1, 0, True) for _ in range(11))),
    ):
        rng = np.random.default_rng(1)
        before = repr(rng.bit_generator.state)
        with pytest.raises(ValueError):
            analyze_prt_conduct(inputs, rng=rng, profile="legacy_1_0")
        assert repr(rng.bit_generator.state) == before


def test_actual_sampler_reproducible_and_reported():
    inputs = replace(study(), prior_mean=-1.5, prior_variance=0.05, draws=128, warmup=100)
    first = analyze_prt_conduct(inputs, rng=np.random.default_rng(123), profile="legacy_1_0")
    second = analyze_prt_conduct(
        read_prt_study(inputs.to_json()), rng=np.random.default_rng(123), profile="legacy_1_0"
    )
    assert first.to_json() == second.to_json()
    assert first.likelihood_evaluations > 0
    assert first.status in ("numerical_hold", "computed")


def test_native_deescalation_without_exact_dose_is_a_hold(monkeypatch):
    monkeypatch.setattr(
        "mdanderson_stats.prt_conduct.fit_prt_model", lambda *a, **kw: synthetic_fit()
    )
    inputs = replace(study(), patients=(*study().patients, PRTPatient(2, 3, True)))
    result = analyze_prt_conduct(inputs, rng=np.random.default_rng(1), profile="legacy_1_0")
    assert (result.rule, result.provisional_action) == ("4", "deescalate")
    assert result.action == "hold" and result.dose is None
    assert result.provisional_dose is None
    assert "exact lower dose" in " ".join(result.warnings)


def test_chain_reprojection_failure_prevents_recommendation(monkeypatch):
    from mdanderson_stats.prt_fit import prt_isotonic_projection

    monkeypatch.setattr(
        "mdanderson_stats.prt_conduct.fit_prt_model", lambda *a, **kw: synthetic_fit()
    )

    def project(raw):
        if len(raw) == study().draws:
            raise ArithmeticError("chain covariance invalid")
        return prt_isotonic_projection(raw)

    monkeypatch.setattr("mdanderson_stats.prt_conduct.prt_isotonic_projection", project)
    result = analyze_prt_conduct(study(), rng=np.random.default_rng(1))
    assert result.max_split_rhat < 1.05 and result.max_risk_mcse < 0.02
    assert not result.chain_decisions_agree
    assert result.action == "hold" and result.provisional_action == "stay"


def test_non_scalar_rng_state_is_preserved(monkeypatch):
    monkeypatch.setattr(
        "mdanderson_stats.prt_conduct.fit_prt_model", lambda *a, **kw: synthetic_fit()
    )
    rng = np.random.Generator(np.random.MT19937(19))
    initial = rng.bit_generator.state["state"]["key"].copy()
    result = analyze_prt_conduct(study(), rng=rng)
    state = json.loads(result.to_json())["rng_state"]
    np.testing.assert_array_equal(state["state"]["key"], initial)


def test_excessive_likelihood_work_rejected_before_sampling():
    rng = np.random.default_rng(1)
    state = repr(rng.bit_generator.state)
    with pytest.raises(ValueError, match="budget"):
        analyze_prt_conduct(study(), rng=rng, max_likelihood_evaluations=20000000)
    assert repr(rng.bit_generator.state) == state
