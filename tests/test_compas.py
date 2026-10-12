import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logit, logsumexp, ndtr

from mdanderson_stats import (
    ComPASStudy,
    compas_decisions,
    compas_pnoc022_variances,
    compas_variances,
)

V = compas_pnoc022_variances("cohorts_1_2")
Q = (0.2,) * 5


def study(**kwargs):
    params = dict(
        completed=(6, 7),
        successes=(2, 5),
        groups=("a", "a"),
        variances=V,
        model_prior=Q,
        null_response=0.4,
        equality_policy="hold",
    )
    params.update(kwargs)
    return ComPASStudy(**params)


@pytest.fixture(scope="module")
def fitted():
    return study().fit()


def test_reference_fit(fitted):
    refs = json.loads(
        (Path(__file__).parent / "fixtures/compas-marginal-reference.json").read_text()
    )[:5]
    np.testing.assert_allclose(
        fitted.log_evidence, [r["log_evidence"] for r in refs], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        fitted.tail_probabilities, [r["tails"] for r in refs], rtol=0, atol=1e-10
    )
    logweights = np.array([r["log_evidence"] for r in refs]) + math.log(0.2)
    probs = np.exp(logweights - logsumexp(logweights))
    np.testing.assert_allclose(fitted.model_probabilities, probs, rtol=0, atol=1e-11)
    assert fitted.selected_model == int(np.argmax(probs))
    assert json.loads(fitted.to_json())["limitations"]
    assert all(
        lo <= p <= hi
        for (lo, hi), p in zip(fitted.model_probability_bounds, fitted.model_probabilities)
    )


def test_zero_data_exact():
    s = study(completed=(0, 0), successes=(0, 0))
    f = s.fit()
    assert f.selected_model is None and f.decisions == ("numerical_hold",) * 2
    assert f.log_evidence == (0.0,) * 5
    assert f.model_probabilities == pytest.approx(Q)
    for v, p in zip(V, f.tail_probabilities):
        assert p == pytest.approx((ndtr(-logit(0.4) / math.sqrt(1e6 + v)),) * 2, abs=1e-15)
    f = study(completed=(0, 0), successes=(0, 0), model_prior=(1.0, 0.0, 0.0, 0.0, 0.0)).fit()
    assert f.selected_model == 0 and f.decisions == ("continue",) * 2


def test_json():
    s = study()
    assert s == ComPASStudy.from_json(s.to_json())
    for key, value in [
        ("schema_version", True),
        ("profile", "other"),
        ("completed", [True, 2]),
        ("model_prior", [1]),
        ("groups", ["a"]),
        ("extra", 1),
    ]:
        d = json.loads(s.to_json())
        d[key] = value
        with pytest.raises(ValueError):
            ComPASStudy.from_json(json.dumps(d))


@pytest.mark.parametrize(
    "key,value",
    [
        ("completed", ()),
        ("completed", (101, 7)),
        ("successes", (7, 5)),
        ("groups", ("", "a")),
        ("variances", (0.0,) * 5),
        ("variances", (float("nan"),) * 5),
        ("model_prior", (0.1,) * 5),
        ("model_prior", (-1.0, 0.5, 0.5, 0.5, 0.5)),
        ("null_response", 0),
        ("null_response", True),
        ("equality_policy", "guess"),
        ("quadrature_tolerance", 1e-12),
    ],
)
def test_invalid(key, value):
    with pytest.raises(ValueError):
        study(**{key: value})


def test_rules():
    assert compas_decisions([0, 0.099, 0.1, 0.5, 0.95, 0.951, 1], equality_policy="hold") == (
        "drop",
        "drop",
        "hold",
        "continue",
        "hold",
        "graduate",
        "graduate",
    )
    assert compas_decisions([0.1, 0.95], equality_policy="continue") == ("continue", "continue")


@pytest.mark.parametrize("n,p", [(33, 0.6), (33, 0.65), (42, 0.7), (70, 0.7)])
def test_calibration(n, p):
    assert compas_variances(n, p) == pytest.approx(
        tuple((1 - w) / (n * p * (1 - p) * w) for w in (0.999, 0.75, 0.5, 0.25, 0.001))
    )


@pytest.mark.parametrize(
    "n,p", [(0, 0.4), (True, 0.4), (33, 0), (33, 1), (33, True), (33, float("nan")), (1, 5e-324)]
)
def test_invalid_calibration(n, p):
    with pytest.raises(ValueError):
        compas_variances(n, p)


@pytest.mark.parametrize("successes", [(0, 0), (6, 7), (3, 4)])
def test_reflection(successes):
    s = study(successes=successes, null_response=0.5)
    f = s.fit()
    g = study(
        successes=tuple(n - y for n, y in zip(s.completed, s.successes)), null_response=0.5
    ).fit()
    np.testing.assert_allclose(f.log_evidence, g.log_evidence, atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        np.array(f.tail_probabilities) + g.tail_probabilities, 1, atol=1e-8, rtol=0
    )


def test_permutation(fitted):
    f = study(completed=(7, 6), successes=(5, 2)).fit()
    np.testing.assert_allclose(f.log_evidence, fitted.log_evidence, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        f.tail_probabilities, np.array(fitted.tail_probabilities)[:, ::-1], atol=1e-10, rtol=0
    )


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "fixtures/compas-marginal-reference.json").read_text()),
)
def test_all_independent_two_arm_references(case):
    from mdanderson_stats._compas_quadrature import compas_group_integral

    z, p, e, pe, _ = compas_group_integral(
        tuple(case["n"]), tuple(case["y"]), case["variance"], case["p0"], 1e-7
    )
    assert z == pytest.approx(case["log_evidence"], abs=1e-10)
    assert p == pytest.approx(case["tails"], abs=1e-10)
    assert e < 1e-7 and max(pe) < 1e-7


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "fixtures/compas-three-arm-reference.json").read_text()),
)
def test_independent_three_arm_evidence(case):
    from mdanderson_stats._compas_quadrature import compas_group_integral

    z, _, _, _, _ = compas_group_integral(
        tuple(case["n"]), tuple(case["y"]), case["variance"], 0.4, 1e-7
    )
    assert z == pytest.approx(case["log_evidence"], abs=1e-9)
    assert case["refinement_difference"] < 1e-10


def test_group_factorization():
    a = study(completed=(4,), successes=(1,), groups=("a",)).fit()
    b = study(completed=(5,), successes=(3,), groups=("b",)).fit()
    combined = study(completed=(4, 5), successes=(1, 3), groups=("a", "b")).fit()
    np.testing.assert_allclose(
        combined.log_evidence, np.array(a.log_evidence) + b.log_evidence, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.array(combined.tail_probabilities)[:, 0],
        np.array(a.tail_probabilities)[:, 0],
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.array(combined.tail_probabilities)[:, 1],
        np.array(b.tail_probabilities)[:, 0],
        rtol=0,
        atol=1e-10,
    )


def test_quality_failure_propagates(monkeypatch):
    import mdanderson_stats.compas as compas_module

    def fail(*args, **kwargs):
        raise ArithmeticError("deliberate unresolved integration")

    monkeypatch.setattr(compas_module, "compas_group_integral", fail)
    with pytest.raises(ArithmeticError, match="unresolved"):
        study().fit()


def test_threshold_uncertainty_holds(monkeypatch):
    import mdanderson_stats.compas as compas_module

    def uncertain(*args, **kwargs):
        return 0.0, (0.1, 0.95), 1e-8, (1e-8, 1e-8), 1

    monkeypatch.setattr(compas_module, "compas_group_integral", uncertain)
    f = study(model_prior=(1.0, 0.0, 0.0, 0.0, 0.0), equality_policy="continue").fit()
    assert f.selected_model == 0 and f.decisions == ("numerical_hold",) * 2


def test_input_ownership():
    completed = [6, 7]
    s = study(completed=completed)
    completed[0] = 0
    assert s.completed == (6, 7)


@pytest.mark.parametrize(
    "text", ["{}", "[]", "null", "{", '{"profile":1,"profile":2}', " " * 32769, "[" * 1100]
)
def test_bad_json(text):
    with pytest.raises(ValueError):
        ComPASStudy.from_json(text)


def test_unobserved_arm_does_not_change_evidence(fitted):
    f = study(completed=(6, 7, 0), successes=(2, 5, 0), groups=("a", "a", "a")).fit()
    np.testing.assert_allclose(f.log_evidence, fitted.log_evidence, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        np.array(f.tail_probabilities)[:, :2], fitted.tail_probabilities, rtol=0, atol=1e-10
    )
    assert all(0 <= p[2] <= 1 for p in f.tail_probabilities)


def test_numerical_model_tie_holds(monkeypatch):
    import mdanderson_stats.compas as compas_module

    def near(*args, **kwargs):
        return 0.0, (0.01, 0.99), 1e-7, (1e-8, 1e-8), 1

    monkeypatch.setattr(compas_module, "compas_group_integral", near)
    f = study(model_prior=(0.200000001, 0.199999999, 0.2, 0.2, 0.2)).fit()
    assert f.selected_model is None and f.decisions == ("numerical_hold",) * 2


def test_report_json_is_finite(fitted):
    def bad_constant(value):
        raise AssertionError(value)

    report = json.loads(fitted.to_json(), parse_constant=bad_constant)
    assert report["study"]["model_prior"] == list(Q)
    assert (
        report["protocol_sha256"]
        == "fa69432cb8e68abec0b206a3e0f753c7cb2b0e40827651a99fc18411e5c37e80"
    )


def test_reported_variances_keep_protocol_discrepancies():
    assert compas_pnoc022_variances("cohorts_1_2") == (0.00012, 0.042, 0.126, 0.379, 126.1)
    assert compas_pnoc022_variances("cohort_3") == (0.00007, 0.023, 0.068, 0.204, 67.96)
    assert compas_pnoc022_variances("cohort_3") != compas_variances(42, 0.7)
    assert compas_pnoc022_variances("cohorts_1_2") != compas_variances(33, 0.65)


@pytest.mark.parametrize(
    "n,successes,expected",
    [
        (11, (0, 0, 0), ("drop",) * 3),
        (11, (11, 11, 11), ("graduate",) * 3),
        (33, (0, 0, 0), ("drop",) * 3),
        (33, (33, 33, 33), ("graduate",) * 3),
        (33, (0, 16, 33), ("drop", "continue", "graduate")),
        (42, (18, 21, 31), ("continue", "continue", "graduate")),
    ],
)
def test_protocol_sample_size_extremes(n, successes, expected):
    fit = study(
        completed=(n,) * 3,
        successes=successes,
        groups=("a",) * 3,
        variances=compas_pnoc022_variances("cohort_3" if n == 42 else "cohorts_1_2"),
        null_response=0.5 if n == 42 else 0.4,
    ).fit()
    assert fit.decisions == expected
    assert max(max(errors) for errors in fit.tail_error_estimates) < 1e-7


@pytest.mark.parametrize("threshold", [0.000001, 0.001, 0.5, 0.999, 0.999999])
def test_six_arm_evidence_is_threshold_invariant_against_independent_reference(threshold):
    from mdanderson_stats._compas_quadrature import compas_group_integral

    reference = json.loads(
        (Path(__file__).parent / "fixtures/compas-six-arm-reference.json").read_text()
    )
    refinement = reference["refinements"]
    assert abs(refinement[-1]["log_evidence"] - refinement[-2]["log_evidence"]) < 1e-9
    result = compas_group_integral((100,) * 6, (50,) * 6, 1000.0, threshold, 1e-7)
    assert result[0] == pytest.approx(refinement[-1]["log_evidence"], abs=1e-9)
    if threshold == 0.5:
        assert result[1] == pytest.approx((0.5,) * 6, abs=1e-12)


@pytest.mark.parametrize(
    "case_index,threshold", [(0, 0.5), (0, 0.95), (0, 0.999), (1, 0.5), (1, 5e-324)]
)
def test_distant_threshold_splits_cannot_hide_outer_mode_mass(case_index, threshold):
    from mdanderson_stats._compas_quadrature import compas_group_integral

    case = json.loads(
        (Path(__file__).parent / "fixtures/compas-extreme-threshold-reference.json").read_text()
    )["cases"][case_index]
    refs = case["refinements"]
    # Fixed-node order and domain refinements independently check the denominator.
    assert abs(refs[-1]["log_evidence"] - refs[-2]["log_evidence"]) < 1e-10
    result = compas_group_integral(
        tuple(case["completed"]), tuple(case["successes"]), case["variance"], threshold, 1e-7
    )
    assert result[0] == pytest.approx(refs[-1]["log_evidence"], abs=1e-9)
    if case_index == 1 and threshold == 5e-324:
        assert result[1] == pytest.approx((1.0,) * 6, abs=1e-12)
