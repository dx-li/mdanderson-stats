"""Native C/F77 evidence, independent erfc identities and wide-domain arithmetic."""

import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdf_normal, cdfnor, cumnor

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_normal.json").read_text())
CASES = [(language, case) for language, p in FIXTURE["profiles"].items() for case in p["cases"]]


@pytest.mark.parametrize("language,case", CASES)
def test_native_c_and_fortran(language, case):
    which, p, q, x, mean, sd = case["input"]
    kwargs = dict(x=x, mean=mean, sd=sd)
    if which != 1:
        kwargs.update(p=p, q=q)
        name = {2: "x", 3: "mean", 4: "sd"}[which]
        kwargs.pop(name)
    r = cdfnor(which, **kwargs)
    assert case["status"] == 0
    np.testing.assert_allclose(
        [r.p, r.q, r.x, r.mean, r.sd], case["result"][:5], rtol=3e-12, atol=2e-13
    )
    if which == 1:
        np.testing.assert_allclose([r.p, r.q], case["result"][:2], rtol=3e-13, atol=0)
    assert FIXTURE["profiles"][language]["adaptations"] == []


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_native_invalid_inputs_and_false_sd_success(language):
    profile = FIXTURE["profiles"][language]
    assert [r["status"] for r in profile["invalid_cases"]] == [-1, -2, -3, 3, -6]
    for case in profile["invalid_sd_results"]:
        assert case["status"] == 0
        _, p, q, x, mean, _ = case["input"]
        if p == 0.5 and x != mean:
            assert case["result"][4] > 1e15  # Rounded native median quantile is nonzero.
        else:
            assert case["result"][4] <= 0
        with pytest.raises(ValueError):
            cdfnor(4, p=p, q=q, x=x, mean=mean)


@pytest.mark.parametrize("z", [-38, -37, -20, -8, -2, -0.1, 0, 0.1, 2, 8, 20, 37, 38])
@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_independent_erfc_all_modes(z, scale):
    p, q = erfc(-z / sqrt(2)) / 2, erfc(z / sqrt(2)) / 2
    mean = scale * 3
    x = mean + scale * z
    r = cdfnor(x=x, mean=mean, sd=scale)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=8e-13, atol=8 * np.nextafter(0.0, 1.0))
    for which, name, truth in [(2, "x", x), (3, "mean", mean), (4, "sd", scale)]:
        if which == 4 and z == 0:
            continue
        kw = dict(p=p, q=q, x=x, mean=mean, sd=scale)
        kw.pop(name)
        np.testing.assert_allclose(
            getattr(cdfnor(which, **kw), name), truth, rtol=2e-9, atol=scale * 2e-9
        )


def test_intermediate_overflow_repaired_in_all_modes():
    p, q = erfc(-2 / sqrt(2)) / 2, erfc(2 / sqrt(2)) / 2
    r = cdfnor(x=1e308, mean=-1e308, sd=1e308)
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-14)
    assert float(cdfnor(2, p=p, q=q, mean=-1e308, sd=1e308).x) == pytest.approx(1e308, rel=3e-14)
    assert float(cdfnor(3, p=p, q=q, x=1e308, sd=1e308).mean) == pytest.approx(-1e308, rel=3e-14)
    assert float(cdfnor(4, p=p, q=q, x=1e308, mean=-1e308).sd) == pytest.approx(1e308, rel=3e-14)
    np.testing.assert_array_equal(
        cdfnor(x=[-1e308, 1e308], mean=[1e308, -1e308], sd=1e-300).p, [0, 1]
    )
    # Non-overflowing subtraction must not evaluate inf-inf in an unused branch.
    with np.errstate(invalid="raise"):
        assert float(cdfnor(x=1e308, mean=1e308, sd=1e-300).p) == 0.5


def test_subnormal_scales_and_probability_tails():
    tiny = np.nextafter(0.0, 1.0)
    r = cdfnor(x=tiny, mean=0, sd=tiny)
    p, q = erfc(-1 / sqrt(2)) / 2, erfc(1 / sqrt(2)) / 2
    np.testing.assert_allclose([r.p, r.q], [p, q], rtol=3e-14)
    assert float(cdfnor(4, p=p, q=q, x=tiny).sd) == tiny
    for prob in [tiny, 1e-320, 1e-300]:
        for lower in [True, False]:
            kw = {"p" if lower else "q": prob}
            r = cdfnor(2, **kw)
            actual = cumnor(r.x)[0 if lower else 1]
            np.testing.assert_allclose(actual, prob, rtol=3e-12, atol=tiny)


def test_broadcasts_defaults_empty_and_immutable_ownership():
    x = np.array([[-2.0], [0.0], [2.0]])
    sd = np.array([0.5, 1.0, 2.0])
    r = cdfnor(x=x, sd=sd)
    f95 = cdf_normal(x=x, sd=sd)
    np.testing.assert_array_equal([r.p, r.q], [f95.cum, f95.ccum])
    np.testing.assert_allclose(cdfnor(2, p=r.p, q=r.q, sd=sd).x, r.x, rtol=3e-14, atol=0)
    x[:] = 7
    sd[:] = 7
    assert r.x[0, 0] == -2 and r.sd[0, 0] == 0.5
    for name in ["p", "q", "x", "mean", "sd"]:
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for w, name in [(1, "p"), (2, "x"), (3, "mean"), (4, "sd")]:
        kw = dict(x=[], mean=[], sd=[])
        if w != 1:
            kw.pop(name)
            kw["p"] = []
        assert getattr(cdfnor(w, **kw), name).size == 0
    np.testing.assert_array_equal(cumnor(0), [0.5, 0.5])


@pytest.mark.parametrize(
    "which,kwargs",
    [
        (True, {}),
        (0, {}),
        (5, {}),
        (1.0, {}),
        (1, {}),
        (1, dict(x=0, p=0.5)),
        (1, dict(x=np.inf)),
        (1, dict(x=0, mean=np.nan)),
        (1, dict(x=0, sd=0)),
        (1, dict(x=0, sd=-1)),
        (2, dict(p=0)),
        (2, dict(q=0)),
        (2, dict(p=0.4, q=0.4)),
        (2, dict(p=0.5, q=0.5 + 4 * np.finfo(float).eps)),
        (2, dict(p=0.5, x=0)),
        (3, dict(p=0.5, x=0, mean=0)),
        (4, dict(p=0.5, x=0, sd=1)),
        (4, dict(p=0.5, x=0)),
        (4, dict(p=0.9, x=-1)),
        (4, dict(p=0.9, x=0)),
        (2, dict(p=0.999, mean=1e308, sd=1e308)),
    ],
)
def test_invalid_inputs_or_unrepresentable_parameters(which, kwargs):
    with pytest.raises(ValueError):
        cdfnor(which, **kwargs)


def test_unresolvable_location_is_not_returned_as_a_valid_quantile():
    with pytest.raises(ArithmeticError, match="forward verification"):
        cdfnor(2, p=0.9, mean=1e300, sd=1)


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_native_wide_domain_defects_are_retained(language):
    cases = FIXTURE["profiles"][language]["wide_cases"]
    assert all(c["status"] == 0 for c in cases)
    assert cases[0]["result"][:2] == ([1.0, 0.0] if language == "c" else ["nan", "nan"])
    assert [cases[i]["result"][i + 1] for i in (1, 2, 3)] == ["inf", "-inf", "inf"]
    assert cases[4]["result"][:2] == [0.0, 1.0]
    assert float(cumnor(-38)[0]) > 0
    small = cases[5]
    r = cdfnor(x=small["input"][3], mean=small["input"][4], sd=small["input"][5])
    np.testing.assert_allclose([r.p, r.q], small["result"][:2], rtol=3e-14)
