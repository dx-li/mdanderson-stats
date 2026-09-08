import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import chi2, f, ncf, ncx2

from mdanderson_stats import RandlibGenerator

FIXTURE = json.loads((Path(__file__).parent / "fixtures/randlib_chi_f.json").read_text())
NAMES = {1: "chi_square", 2: "noncentral_chi_square", 3: "f", 4: "noncentral_f"}


def options(case):
    result = {"df" if case["kind"] < 3 else "dfn": case["dfn"]}
    if case["kind"] >= 3:
        result["dfd"] = case["dfd"]
    if case["kind"] in (2, 4):
        result["noncentrality"] = case["nc"]
    return result


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_values_states_and_truncation(case):
    bank = RandlibGenerator(case["seed"], stream=case["stream"])
    bank.set_antithetic(case["antithetic"])
    method = getattr(bank, NAMES[case["kind"]])
    kwargs = dict(
        **options(case), legacy=True, source="c" if case["language"] == "c" else "fortran"
    )
    values, states = [], []
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        for _ in case["values"]:
            values.append(method(**kwargs)[0])
            states.append(bank.get_seeds())
    assert bool(observed) == case["truncated"]
    assert all("truncated to 1e37" in str(w.message) for w in observed)
    assert_allclose(
        values,
        np.array(case["values"], dtype=np.float32).astype(float),
        rtol=5e-6,
        atol=2 * float(np.finfo(np.float32).smallest_subnormal),
    )
    assert_array_equal(states, case["states"])
    bank.reinitialize()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert_array_equal(method(len(values), **kwargs), values)
    assert bank.get_seeds() == tuple(case["states"][-1])


@pytest.mark.parametrize(
    "name,kwargs,distribution",
    [
        ("chi_square", {"df": 5}, chi2(5)),
        ("noncentral_chi_square", {"df": 5, "noncentrality": 2.3}, ncx2(5, 2.3)),
        ("f", {"dfn": 5, "dfd": 12}, f(5, 12)),
        ("noncentral_f", {"dfn": 5, "dfd": 12, "noncentrality": 2.3}, ncf(5, 12, 2.3)),
    ],
)
@pytest.mark.parametrize("legacy,source", [(False, "fortran"), (True, "fortran"), (True, "c")])
def test_distribution(name, kwargs, distribution, legacy, source):
    bank = RandlibGenerator()
    values = getattr(bank, name)(20000, **kwargs, legacy=legacy, source=source)
    assert np.all(values >= 0)
    assert abs(values.mean() - distribution.mean()) < 6 * np.sqrt(distribution.var() / len(values))
    for q in [0.1, 0.5, 0.9]:
        assert abs(np.mean(values <= distribution.ppf(q)) - q) < 0.015
    assert not values.flags.writeable


@pytest.mark.parametrize(
    "name,kwargs,distribution",
    [
        ("chi_square", {"df": 0.5}, chi2(0.5)),
        ("noncentral_chi_square", {"df": 0.5, "noncentrality": 2.3}, ncx2(0.5, 2.3)),
        ("f", {"dfn": 0.5, "dfd": 2}, f(0.5, 2)),
        ("noncentral_f", {"dfn": 0.5, "dfd": 2, "noncentrality": 2.3}, ncf(0.5, 2, 2.3)),
    ],
)
def test_modern_inverse_cdf_and_consumption(name, kwargs, distribution):
    bank, reference = RandlibGenerator(), RandlibGenerator()
    values = getattr(bank, name)(100, **kwargs)
    assert_allclose(distribution.cdf(values), reference.uniform(100), atol=3e-13, rtol=0)
    assert bank.get_seeds() == reference.get_seeds()
    bank.reinitialize()
    assert_array_equal(values, [getattr(bank, name)(**kwargs)[0] for _ in values])


@pytest.mark.parametrize("name", NAMES.values())
@pytest.mark.parametrize("legacy", [False, True])
def test_limits_empty_and_state_rollback(name, legacy):
    bank = RandlibGenerator()
    method = getattr(bank, name)
    before = bank.get_seeds()
    assert method(0, legacy=legacy).size == 0
    assert bank.get_seeds() == before
    with pytest.raises(ArithmeticError, match="max_attempts"):
        method(20, legacy=legacy, max_attempts=1)
    assert bank.get_seeds() == before


@pytest.mark.parametrize("name", ["f", "noncentral_f"])
def test_legacy_truncation_and_modern_overflow(name):
    bank = RandlibGenerator()
    method = getattr(bank, name)
    before = bank.get_seeds()
    with pytest.raises(ArithmeticError, match="nonfinite"):
        method(20, dfn=5, dfd=0.0001)
    assert bank.get_seeds() == before
    with pytest.warns(RuntimeWarning, match="truncated to 1e37"):
        values = method(20, dfn=5, dfd=0.0001, legacy=True)
    assert np.any(values == float(np.float32(1e37)))
    assert bank.get_seeds() != before
    bank.reinitialize()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="truncated"):
            method(20, dfn=5, dfd=0.0001, legacy=True)
    assert bank.get_seeds() == before


@pytest.mark.parametrize(
    "central,noncentral,kwargs",
    [
        ("chi_square", "noncentral_chi_square", {"df": 5}),
        ("f", "noncentral_f", {"dfn": 5, "dfd": 12}),
    ],
)
def test_zero_noncentrality_modern_identity(central, noncentral, kwargs):
    a, b = RandlibGenerator(), RandlibGenerator()
    assert_array_equal(
        getattr(a, central)(100, **kwargs), getattr(b, noncentral)(100, noncentrality=0, **kwargs)
    )
    assert a.get_seeds() == b.get_seeds()


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("chi_square", {"df": 0}),
        ("chi_square", {"df": float("nan")}),
        ("chi_square", {"df": 1e-300, "legacy": True}),
        ("chi_square", {"df": float(np.finfo(np.float32).smallest_subnormal), "legacy": True}),
        ("chi_square", {"df": 1e300, "legacy": True}),
        ("noncentral_chi_square", {"df": 0.5, "legacy": True}),
        ("noncentral_chi_square", {"noncentrality": -1}),
        ("noncentral_chi_square", {"noncentrality": None}),
        ("noncentral_chi_square", {"noncentrality": float("inf")}),
        ("f", {"dfd": None}),
        ("f", {"dfd": 0}),
        ("f", {"dfn": -1}),
        ("noncentral_f", {"dfn": 0.5, "legacy": True}),
        ("noncentral_f", {"noncentrality": None}),
        ("noncentral_f", {"dfd": None}),
        ("f", {"legacy": 1}),
        ("chi_square", {"source": "c"}),
        ("f", {"max_attempts": 0}),
        ("f", {"size": True}),
    ],
)
def test_invalid_requests(name, kwargs):
    bank = RandlibGenerator()
    before = bank.get_seeds()
    with pytest.raises(ValueError):
        getattr(bank, name)(**kwargs)
    assert bank.get_seeds() == before
