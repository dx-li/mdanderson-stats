"""Native metadata, first-error precedence and batch/root adapter behavior."""

import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdflib_aux as aux
from mdanderson_stats import interval_zf, set_zero_finder

REFERENCE = json.loads((Path(__file__).parent / "fixtures/cdflib_aux.json").read_text())


@pytest.mark.parametrize("name,d", REFERENCE["distributions"].items())
def test_all_native_descriptors_and_defined_failure_statuses(name, d):
    native = {k: d[k] for k in ("name", "max_which", "nparam", "parameters")}
    python = asdict(aux.DISTRIBUTIONS[name])
    python["parameters"] = list(python["parameters"])
    assert python == native
    assert (
        aux.validate_parameters(aux.DISTRIBUTIONS[name], 0, np.zeros(6))
        == d["invalid_which_status"]
    )
    assert aux.validate_parameters(aux.DISTRIBUTIONS[name], 2, np.zeros(6)) == d["zero_pair_status"]


@pytest.mark.parametrize("case", REFERENCE["ranges"])
def test_native_range_contract_and_nan_repair(case):
    if case["value"] == "NaN":
        assert case["ok"] is True
        with pytest.raises(ValueError):
            aux.dbl_in_range(float("nan"), 0, 1)
    else:
        assert bool(aux.dbl_in_range(case["value"], 0, 1)) == case["ok"]


@pytest.mark.parametrize("case", REFERENCE["complements"])
def test_native_sum_contract_does_not_imply_probability_range(case):
    assert bool(aux.add_to_one(case["x"], case["y"])) == case["ok"]


def valid_row(d):
    values = np.array([(p.low_bound / 2 + p.high_bound / 2) for p in d.parameters])
    values[:2] = 0.5
    return values


SELECTORS = [(d, which) for d in aux.DISTRIBUTIONS.values() for which in range(1, d.max_which + 1)]


@pytest.mark.parametrize("d,which", SELECTORS)
def test_valid_rows_have_explicit_success_and_ignore_unknowns(d, which):
    values = valid_row(d)
    for i, p in enumerate(d.parameters):
        if i >= d.nparam or which == p.no_check:
            values[i] = np.nan
    result = aux.validate_parameters(d, which, values)
    assert result == 0
    assert result.dtype == np.int64
    with pytest.raises(ValueError):
        result.setflags(write=True)


BOUNDS = [
    (d, i, side)
    for d in aux.DISTRIBUTIONS.values()
    if d.max_which > 0
    for i in range(2, d.nparam)
    for side in (-1, 1)
]


@pytest.mark.parametrize("d,i,side", BOUNDS)
def test_inclusive_bounds_and_first_parameter_failure(d, i, side):
    values = valid_row(d)
    p = d.parameters[i]
    bound = p.low_bound if side < 0 else p.high_bound
    values[i] = bound
    assert aux.validate_parameters(d, 1, values) == 0
    values[i] = np.nextafter(bound, -np.inf if side < 0 else np.inf)
    assert aux.validate_parameters(d, 1, values) == -(i + 2)
    values[i:] = np.nan
    assert aux.validate_parameters(d, 1, values) == -(i + 2)


def test_batch_precedence_and_owned_results():
    values = np.tile(valid_row(aux.the_beta), (5, 1))
    values[1, :2] = 0
    values[2, 4] = -1
    values[3, 5] = np.inf
    values[4, 2] = np.nan  # unknown coordinate for selector 2
    result = aux.validate_parameters(aux.the_beta, 2, values)
    np.testing.assert_array_equal(result, [0, 3, -6, -7, 0])
    values[:] = 0
    np.testing.assert_array_equal(result, [0, 3, -6, -7, 0])
    assert np.all(aux.validate_parameters(aux.the_beta, 0, values) == -1)


def test_complements_preserve_inputs_and_only_check_presence_when_requested():
    x = np.array([0.25, 0.4])
    a, b = aux.check_complements(x, np.array([0.75, 0.7]))
    np.testing.assert_array_equal(a, x)
    np.testing.assert_array_equal(b, [0.75, 0.7])
    x[:] = 0
    assert a[0] == 0.25
    with pytest.raises(ValueError):
        a.setflags(write=True)
    a, b = aux.check_complements(y=[1e-300, 0.5])
    np.testing.assert_array_equal(b, [1e-300, 0.5])
    np.testing.assert_array_equal(a, [1, 0.5])
    assert aux.check_complements(x=float("nan"), set_values=False) is None
    with pytest.raises(ValueError):
        aux.check_complements(set_values=False)
    with pytest.raises(ValueError):
        aux.check_complements(x=float("nan"))


def test_three_epsilon_tolerance_and_broadcast_ranges():
    eps = np.finfo(float).eps
    np.testing.assert_array_equal(
        aux.add_to_one([1, 1 + 3 * eps, 1 + 4 * eps], 0), [True, True, False]
    )
    np.testing.assert_array_equal(
        aux.dbl_in_range([[0], [2]], [-1, 0, 1], [0, 1, 2]),
        [[True, True, False], [False, False, True]],
    )
    assert aux.int_in_range(2**100 + 1, 2**100 + 1, 2**100 + 1)
    assert not aux.in_range(2**100, 2**100 + 1, 2**100 + 2)
    assert not aux.add_to_one(1e308, 1e308)


@pytest.mark.parametrize("target,expected", [(0.25, 0), (-1, -50), (2, 50)])
def test_native_root_adapter_settings_and_status_translation(target, expected):
    # Slot 3 is x in the beta descriptor; inverse selector 3 would mean shape a.
    local = aux.cdf_set_zero_finder(aux.the_beta, 3)
    result = interval_zf(lambda x: x, target, local=local)
    assert aux.cdf_finalize_status(local) == expected
    if expected == 0:
        assert result.x == target
    else:
        assert (result.bound_low, result.bound_high) == (0, 1)


def test_unfinished_and_exhausted_root_states_are_not_boundary_failures():
    with pytest.raises(ValueError):
        aux.cdf_finalize_status(set_zero_finder())
    local = set_zero_finder(low_limit=0, hi_limit=2, max_evaluations=1)
    with pytest.raises(ArithmeticError):
        interval_zf(lambda x: x * x, 2, local=local)
    with pytest.raises(ArithmeticError):
        aux.cdf_finalize_status(local)


def test_descriptors_are_immutable_and_retain_source_limitations():
    assert aux.the_f.max_which == 2
    assert aux.the_non_central_f.max_which == 3
    assert aux.the_dummy_binomial.max_which == 0
    with pytest.raises(FrozenInstanceError):
        aux.the_f.max_which = 4
    with pytest.raises(TypeError):
        aux.DISTRIBUTIONS["new"] = aux.the_f


@pytest.mark.parametrize("which", [0, 7, True, 1.5])
def test_invalid_adapter_slots(which):
    with pytest.raises(ValueError):
        aux.cdf_set_zero_finder(aux.the_beta, which)


def test_malformed_inputs_and_missing_argument_error():
    with pytest.raises(ValueError):
        aux.validate_parameters(aux.the_beta, 1, [1, 2])
    with pytest.raises(ValueError):
        aux.validate_parameters(aux.the_beta, True, np.zeros(6))
    with pytest.raises(ValueError):
        aux.dbl_in_range(0, 1, -1)
    with pytest.raises(ValueError):
        aux.int_in_range(1.5, 0, 2)
    with pytest.raises(ValueError, match="cdf_beta: which=3 requires missing argument b"):
        aux.which_miss(3, "cdf_beta", "b")


def test_custom_descriptors_validate_schema_and_copy_scalar_bounds():
    low = np.array(0.0)
    p = aux.CDFParameter("x", 2, low, 1)
    low[...] = 2
    assert p.low_bound == 0 and isinstance(p.low_bound, float)
    with pytest.raises(ValueError):
        aux.CDFParameter("x", True, 0, 1)
    with pytest.raises(ValueError):
        aux.CDFParameter("x", 2, 2, 1)
    with pytest.raises(ValueError):
        aux.CDFParameter("x", 2, 0, np.inf)
    with pytest.raises(ValueError):
        aux.CDFDistribution("bad", 2, 6, (p,))
    with pytest.raises(ValueError):
        aux.CDFDistribution("bad", True, 6, (p,) * 6)
