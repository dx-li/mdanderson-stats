"""Compare matched-pair calculations with native Fortran and independent R."""

import csv
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.stplan_matched_pairs import (
    stplan_matched_pairs_initial_size,
    stplan_matched_pairs_power,
)
from mdanderson_stats.stplan_planning import stplan_solve


def _references(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as stream:
        return {row["case_id"]: row for row in csv.DictReader(stream)}


def test_matched_pair_power_and_inverses_against_native_and_r():
    native = _references("stplan-matched-pairs-native.csv")
    independent = _references("stplan-matched-pairs-reference.csv")
    assert native.keys() == independent.keys()
    for name, row in native.items():
        if name == "low_target_no_pilot_lost_sign":
            continue
        assert row["ok"] == "T" and row["status"] == "0", name
        method = int(row["iwhich"])
        fixed = {
            "difference": float(row["delta"]),
            "sample_size": float(row["n"]),
            "alpha": float(row["alpha"]),
            "sides": int(row["sides"]),
            **{key: float(row[key]) for key in ("z11", "z10", "z01", "z00")},
        }
        if method == 4:
            value = achieved = float(stplan_matched_pairs_power(**fixed))
        elif method in (1, 2, 3):
            compute, bounds = {
                1: ("difference", (1e-8, 0.99999999)),
                2: ("sample_size", (1, 10000)),
                3: ("alpha", (1e-8, 0.49)),
            }[method]
            del fixed[compute]
            solution = stplan_solve(
                "stplan_matched_pairs_power",
                compute=compute,
                target_power=float(row["target"]),
                bounds=bounds,
                parameters=fixed,
            )
            value, achieved = solution.value, solution.achieved_power
        else:
            assert method in (5, 6), name
            options = {"theta1": fixed["z10"], "theta2": fixed["z01"]} if method == 5 else {}
            initial = stplan_matched_pairs_initial_size(
                fixed["difference"],
                float(row["target"]),
                alpha=fixed["alpha"],
                sides=fixed["sides"],
                **options,
            )
            value, achieved = initial.sample_size, initial.achieved_power
            assert_allclose(
                initial.recommended_preliminary_size,
                float(independent[name]["recommendation"]),
                rtol=2e-11,
                atol=1e-12,
                err_msg=name,
            )
        assert_allclose(value, float(independent[name]["result"]), rtol=2e-10, atol=2e-11)
        assert_allclose(achieved, float(independent[name]["achieved_power"]), atol=1e-10, rtol=0)
        # The original delta solver has a 1e-6 relative convergence setting.
        assert_allclose(value, float(row["result"]), rtol=2e-6, atol=2e-8, err_msg=name)
        assert_allclose(achieved, float(row["achieved"]), atol=1.1e-7, rtol=0, err_msg=name)


def test_native_squared_inverse_can_return_valid_sized_but_wrong_power_design():
    row = _references("stplan-matched-pairs-native.csv")["low_target_no_pilot_lost_sign"]
    independent = _references("stplan-matched-pairs-reference.csv")[row["case_id"]]
    assert row["ok"] == "T" and row["status"] == "0"
    assert float(row["result"]) >= 1
    assert float(row["achieved"]) > 5 * float(row["target"])
    assert_allclose(float(row["achieved"]), float(independent["achieved_power"]), atol=1e-12)
    with pytest.raises(ValueError, match="positive sample-size solution"):
        stplan_matched_pairs_initial_size(
            float(row["delta"]),
            float(row["target"]),
            theta1=float(row["z10"]),
            theta2=float(row["z01"]),
            alpha=float(row["alpha"]),
            sides=int(row["sides"]),
        )


def test_extreme_pilot_scale_and_near_boundary_initial_size_remain_resolved():
    assert_allclose(
        stplan_matched_pairs_power(0.1, 40, 1e308, 1e308, 1e308, 1e308),
        stplan_matched_pairs_power(0.1, 40, 1, 1, 1, 1),
        rtol=2e-14,
    )
    # Here psi=(1+sqrt(2))*1e-300. Computing n/psi or the discriminant
    # before its square root would overflow/underflow despite finite power.
    factor = 1 + sqrt(2)
    q = 1 - 3 / (4 * factor**2)
    z = (-1.6448536269514722 + 1 / sqrt(factor)) / sqrt(q)
    expected = 0.5 * erfc(-z / sqrt(2))
    result = stplan_matched_pairs_power(1e-300, 1e300, 1e300, 1, 1, 0)
    assert_allclose(result, expected, atol=3e-15, rtol=0)

    # With estimated marginals, delta can exceed psi while the variance
    # remains positive. Exercise that source branch near its limiting effect.
    psi = 0.5
    limit = 2 * psi / sqrt(3 + psi)
    initial = stplan_matched_pairs_initial_size(limit * (1 - 1e-7), 0.8, theta1=0.5, theta2=0.5)
    assert initial.sample_size >= 1
    assert_allclose(initial.achieved_power, 0.8, atol=1e-10, rtol=0)

    empty = stplan_matched_pairs_power(np.empty(0), 100, 28, 17, 9, 26)
    assert empty.shape == (0,)
    with pytest.raises(ValueError, match="broadcast result"):
        stplan_matched_pairs_power(np.zeros((500, 1)), np.ones((1, 500)), 28, 17, 9, 26)
