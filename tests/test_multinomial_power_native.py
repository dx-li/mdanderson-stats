"""Compare against archived Fortran with the fixture's explicit repairs."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import multinomial_power

FIXTURE = json.loads((Path(__file__).parent / "fixtures/multinompow_native.json").read_text())
CASES = [case for case in FIXTURE["cases"] if case["profile"] == "corrected_double"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"n{c['n']}-p{c['null']}")
def test_native_corrected_double(case):
    result = multinomial_power(case["n"], case["null"], case["alternatives"], case["alpha"])
    sizes = np.array([row["size"] for row in case["rows"]]).T
    critical = np.array(
        [[np.inf if x is None else x for x in row["critical"]] for row in case["rows"]]
    ).T
    powers = np.array([row["power"] for row in case["rows"]]).transpose(2, 1, 0)
    np.testing.assert_allclose(result.actual_size, sizes, rtol=2e-11, atol=2e-13)
    np.testing.assert_allclose(result.critical_values, critical, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(result.power, powers, rtol=2e-11, atol=2e-13)
    assert case["null_mass"] == pytest.approx(1, abs=2e-11)
