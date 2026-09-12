"""Independent posterior references and an exhaustive small-design identity."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.bop2_dc import bop2_dc_design


def test_binary_decisions_against_base_r():
    fixture = Path(__file__).parent / "fixtures" / "bop2-dc-binary.csv"
    with fixture.open(newline="") as stream:
        for row in csv.DictReader(stream):
            design = bop2_dc_design(
                int(row["max_subjects"]),
                float(row["lrv"]),
                float(row["cmv"]),
                prior=(float(row["prior_alpha"]), float(row["prior_beta"])),
                looks=[10, 20, 30, 40],
                **{
                    key: float(row[key])
                    for key in ("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")
                },
            )
            n = int(row["patients"])
            state = design.monitor(int(row["responses"]), n)
            np.testing.assert_allclose(
                [state.posterior_lrv, state.posterior_cmv],
                [float(row["posterior_lrv"]), float(row["posterior_cmv"])],
                rtol=1e-12,
                atol=0,
            )
            expected = ("final_" if n == design.max_subjects else "stop_") + row["decision"]
            if row["decision"] == "continue":
                expected = "continue"
            assert state.decision.item() == expected


def test_exact_operating_characteristics_with_absorbing_interim_stop():
    design = bop2_dc_design(
        4,
        0.2,
        0.5,
        prior=(1, 1),
        looks=[2, 4],
        lambda_lrv=0.8,
        lambda_cmv=0.5,
        gamma_lrv=0.5,
        gamma_cmv=0.5,
    )
    p = np.array([0.0, 0.2, 0.5, 1.0])
    result = design.operating_characteristics(p)
    # Only 00 stops at n=2. At n=4, 3+ responses go, 2 consider, and 0/1 no-go.
    go = 4 * p**3 * (1 - p) + p**4
    consider = 5 * p**2 * (1 - p) ** 2  # 0011 was already stopped.
    np.testing.assert_allclose(result.final_go, go, atol=1e-15)
    np.testing.assert_allclose(result.final_consider, consider, atol=1e-15)
    np.testing.assert_allclose(
        result.stop_no_go.sum(axis=-1) + result.final_no_go, 1 - go - consider, atol=1e-15
    )
    np.testing.assert_allclose(result.expected_sample_size, 4 - 2 * (1 - p) ** 2)
