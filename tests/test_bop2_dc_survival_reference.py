"""Independent base-R checks for the exponential/inverse-gamma posterior."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.bop2_dc_survival import bop2_dc_survival_design


def test_survival_posterior_and_decisions_against_base_r():
    with (Path(__file__).parent / "fixtures" / "bop2-dc-survival.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            design = bop2_dc_survival_design(
                int(row["max_subjects"]),
                float(row["lrv"]),
                float(row["cmv"]),
                looks=[10, 20, 30, 40],
                **{
                    key: float(row[key])
                    for key in (
                        "lambda_lrv",
                        "lambda_cmv",
                        "gamma_lrv",
                        "gamma_cmv",
                        "prior_shape",
                        "prior_scale",
                    )
                },
            )
            state = design.monitor(int(row["n"]), int(row["events"]), float(row["total_time"]))
            np.testing.assert_allclose(
                [state.posterior_lrv, state.posterior_cmv],
                [float(row["posterior_lrv"]), float(row["posterior_cmv"])],
                rtol=1e-12,
                atol=0,
            )
            assert state.posterior_shape == float(row["posterior_shape"])
            assert state.decision.item() == row["decision"]
