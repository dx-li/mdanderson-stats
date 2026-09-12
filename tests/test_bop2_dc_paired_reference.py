"""Independent Dirichlet marginal posteriors and paired decision composition."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.bop2_dc_paired import bop2_dc_paired_design


def test_paired_decisions_against_base_r():
    with (Path(__file__).parent / "fixtures" / "bop2-dc-paired.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            design = bop2_dc_paired_design(
                int(row["max_subjects"]),
                "multiple_efficacy" if row["mode"] == "multiple" else "efficacy_toxicity",
                [float(row["lrv_first"]), float(row["lrv_second"])],
                [float(row["cmv_first"]), float(row["cmv_second"])],
                prior=[float(row["cell_prior"])] * 4,
                looks=[10, 20],
                **{
                    key: [float(row[key])] * 2
                    for key in ("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")
                },
            )
            counts = [int(row[key]) for key in ("both", "first_only", "second_only", "neither")]
            state = design.monitor(counts)
            expected = [
                [float(row[f"posterior_{criterion}_{endpoint}"]) for criterion in ("lrv", "cmv")]
                for endpoint in ("first", "second")
            ]
            np.testing.assert_allclose(state.marginal_posterior, expected, rtol=1e-12, atol=0)
            np.testing.assert_allclose(state.posterior_shapes, np.asarray(counts) + 0.25)
            final = int(row["patients"]) == design.max_subjects
            decision = ("final_" if final else "stop_") + row["decision"]
            if row["decision"] == "continue":
                decision = "continue"
            assert state.decision.item() == decision
