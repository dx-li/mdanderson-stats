"""Independent R checks for concurrent selection and posterior ordering."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.plbarpo_control import plbarpo_control_counts, plbarpo_control_monitor


def test_control_selection_and_posterior_comparisons_against_base_r():
    fixture = Path(__file__).parent / "fixtures" / "plbarpo-control-posterior.csv"
    with fixture.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    enrollment = np.arange(6)
    outcomes = [1, 1, 0, 0, 1, 0]
    observed = [0.5, 1.5, 2.5, 3.5, 6, 5.5]
    for now in (5.75, 7):
        concurrent = plbarpo_control_counts(
            enrollment, outcomes, observed, [[0, 4], [3, np.inf]], as_of=now
        )
        entire = plbarpo_control_counts(enrollment, outcomes, observed, [[0, np.inf]], as_of=now)[0]
        for mode, counts in (("entire", entire), ("concurrent", concurrent)):
            expected = [r for r in rows if r["mode"] == mode and float(r["as_of"]) == now]
            expected_counts = [
                [int(r["control_successes"]), int(r["control_failures"])] for r in expected
            ]
            np.testing.assert_array_equal(np.broadcast_to(counts, (2, 2)), expected_counts)
            result = plbarpo_control_monitor(
                [3, 4],
                [1, 2],
                prior=[[1, 1]] * 2,
                control_prior=[1, 1],
                control_counts=counts,
                control_mode=mode,
                pfut=0.25,
                peff=0.8,
                pfinal=0.7,
            )
            for field, column in (
                ("futility_probability", "control_greater"),
                ("efficacy_probability", "treatment_greater"),
                ("final_efficacy_probability", "treatment_greater"),
            ):
                np.testing.assert_allclose(
                    getattr(result, field),
                    [float(r[column]) for r in expected],
                    rtol=1e-9,
                    atol=5e-9,
                )
            for field in ("futile", "efficacious", "final_efficacious"):
                np.testing.assert_array_equal(
                    getattr(result, field), [r[field] == "TRUE" for r in expected]
                )
            np.testing.assert_array_equal(
                result.control_alpha, np.asarray(expected_counts)[:, 0] + 1
            )
            np.testing.assert_array_equal(
                result.control_beta, np.asarray(expected_counts)[:, 1] + 1
            )
