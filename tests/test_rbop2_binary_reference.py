"""Independent outcome-path and fractional-prior references for rBOP2."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.rbop2_binary import rbop2_binary_design


def _rows(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_exact_operating_characteristics_against_all_outcome_paths():
    reference = _rows("rbop2-binary-reference.csv")
    for endpoint in ("efficacy", "toxicity"):
        rows = [row for row in reference if row["endpoint"] == endpoint]
        design = rbop2_binary_design(
            [[1, 1], [2, 2]],
            prior=[[1, 1], [1, 1]],
            endpoint=endpoint,
            margin=0,
            lower_cutoffs=[0.25, 0.8],
            upper_cutoffs=[0.8, 0.8],
        )
        result = design.operating_characteristics(
            [float(row["experimental_rate"]) for row in rows],
            [float(row["control_rate"]) for row in rows],
        )
        for values, column in (
            (result.stop_futility[:, 0], "early_futility"),
            (result.stop_superiority[:, 0], "early_superiority"),
            (result.final_negative, "final_negative"),
            (result.final_positive, "final_positive"),
            (result.overall_positive, "positive_probability"),
            (result.expected_total_sample_size, "expected_total"),
        ):
            np.testing.assert_allclose(
                values, [float(row[column]) for row in rows], atol=1e-12, rtol=0
            )
        np.testing.assert_allclose(result.overall_positive + result.overall_negative, 1)
        np.testing.assert_allclose(
            result.expected_experimental_sample_size + result.expected_control_sample_size,
            result.expected_total_sample_size,
        )
        np.testing.assert_allclose(
            result.final_negative + result.final_positive, result.probability_no_early_stop
        )
        assert not result.stop_futility.flags.writeable


def test_fractional_priors_and_signed_margins_against_r_integration():
    for row in _rows("rbop2-binary-margins.csv"):
        design = rbop2_binary_design(
            [[5, 5]],
            prior=[[1.5, 2.5], [2.25, 0.75]],
            endpoint=row["endpoint"],
            margin=float(row["margin"]),
            lower_cutoffs=[0.5],
            upper_cutoffs=[0.5],
        )
        result = design.monitor(2, 1, sample_size=[5, 5])
        expected = float(row["probability"])
        np.testing.assert_allclose(result.probability, expected, atol=1e-9, rtol=0)
        assert result.superior == (expected >= 0.5)
        assert result.futile == (expected < 0.5)


def test_declared_unequal_arm_sizes_determine_early_enrollment():
    design = rbop2_binary_design(
        [[2, 1], [4, 2]],
        prior=[[1, 1], [1, 1]],
        endpoint="efficacy",
        margin=0,
        lower_cutoffs=[0.1, 0.8],
        upper_cutoffs=[0.8, 0.8],
    )
    result = design.operating_characteristics(1, 0)
    assert result.overall_positive == 1
    assert result.expected_experimental_sample_size == 2
    assert result.expected_control_sample_size == 1
    assert result.expected_total_sample_size == 3
