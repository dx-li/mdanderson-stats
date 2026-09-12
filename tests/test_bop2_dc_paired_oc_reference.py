"""Independent exhaustive paired trial probabilities with correlated endpoints."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.bop2_dc_paired import bop2_dc_paired_design


def test_paired_oc_matches_all_four_patient_paths():
    with (Path(__file__).parent / "fixtures" / "bop2-dc-paired-oc.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            multiple = row["mode"] == "multiple_efficacy"
            design = bop2_dc_paired_design(
                4,
                row["mode"],
                [0.2, 0.2] if multiple else [0.2, 0.5],
                [0.5, 0.5] if multiple else [0.5, 0.2],
                prior=[0.25] * 4,
                looks=[2, 4],
                lambda_lrv=[0.8, 0.8],
            )
            probability = [
                float(row[key]) for key in ("both", "first_only", "second_only", "neither")
            ]
            result = design.operating_characteristics(probability)
            for field in ("final_go", "final_consider", "final_no_go", "expected_sample_size"):
                np.testing.assert_allclose(
                    getattr(result, field), float(row[field]), rtol=1e-12, atol=0
                )
            stop = float(row["stop_no_go"])
            np.testing.assert_allclose(result.stop_no_go, [stop, 0], rtol=1e-12, atol=0)
            np.testing.assert_allclose(
                result.sample_size_probability, [stop, 1 - stop], rtol=1e-12, atol=0
            )
            np.testing.assert_allclose(
                result.no_go_probability, stop + float(row["final_no_go"]), rtol=1e-12, atol=0
            )
