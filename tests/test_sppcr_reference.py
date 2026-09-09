"""Independent checks of valid native SPPCR fits and evidence of boundary defects."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

FIXTURE = json.loads((Path(__file__).parent / "fixtures/sppcr.json").read_text())


def rows(name, tag):
    result = []
    for line in FIXTURE["cases"][name]["stdout"].splitlines():
        words = line.split()
        if words and words[0] == tag:
            offset = 2 if tag in {"mu", "frequency", "adjusted_seen"} else 1
            result.append([float(x) for x in words[offset:]])
    return np.array(result)


@pytest.mark.parametrize(
    "name", ["single_level", "scaled_dna", "heterozygous", "two_levels_exact_model"]
)
def test_native_interior_fit_matches_closed_form_and_observed_information(name):
    record = FIXTURE["cases"][name]
    dna, wells, seen = (np.array(record[key]) for key in ("dna", "wells", "seen"))
    # These cases have a single DNA level, or exact common Poisson means across levels.
    expected_mu = -np.log1p(-seen[0] / wells[0]) / dna[0]
    q = np.exp(-dna[:, None] * expected_mu)
    p = -np.expm1(-dna[:, None] * expected_mu)
    expected_variance = 1 / np.sum(seen * dna[:, None] ** 2 * q / p**2, axis=0)
    actual = rows(name, "mu")
    assert record["returncode"] == 0
    np.testing.assert_allclose(actual[:, 1], expected_mu, rtol=3e-14)
    np.testing.assert_allclose(actual[:, 2], expected_variance, rtol=3e-14)
    np.testing.assert_allclose(actual[:, 3], np.sqrt(expected_variance), rtol=3e-14)


@pytest.mark.parametrize(
    "name", ["single_level", "scaled_dna", "heterozygous", "two_levels_exact_model"]
)
def test_frequency_and_mutant_variance_match_independent_delta_method(name):
    record = FIXTURE["cases"][name]
    dna, wells, seen = (np.array(record[key]) for key in ("dna", "wells", "seen"))
    means = -np.log1p(-seen[0] / wells[0]) / dna[0]
    q = np.exp(-dna[:, None] * means)
    variances = 1 / np.sum(seen * dna[:, None] ** 2 * q / (1 - q) ** 2, axis=0)
    total = means.sum()
    gradient = (np.eye(len(means)) * total - means[:, None]) / total**2
    covariance = (gradient * variances) @ gradient.T
    actual = rows(name, "frequency")
    np.testing.assert_allclose(actual[:, 0], means / total, rtol=3e-14)
    np.testing.assert_allclose(actual[:, 1], np.diag(covariance), rtol=3e-14)
    progenitor = set(i - 1 for i in record["progenitor"])
    mask = np.array([i not in progenitor for i in range(len(means))])
    mutant_mean = means[mask].sum() / total
    mutant_gradient = (mask - mutant_mean) / total
    mutant_variance = np.sum(mutant_gradient**2 * variances)
    np.testing.assert_allclose(
        rows(name, "mutant")[0],
        [mutant_mean, mutant_variance, math.sqrt(mutant_variance)],
        rtol=4e-14,
    )
    np.testing.assert_allclose(
        rows(name, "calibration")[0],
        [total, variances.sum(), math.sqrt(variances.sum())],
        rtol=3e-14,
    )


def test_dna_unit_change_preserves_frequency_and_scales_mean_variance():
    np.testing.assert_allclose(
        rows("single_level", "frequency"), rows("scaled_dna", "frequency"), rtol=3e-14
    )
    np.testing.assert_allclose(
        rows("scaled_dna", "mu")[:, 1], 4 * rows("single_level", "mu")[:, 1], rtol=3e-14
    )
    np.testing.assert_allclose(
        rows("scaled_dna", "mu")[:, 2], 16 * rows("single_level", "mu")[:, 2], rtol=3e-14
    )


def test_native_correction_triggers_for_only_one_saturated_level():
    # Positive nondetections at level 1 already ensure a finite maximum.
    actual = rows("partly_saturated", "adjusted_seen")
    np.testing.assert_array_equal(actual[0], [9.5, 20])
    assert FIXTURE["cases"]["partly_saturated"]["seen"][0][0] == 10


def test_native_correction_can_create_negative_counts_without_reporting_failure():
    assert FIXTURE["cases"]["negative_adjustment"]["returncode"] == 0
    assert rows("negative_adjustment", "adjusted_seen")[0, 0] == -0.5
    # This is defect evidence, not a valid fit target for the Python implementation.


def test_never_seen_native_allele_is_perturbed_away_from_boundary_mle():
    assert rows("never_seen", "mu")[0, 1] > 0
    assert rows("never_seen", "mu")[0, 2] == 0
    # For zero detections log L(mu)=-mu*sum(n*d), strictly decreasing for mu>=0.
    record = FIXTURE["cases"]["never_seen"]
    score = -sum(n * d for n, d in zip(record["wells"], record["dna"], strict=True))
    assert score < 0
