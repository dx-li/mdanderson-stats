import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.mtadf_author_global import (
    MTADFAuthorGlobalFit,
    mtadf_author_global_decision,
    mtadf_author_global_fit,
)

FIXTURE = Path(__file__).parent / "fixtures/mtadf-author-global-reference.csv"


def _reference_cases() -> dict[str, list[dict[str, str]]]:
    cases: dict[str, list[dict[str, str]]] = defaultdict(list)
    with FIXTURE.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            cases[row["case"]].append(row)
    return cases


def test_grouped_author_irls_matches_private_arm_reference_cases():
    cases = _reference_cases()
    assert set(cases) == {
        "one_dose_zero_events",
        "unequal_prefix_counts",
        "all_zero_prefix",
        "all_response_prefix",
    }
    for name, rows in cases.items():
        n = np.array([int(value) for value in rows[0]["n_vector"].split(";")])
        y = np.array([int(value) for value in rows[0]["response_vector"].split(";")])
        fit = mtadf_author_global_fit(n, y)
        coefficients = np.array([float(rows[0][f"beta{index}"]) for index in range(3)])
        efficacy = np.array([float(row["fitted_efficacy"]) for row in rows])
        prior_scales = np.array([float(rows[0][f"prior_scale{index}"]) for index in range(3)])
        final_prior_sd = np.array([float(rows[0][f"final_prior_sd{index}"]) for index in range(3)])
        assert fit.coefficients == pytest.approx(coefficients, rel=0, abs=2e-12)
        assert fit.fitted_efficacy == pytest.approx(efficacy, rel=0, abs=2e-12)
        assert fit.prior_scales == pytest.approx(prior_scales, rel=0, abs=2e-12)
        assert fit.final_prior_sd == pytest.approx(final_prior_sd, rel=0, abs=2e-12)
        assert fit.deviance == pytest.approx(float(rows[0]["deviance"]), rel=0, abs=2e-12)
        assert fit.iterations == int(rows[0]["iterations"])
        assert fit.converged is (rows[0]["converged"] == "TRUE")
        assert int(
            np.flatnonzero(fit.fitted_efficacy == np.max(fit.fitted_efficacy))[-1]
        ) + 1 == int(rows[0]["rightmost_peak_dose"])
        if name == "one_dose_zero_events":
            assert fit.coefficients[1:] == pytest.approx([0.0, 0.0], rel=0, abs=0)
            assert np.all(fit.fitted_efficacy == fit.fitted_efficacy[0])


def test_global_decision_moves_one_step_and_fresh_safety_cap_wins():
    n = [3, 6, 9, 0, 0]
    responses = [0, 3, 8, 0, 0]
    tox_safe = mtadf_author_global_decision(n, [0, 0, 0, 0, 0], responses, current_dose=1)
    assert tox_safe.target_dose == 3
    assert tox_safe.dose == 2
    assert tox_safe.action == "escalate"

    unsafe = mtadf_author_global_decision(n, [3, 6, 9, 0, 0], responses, current_dose=1)
    assert unsafe.admissible_count == 1
    assert unsafe.dose == 0
    assert unsafe.action == "deescalate"
    assert "capped" in unsafe.reason

    final = mtadf_author_global_decision(n, [0, 0, 0, 0, 0], responses, final=True)
    assert final.dose == 3
    assert final.action == "recommend"


def test_global_decision_rejects_nonconverged_or_mismatched_fit():
    fit = mtadf_author_global_fit([6, 0, 0], [0, 0, 0])
    nonconverged = MTADFAuthorGlobalFit(
        subjects=fit.subjects,
        responses=fit.responses,
        standardized_doses=fit.standardized_doses,
        coefficients=fit.coefficients,
        fitted_efficacy=fit.fitted_efficacy,
        prior_scales=fit.prior_scales,
        final_prior_sd=fit.final_prior_sd,
        deviance=fit.deviance,
        iterations=fit.iterations,
        converged=False,
    )
    with pytest.raises(ArithmeticError, match="did not converge"):
        mtadf_author_global_decision(
            [6, 0, 0], [0, 0, 0], [0, 0, 0], current_dose=0, fit=nonconverged
        )
    with pytest.raises(ValueError, match="does not match"):
        mtadf_author_global_decision([6, 0, 0], [0, 0, 0], [1, 0, 0], current_dose=0, fit=fit)
