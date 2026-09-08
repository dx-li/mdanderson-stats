"""End-to-end comparison with the original CTA program's retained-choice loop."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import CTAStudySpecification

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cta_workflow.json").read_text())


def test_original_program_two_table_session_and_all_reported_analyses(tmp_path):
    settings = CTAStudySpecification(
        legacy=True, kappa=True, mcnemar=True, diagnostic=True, odds=True, binomial=True
    )
    for i, case in enumerate(FIXTURE["cases"]):
        study = settings.run(case["observed"])
        chi = study.chi_square
        assert_allclose(
            [
                [chi.statistic, chi.pvalue],
                [chi.yates_statistic, chi.yates_pvalue],
                [chi.cochran_statistic, chi.cochran_pvalue],
            ],
            case["chi_square"],
            atol=5.1e-6,
            rtol=0,
        )
        assert_allclose(study.fisher.pvalue, case["fisher"], rtol=1e-5)
        assert_allclose(study.kappa.kappa, case["kappa"], atol=5.1e-5, rtol=0)
        assert_allclose(
            [study.kappa.variance, study.kappa.null_variance], case["variances"], atol=1e-7, rtol=0
        )
        assert_allclose(study.mcnemar.summed_statistic, case["mcnemar"][0], atol=0.0051, rtol=0)
        assert_allclose(study.mcnemar.summed_pvalue, case["mcnemar"][1], atol=5.1e-5, rtol=0)
        diagnostic = study.diagnostic
        probabilities = [
            diagnostic.sensitivity,
            diagnostic.specificity,
            diagnostic.positive_predictive_value,
            diagnostic.negative_predictive_value,
        ]
        assert_allclose(
            np.column_stack([probabilities, diagnostic.standard_errors]),
            case["diagnostic"],
            atol=5.2e-6,
            rtol=0,
        )
        assert_allclose(
            [study.odds.odds_ratio, study.odds.lower, study.odds.upper],
            case["odds"],
            atol=5.2e-6,
            rtol=0,
        )
        assert_allclose(study.binomial.reported_tails, case["binomial_tails"], atol=5.1e-7, rtol=0)
        assert_allclose(study.binomial.pvalue, case["binomial_two_sided"], atol=5.1e-6, rtol=0)
        destination = tmp_path / f"study-{i}.txt"
        study.write_report(destination, details=True, digits=17)
        detail = (
            destination.read_text(encoding="utf-8").split("\nCell details\n")[1].split("\n\n")[0]
        )
        cells = np.array([[float(x) for x in line.split()] for line in detail.splitlines()[1:]])
        cells[:, :2] += 1  # Original report indices are one-based.
        # Native contributions are printed to two decimal places.
        assert_allclose(cells, case["cells"], atol=0.00501, rtol=0)


def test_original_ten_category_boundary_without_workspace_mutation():
    observed = np.ones((10, 10))
    study = CTAStudySpecification(kappa=True, mcnemar=True).run(observed)
    assert study.chi_square.statistic == 0
    assert study.chi_square.degrees_of_freedom == 81
    assert_allclose(study.kappa.kappa, 0, atol=2e-15)
    assert_allclose(study.kappa.variance, 1 / 900, rtol=2e-14)
    assert_allclose(study.kappa.null_variance, 1 / 900, rtol=2e-14)
    assert study.mcnemar.summed_df == 45
    assert study.mcnemar.heterogeneity_df == 44
    assert study.mcnemar.summed_statistic == 0
    assert_allclose(observed, np.ones((10, 10)), rtol=0, atol=0)
    assert "Cell details" in study.report(details=True)
