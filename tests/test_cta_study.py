"""Combined CTA workflow, reusable settings and real report-file integration."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import CTAStudySpecification


def test_all_analyses_and_reusable_independent_snapshots(tmp_path):
    specification = CTAStudySpecification(
        kappa=True,
        mcnemar=True,
        diagnostic=True,
        odds=True,
        binomial=True,
    )
    observed = np.array([[1.0, 9], [5, 15]])
    first = specification.run(observed)
    second = specification.run([[2, 8], [4, 16]])
    observed[:] = 0
    assert_array_equal(first.observed, [[1, 9], [5, 15]])
    assert_allclose(first.chi_square.expected, [[2, 8], [4, 16]])
    assert_allclose(first.chi_square.statistic, 0.9375)
    assert second.chi_square.statistic == 0
    assert first.fisher is not None and first.kappa is not None
    assert first.mcnemar is not None and first.diagnostic is not None
    assert_allclose(first.odds.odds_ratio, 1 / 3)
    assert_array_equal(first.binomial.events, [1, 5])
    assert_allclose(first.diagnostic.sensitivity, 1 / 6)
    assert not first.observed.flags.writeable
    assert first.specification is specification and second.specification is specification
    report = first.report(digits=12)
    assert "Row totals: 10\t20" in report
    assert "Column totals: 6\t24" in report and "Total: 30" in report
    assert "statistic: 0.9375" in report
    assert "heterogeneity_pvalue: NA" in report
    assert "auto: minimum expected count below" in report
    for section in ("chi_square", "fisher", "kappa", "mcnemar", "diagnostic", "odds", "binomial"):
        assert f"\n{section}\n" in report
    destination = tmp_path / "CTA results.txt"
    destination.write_text("previous report\n", encoding="utf-8")
    first.write_report(destination, digits=12)
    assert destination.read_text(encoding="utf-8") == report
    with pytest.raises(FrozenInstanceError):
        specification.kappa = False


def test_legacy_options_propagate_and_can_be_revised():
    source_spec = CTAStudySpecification(
        legacy=True,
        kappa=True,
        diagnostic=True,
        odds=True,
        binomial=True,
    )
    source = source_spec.run([[4, 6], [5, 95]])
    corrected = replace(source_spec, legacy=False).run([[4, 6], [5, 95]])
    for name in ("chi_square", "fisher", "kappa", "diagnostic", "odds", "binomial"):
        assert getattr(source, name).legacy
        assert not getattr(corrected, name).legacy
    assert source.fisher.alternative == "source"
    assert corrected.fisher.alternative == "two-sided"
    assert source.binomial.pvalue > 1.9 and corrected.binomial.pvalue < 0.02
    assert source.odds.lower > corrected.odds.lower
    assert_allclose(
        source.diagnostic.standard_errors,
        corrected.diagnostic.standard_errors * corrected.diagnostic.denominators,
    )


def test_auto_fisher_threshold_is_strict_and_explicit_request_overrides():
    equal = CTAStudySpecification().run([[10, 10], [10, 10]])
    assert equal.fisher is None and "not below" in equal.fisher_note
    forced = CTAStudySpecification(fisher="always").run([[10, 10], [10, 10]])
    assert_allclose(forced.fisher.pvalue, 1)
    assert forced.fisher_note == "explicitly requested"
    disabled = CTAStudySpecification(fisher="never").run([[1, 9], [5, 15]])
    assert disabled.fisher is None and disabled.fisher_note == "disabled"
    changed = CTAStudySpecification(fisher_threshold=11).run([[10, 10], [10, 10]])
    assert changed.fisher is not None


@pytest.mark.parametrize(
    "table,kwargs,note",
    [
        ([[1, 2, 3], [4, 5, 6]], {}, "2x2"),
        ([[1, 9], [5, 15]], {"chi_square": False}, "requires chi-square"),
        ([[0.5, 9.5], [5, 15]], {}, "integer counts"),
        ([[1, 1], [25000, 25000]], {}, "exceeds 50,000"),
    ],
)
def test_auto_skips_are_explicit_in_result_and_report(table, kwargs, note):
    study = CTAStudySpecification(**kwargs).run(table)
    assert study.fisher is None
    assert note in study.fisher_note and note in study.report()


def test_rectangular_report_preserves_fractional_counts_and_omissions():
    study = CTAStudySpecification().run([[0.5, 1.5, 2], [3, 4, 5]])
    report = study.report()
    assert "0.5\t1.5\t2" in report
    assert "yates_statistic: Not applicable" in report
    assert "\nkappa\nNot calculated" in report
    assert "expected:" in report and "contributions:" in report


def test_analysis_without_chi_square_can_handle_degenerate_diagnostic_table():
    study = CTAStudySpecification(chi_square=False, diagnostic=True).run([[0, 0], [0, 0]])
    assert np.isnan(study.diagnostic.sensitivity)
    assert "sensitivity: NA" in study.report()
    assert study.chi_square is None


@pytest.mark.parametrize(
    "kwargs,table",
    [
        ({"kappa": True}, [[1, 2, 3], [4, 5, 6]]),
        ({"odds": True}, [[0, 2], [3, 4]]),
        ({"binomial": True}, [[1, 9], [9, 1]]),
        ({"fisher": "always"}, [[0.5, 1.5], [3, 4]]),
    ],
)
def test_explicitly_requested_invalid_analysis_fails(kwargs, table):
    with pytest.raises(ValueError):
        CTAStudySpecification(**kwargs).run(table)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chi_square": 1},
        {"legacy": "yes"},
        {"fisher": "bad"},
        {"fisher_alternative": "bad"},
        {"legacy": True, "fisher_alternative": "less"},
        {"standard": "bad"},
        {"standard": np.array("columns")},
        {"fisher": np.array("auto")},
        {"groups": "bad"},
        {"risk_factor": "bad"},
        {"positive_index": True},
        {"response_index": 2},
        {"event_index": 0.5},
        {"alpha": 0},
        {"alpha": 1},
        {"alpha": [0.05]},
        {"expected_threshold": -1},
        {"fisher_threshold": np.nan},
    ],
)
def test_invalid_specification(kwargs):
    with pytest.raises(ValueError):
        CTAStudySpecification(**kwargs)


@pytest.mark.parametrize(
    "table",
    [
        [1, 2],
        [[[1, 2], [3, 4]]],
        [[1]],
        [[1, -2], [3, 4]],
        [[1, np.nan], [3, 4]],
        [[1e308, 1e308], [1, 1]],
    ],
)
def test_invalid_study_table(table):
    with pytest.raises(ValueError):
        CTAStudySpecification(chi_square=False).run(table)


@pytest.mark.parametrize("digits", [0, 18, True, 1.5])
def test_invalid_report_precision_does_not_overwrite_file(tmp_path, digits):
    study = CTAStudySpecification().run([[1, 9], [5, 15]])
    path = tmp_path / "report.txt"
    path.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError):
        study.write_report(path, digits=digits)
    assert path.read_text(encoding="utf-8") == "keep"
