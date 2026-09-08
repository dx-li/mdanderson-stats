"""Detailed report rows as independently checked probability distributions."""

import math
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import CTAStudySpecification


def rows(report, heading):
    block = report.split("\n" + heading + "\n", 1)[1].split("\n\n", 1)[0]
    return np.array([[float(x) for x in line.split("\t")] for line in block.splitlines()[1:]])


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "table",
    [
        [[1, 9], [5, 15]],
        [[10, 10], [10, 10]],
        [[0, 10], [0, 20]],
        [[4, 6], [5, 95]],
    ],
)
def test_every_probability_row_and_cumulative_sum(table, legacy):
    study = CTAStudySpecification(
        chi_square=False, fisher="always", binomial=True, event_index=0, legacy=legacy
    ).run(table)
    report = study.report(details=True, digits=17)
    fisher = rows(report, "Fisher terms")
    assert len(fisher) == study.fisher.terms
    row_total, col_total, total = sum(table[0]), table[0][0] + table[1][0], sum(map(sum, table))
    expected = []
    for a, b, c, d, probability, cumulative in fisher:
        assert a + b == row_total and a + c == col_total and a + b + c + d == total
        mass = Fraction(
            math.comb(row_total, int(a)) * math.comb(total - row_total, col_total - int(a)),
            math.comb(total, col_total),
        )
        expected.append(float(mass))
        assert_allclose(probability, float(mass), rtol=2e-13)
    assert_allclose(fisher[:, 5], np.cumsum(expected), rtol=2e-13)
    assert_allclose(fisher[-1, 5], study.fisher.pvalue, rtol=2e-13)
    for i, label in enumerate(("lower", "upper")):
        terms = rows(report, f"Binomial {label} terms")
        group = int(terms[0, 0])
        n = int(study.binomial.events.sum())
        p = Fraction(int(study.binomial.group_sizes[group]), total)
        expected = [
            float(math.comb(n, int(k)) * p ** int(k) * (1 - p) ** (n - int(k))) for k in terms[:, 1]
        ]
        assert_allclose(terms[:, 2], expected, rtol=2e-13)
        assert_allclose(terms[:, 3], np.cumsum(expected), rtol=2e-13)
        assert_allclose(terms[-1, 3], study.binomial.reported_tails[i], rtol=2e-13)
        differences = np.diff(terms[:, 1])
        assert_array_equal(differences, np.full(len(differences), 1 if label == "lower" else -1))


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("table", [[[10, 10], [10, 10]], [[1, 9], [5, 15]], [[1, 2, 3], [4, 5, 6]]])
def test_cell_contributions_sum_to_reported_statistics(table, legacy):
    study = CTAStudySpecification(legacy=legacy, fisher="never").run(table)
    block = study.report(details=True, digits=17).split("\nCell details\n")[1]
    values = np.array(
        [
            [float(v) if v != "NA" else np.nan for v in line.split("\t")]
            for line in block.splitlines()[1:]
        ]
    )
    assert_allclose(values[:, 2], np.asarray(table).ravel())
    assert_allclose(values[:, 4].sum(), study.chi_square.statistic, rtol=1e-14)
    if study.chi_square.yates_statistic is None:
        assert np.isnan(values[:, 5]).all()
    else:
        assert_allclose(values[:, 5].sum(), study.chi_square.yates_statistic, rtol=1e-14)
    assert_allclose(values[:, 6].reshape(np.shape(table)).sum(axis=1), 100)
    assert_allclose(values[:, 7].reshape(np.shape(table)).sum(axis=0), 100)


def test_source_fisher_cutoff_order_and_triggering_term():
    study = CTAStudySpecification(legacy=True).run([[100, 100], [100, 100]])
    # Force Fisher because the original automatic trigger does not fire here.
    study = CTAStudySpecification(legacy=True, fisher="always").run(study.observed)
    terms = rows(study.report(details=True, digits=17), "Fisher terms")
    assert len(terms) == 25
    assert_array_equal(terms[0, :4], [100, 100, 100, 100])
    assert_array_equal(terms[:, 0], np.arange(100, 125))
    assert terms[-1, 4] < 1e-5 * terms[0, 4] <= terms[-2, 4]


def test_limit_is_explicit_exact_and_preserves_existing_file(tmp_path):
    study = CTAStudySpecification(binomial=True).run([[1, 9], [5, 15]])
    required = 4 + int(study.fisher.terms) + int(study.binomial.events.sum()) + 2
    path = tmp_path / "details.txt"
    path.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="requires"):
        study.write_report(path, details=True, max_terms=required - 1)
    assert path.read_text(encoding="utf-8") == "keep"
    study.write_report(path, details=True, max_terms=required, digits=17)
    assert "Fisher terms" in path.read_text(encoding="utf-8")
    assert "Fisher terms" not in study.report()
    large = CTAStudySpecification(chi_square=False, binomial=True, event_index=0).run(
        [[1000000, 1000000], [1000000, 1000000]]
    )
    with pytest.raises(ValueError, match="2000002 rows"):
        large.report(details=True)


@pytest.mark.parametrize(
    "kwargs", [{"details": 1}, {"max_terms": 0}, {"max_terms": True}, {"max_terms": 1.5}]
)
def test_invalid_detail_options(kwargs):
    with pytest.raises(ValueError):
        CTAStudySpecification().run([[1, 9], [5, 15]]).report(**kwargs)
