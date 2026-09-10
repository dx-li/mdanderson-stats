"""Report field provenance, unit labeling, missingness and output bounds."""

import csv
import io
from dataclasses import replace

import numpy as np
import pytest

from mdanderson_stats import (
    format_sppcr_report,
    format_sppcr_simulations,
    sppcr_bootstrap,
    sppcr_bootstrap_intervals,
    sppcr_data,
    sppcr_frequencies,
)


def experiment(seen=None):
    d = sppcr_data(
        [0.5, 1],
        [[4, 8, 2], [10, 15, 5]] if seen is None else seen,
        [20, 30],
        [100, 102, 104],
        (100, 102),
        unseen_alleles="retain",
    )
    b = sppcr_bootstrap(
        d.dna,
        d.seen,
        d.wells,
        progenitor=d.progenitor,
        rng=np.random.default_rng(92),
        replicates=20,
    )
    return d, b


def section(text, name):
    block = text.split("[" + name + "]\n", 1)[1].split("\n[", 1)[0]
    return list(csv.DictReader(io.StringIO(block), delimiter="\t"))


def test_report_fields_match_fitted_bootstrap_and_interval_results():
    d, b = experiment()
    text = format_sppcr_report(d, b, precision=17)
    observed = sppcr_frequencies(
        b.observed_fit.mu, b.observed_fit.variance, progenitor=d.progenitor
    )
    intervals = sppcr_bootstrap_intervals(b)
    data = section(text, "DATA")
    assert [float(r["DNA_genome_input"]) for r in data] == [0.5, 1]
    assert [float(r["DNA_model"]) for r in data] == [1, 2]
    assert [int(r["seen_104"]) for r in data] == [2, 5]
    probs = section(text, "SAMPLING PROBABILITIES")
    assert float(probs[0]["104"]) == b.samples.probability[0, 2]
    for i, row in enumerate(section(text, "MEANS")):
        assert int(row["allele"]) == d.allele_sizes[i]
        assert row["role"] == ("P" if i in d.progenitor else "M")
        assert float(row["estimate"]) == b.observed_fit.mu[i]
        assert float(row["bootstrap_mean"]) == b.summary.mu.mean[i]
        assert float(row["asymptotic_sd"]) == np.sqrt(b.observed_fit.variance[i])
        assert float(row["bootstrap_sd"]) == b.summary.mu.standard_deviation[i]
        assert float(row["solver_lower"]) <= float(row["estimate"]) <= float(row["solver_upper"])
    summary = {r["quantity"]: r for r in section(text, "SUMMARY")}
    assert float(summary["calibration"]["estimate"]) == observed.calibration.value
    assert float(summary["calibration"]["lower"]) == intervals.calibration.lower
    assert float(summary["inverse_calibration"]["upper"]) == intervals.inverse_calibration.upper
    assert float(summary["mutant"]["estimate"]) == observed.mutant.value
    for i, row in enumerate(section(text, "FREQUENCIES")):
        assert float(row["estimate"]) == observed.frequency.value[i]
        assert float(row["bootstrap_mean"]) == b.summary.frequency.mean[i]
        assert float(row["asymptotic_sd"]) == observed.frequency.standard_error[i]
        assert float(row["lower"]) == intervals.frequency.lower[i]
        assert float(row["upper"]) == intervals.frequency.upper[i]
    transformed = section(text, "TRANSFORMED FREQUENCIES: 2*asin(sqrt(p))")
    assert float(transformed[-1]["estimate"]) == observed.mutant.transformed.value
    assert (
        float(transformed[0]["bootstrap_sd"])
        == b.summary.transformed_frequency.standard_deviation[0]
    )


def test_simulations_retain_all_replicates_and_values():
    d, b = experiment()
    rows = list(
        csv.DictReader(io.StringIO(format_sppcr_simulations(d, b, precision=17)), delimiter="\t")
    )
    assert len(rows) == 20
    for i, row in enumerate(rows):
        assert int(row["replicate"]) == i + 1
        assert int(row["frequency_defined"]) == b.summary.defined[i]
        assert float(row["calibration"]) == b.summary.calibration.values[i]
        assert float(row["mu_104"]) == b.fit.mu[i, 2]
        assert float(row["frequency_104"]) == b.summary.frequency.values[i, 2]
        assert float(row["mutant"]) == b.summary.mutant.values[i]


def test_all_zero_results_show_missing_frequencies_not_fake_zero_ratios():
    d, b = experiment([[0, 0, 0], [0, 0, 0]])
    text = format_sppcr_report(d, b)
    assert "Undefined frequency replicates\t20\n" in text
    rows = section(text, "FREQUENCIES")
    assert all(r["estimate"] == "NA" and r["lower"] == "NA" and r["available"] == "0" for r in rows)
    summary = {r["quantity"]: r for r in section(text, "SUMMARY")}
    assert summary["calibration"]["estimate"] == "0"
    assert summary["inverse_calibration"]["estimate"] == "NA"
    simulations = list(csv.DictReader(io.StringIO(format_sppcr_simulations(d, b)), delimiter="\t"))
    assert len(simulations) == 20
    assert all(r["frequency_100"] == "NA" and r["frequency_defined"] == "0" for r in simulations)


def test_adjusted_and_zero_boundary_counts_are_visible():
    d, b = experiment([[20, 0, 2], [30, 0, 5]])
    rows = section(format_sppcr_report(d, b), "MEANS")
    assert rows[0]["adjusted"] == "1" and rows[0]["bootstrap_adjusted"] == "20"
    assert rows[1]["boundary"] == "1" and rows[1]["bootstrap_boundary"] == "20"
    assert rows[1]["asymptotic_sd"] == "NA"


def test_unbounded_inverse_and_clipped_bounds_are_explicit():
    d, b = experiment()
    text = format_sppcr_report(d, b, multiplier=1000)
    row = next(r for r in section(text, "SUMMARY") if r["quantity"] == "inverse_calibration")
    assert row["upper"] == "inf" and row["upper_clipped"] == "1"


@pytest.mark.parametrize("change", ["counts", "dna", "parents", "units"])
def test_data_fit_mismatch_rejected(change):
    d, b = experiment()
    if change == "counts":
        d = replace(d, seen=d.seen + 1)
    elif change == "dna":
        d = replace(d, dna=d.dna * 2, genome_dna=d.genome_dna * 2)
    elif change == "parents":
        d = replace(d, progenitor=(0, 0), progenitor_sizes=(100, 100))
    else:
        d = replace(d, genome_dna=d.genome_dna * 2)
    for formatter in [format_sppcr_report, format_sppcr_simulations]:
        with pytest.raises(ValueError):
            formatter(d, b)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(precision=0),
        dict(precision=18),
        dict(precision=True),
        dict(max_characters=0),
        dict(max_characters=20),
    ],
)
def test_output_limits_and_precision(kwargs):
    d, b = experiment()
    for formatter in [format_sppcr_report, format_sppcr_simulations]:
        with pytest.raises(ValueError):
            formatter(d, b, **kwargs)


def test_reports_are_deterministic_and_do_not_change_results():
    d, b = experiment()
    seen = b.samples.seen.copy()
    for formatter in [format_sppcr_report, format_sppcr_simulations]:
        assert formatter(d, b) == formatter(d, b)
    np.testing.assert_array_equal(b.samples.seen, seen)
