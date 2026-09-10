"""SPPCR application composition with explicit random state and retained identities."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import (
    RandlibGenerator,
    SPPCRSimulationRequest,
    format_sppcr_analysis,
    format_sppcr_report,
    sppcr_analyze,
    sppcr_bootstrap,
    sppcr_data,
    sppcr_fit_means,
    sppcr_generate,
    sppcr_generate_legacy,
    sppcr_simulate,
    sppcr_truth,
)


def request(from_truth=True, simulations=True):
    return SPPCRSimulationRequest(
        sppcr_truth([0.25, 1], [50, 100], [2, 5, 3], 1.7, progenitor=(0, 1)),
        from_truth,
        simulations,
    )


def data():
    return sppcr_data(
        [0.25, 0.5], [[4, 10, 2, 0], [10, 20, 5, 0]], [50, 100], [100, 102, 104, 106], (100, 102)
    )


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("from_truth", [False, True])
def test_simulation_equals_explicit_components_and_consumes_same_rng(legacy, from_truth):
    factory = RandlibGenerator if legacy else lambda: np.random.default_rng(42)
    sample = sppcr_generate_legacy if legacy else sppcr_generate
    a, b = factory(), factory()
    req = request(from_truth)
    t = req.truth
    first = sample(t.probability, t.wells, rng=a).seen[0]
    expected = sppcr_bootstrap(
        t.dna,
        first,
        t.wells,
        progenitor=t.progenitor,
        rng=a,
        probability=t.probability if from_truth else None,
        replicates=12,
    )
    actual = sppcr_simulate(req, rng=b, replicates=12)
    assert_array_equal(actual.data.seen, first)
    assert_array_equal(actual.data.dna, t.dna)
    assert_array_equal(actual.data.genome_dna * 2, t.dna)
    assert_array_equal(actual.bootstrap.samples.seen, expected.samples.seen)
    assert_array_equal(actual.bootstrap.fit.mu, expected.fit.mu)
    assert_array_equal(actual.bootstrap.summary.frequency.values, expected.summary.frequency.values)
    assert_array_equal(a.uniform(size=5), b.uniform(size=5))
    report = format_sppcr_analysis(actual)
    assert report.report.startswith("SPPCR TRUTH PARAMETERS")
    assert "SPPCR analysis" in report.report
    assert report.simulations is not None and len(report.simulations.splitlines()) == 13


@pytest.mark.parametrize("legacy", [False, True])
def test_bootstrap_legacy_and_modern_fit_actual_draws(legacy):
    rng = RandlibGenerator() if legacy else np.random.default_rng(1)
    result = sppcr_bootstrap([1], [[20, 50, 80]], [100], progenitor=(0, 1), rng=rng, replicates=8)
    sampler = sppcr_generate_legacy if legacy else sppcr_generate
    rng = RandlibGenerator() if legacy else np.random.default_rng(1)
    samples = sampler([[0.2, 0.5, 0.8]], [100], rng=rng, replicates=8)
    assert_array_equal(result.samples.seen, samples.seen)
    scalar = [sppcr_fit_means([1], s, [100], saturation="half").mu for s in samples.seen]
    assert_array_equal(result.fit.mu, scalar)


def test_observed_analysis_preserves_dropped_labels_and_reports():
    d = data()
    analysis = sppcr_analyze(d, rng=np.random.default_rng(1), replicates=10)
    assert analysis.data is d and d.omitted_allele_sizes == (106,)
    assert analysis.request is None
    output = format_sppcr_analysis(analysis)
    assert output.report == format_sppcr_report(d, analysis.bootstrap)
    assert "106" in output.report and output.simulations is None
    assert format_sppcr_analysis(analysis, write_simulations=True).simulations is not None


def test_all_zero_observed_data_retained_with_explicit_undefined_results():
    d = sppcr_data([0.5], [[0, 0]], 20, [100, 102], (100, 102), unseen_alleles="retain")
    result = sppcr_analyze(d, rng=RandlibGenerator(), replicates=3)
    assert not np.any(result.bootstrap.summary.defined)
    text = format_sppcr_analysis(result, write_simulations=True)
    assert "NA" in text.report and text.simulations is not None


@pytest.mark.parametrize(
    "options", [{"replicates": 0}, {"replicates": True}, {"saturation": "bad"}]
)
def test_invalid_simulation_options_do_not_consume_rng(options):
    rng = RandlibGenerator()
    before = rng.get_seeds()
    with pytest.raises(ValueError):
        sppcr_simulate(request(), rng=rng, **options)
    assert rng.get_seeds() == before


def test_invalid_data_does_not_consume_rng():
    rng = RandlibGenerator()
    before = rng.get_seeds()
    with pytest.raises(ValueError, match="inconsistent"):
        sppcr_analyze(replace(data(), dna=np.array([99.0, 100.0])), rng=rng)
    assert rng.get_seeds() == before


@pytest.mark.parametrize("bad", ["probability", "choices", "dna"])
def test_invalid_truth_before_draws(bad):
    req = request()
    if bad == "probability":
        req = replace(req, truth=replace(req.truth, probability=req.truth.probability * 0.5))
    elif bad == "choices":
        req = replace(req, bootstrap_from_truth="n")
    else:
        req = replace(req, truth=sppcr_truth([5e-324], 20, [1], 1, progenitor=(0, 0)))
    rng = RandlibGenerator()
    before = rng.get_seeds()
    with pytest.raises((ValueError, ArithmeticError)):
        sppcr_simulate(req, rng=rng, replicates=3)
    assert rng.get_seeds() == before


def test_legacy_guard_before_initial_sample():
    t = sppcr_truth([100], 20, [1], 1, progenitor=(0, 0))
    # Exact endpoint probabilities are supported; saturation='half' handles the fit.
    result = sppcr_simulate(
        SPPCRSimulationRequest(t, True, False), rng=RandlibGenerator(), replicates=2
    )
    assert_array_equal(result.data.seen, [[20]])
    t = sppcr_truth([20], 20, [1], 1, progenitor=(0, 0))
    rng = RandlibGenerator()
    before = rng.get_seeds()
    with pytest.raises(ValueError, match="endpoint"):
        sppcr_simulate(SPPCRSimulationRequest(t, True, False), rng=rng, replicates=2)
    assert rng.get_seeds() == before


def test_report_combined_limits_and_explicit_output_override():
    result = sppcr_simulate(request(), rng=np.random.default_rng(1), replicates=3)
    output = format_sppcr_analysis(result)
    assert (
        format_sppcr_analysis(result, max_report_characters=len(output.report)).report
        == output.report
    )
    with pytest.raises(ValueError):
        format_sppcr_analysis(result, max_report_characters=len(output.report) - 1)
    with pytest.raises(ValueError):
        format_sppcr_analysis(result, max_simulation_characters=1)
    assert format_sppcr_analysis(result, write_simulations=False).simulations is None
    with pytest.raises(ValueError, match="boolean"):
        format_sppcr_analysis(result, write_simulations="n")


def test_report_rejects_different_generating_design():
    result = sppcr_simulate(request(), rng=np.random.default_rng(1), replicates=3)
    t = sppcr_truth([1, 2], [50, 100], [2, 5, 3], 1.7, progenitor=(0, 1))
    with pytest.raises(ValueError, match="does not match"):
        format_sppcr_analysis(replace(result, request=SPPCRSimulationRequest(t, True, True)))


def test_report_rejects_changed_bootstrap_choice():
    result = sppcr_simulate(request(), rng=np.random.default_rng(1), replicates=3)
    changed = replace(result.request, bootstrap_from_truth=False)
    with pytest.raises(ValueError, match="bootstrap model"):
        format_sppcr_analysis(replace(result, request=changed))
