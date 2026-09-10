"""Reusable SPPCR observed-data and truth-simulation analysis workflows."""

from dataclasses import dataclass

import numpy as np

from .randlib import RandlibGenerator
from .sppcr_bootstrap import SPPCRBootstrap, _bootstrap_options, _sample, sppcr_bootstrap
from .sppcr_data import SPPCRData, sppcr_data
from .sppcr_reporting import format_sppcr_report, format_sppcr_simulations
from .sppcr_truth_console import SPPCRSimulationRequest, _validate_request
from .sppcr_truth_reporting import format_sppcr_truth


@dataclass(frozen=True)
class SPPCRAnalysis:
    """One observed experiment, bootstrap and optional generating truth request."""

    data: SPPCRData
    bootstrap: SPPCRBootstrap
    request: SPPCRSimulationRequest | None = None


@dataclass(frozen=True)
class SPPCRReports:
    """Ready-to-write analysis text and optional replicate text; no files opened."""

    report: str
    simulations: str | None


def _validate_data(data: SPPCRData) -> None:
    canonical = sppcr_data(
        data.genome_dna,
        data.seen,
        data.wells,
        data.allele_sizes,
        data.progenitor_sizes,
        unseen_alleles="retain",
    )
    if not np.array_equal(data.dna, canonical.dna) or data.progenitor != canonical.progenitor:
        raise ValueError("SPPCR data has inconsistent DNA units or progenitor mapping")


def sppcr_analyze(
    data: SPPCRData,
    *,
    rng: np.random.Generator | RandlibGenerator,
    replicates: int = 1000,
    saturation: str = "half",
) -> SPPCRAnalysis:
    """Analyze validated input data using observed-fraction bootstrap sampling.

    Original identities and omission diagnostics are retained. RNG is explicit;
    numerical failures propagate and do not roll back consumed randomness.
    """
    _validate_data(data)
    result = sppcr_bootstrap(
        data.dna,
        data.seen,
        data.wells,
        progenitor=data.progenitor,
        rng=rng,
        replicates=replicates,
        saturation=saturation,
    )
    return SPPCRAnalysis(data, result)


def _truth_data(request: SPPCRSimulationRequest, seen: np.ndarray) -> SPPCRData:
    t = request.truth
    labels = tuple(range(1, t.frequency.size + 1))
    with np.errstate(under="ignore"):
        genome_dna = t.dna / 2
    if np.any(genome_dna == 0) or not np.array_equal(2 * genome_dna, t.dna):
        raise ArithmeticError("model DNA cannot be represented exactly in genome input units")
    return sppcr_data(
        genome_dna,
        seen,
        t.wells,
        labels,
        (t.progenitor[0] + 1, t.progenitor[1] + 1),
        unseen_alleles="retain",
    )


def sppcr_simulate(
    request: SPPCRSimulationRequest,
    *,
    rng: np.random.Generator | RandlibGenerator,
    replicates: int = 1000,
    saturation: str = "half",
) -> SPPCRAnalysis:
    """Generate the initial experiment from truth, then fit and bootstrap it.

    The request chooses truth or observed-fraction bootstrap probabilities. The
    initial experiment always uses truth. One caller RNG continues through both
    stages, without clock reseeding. All alleles, including unseen ones, remain.
    Input/configuration validation precedes draws. Once sampling begins, numerical
    failures leave consumed RNG state in place. Model DNA must convert exactly to
    genome input units for the shared input/report model.
    """
    _validate_request(request)
    _bootstrap_options(rng, replicates, saturation)
    t = request.truth
    _truth_data(request, np.zeros(t.probability.shape))
    samples = _sample(t.probability, t.wells, rng=rng, replicates=1)
    data = _truth_data(request, samples.seen[0])
    result = sppcr_bootstrap(
        data.dna,
        data.seen,
        data.wells,
        progenitor=data.progenitor,
        rng=rng,
        replicates=replicates,
        saturation=saturation,
        probability=t.probability if request.bootstrap_from_truth else None,
    )
    return SPPCRAnalysis(data, result, request)


def format_sppcr_analysis(
    analysis: SPPCRAnalysis,
    *,
    write_simulations: bool | None = None,
    precision: int = 10,
    multiplier: float = 1.959964,
    max_report_characters: int = 1000000,
    max_simulation_characters: int = 10000000,
) -> SPPCRReports:
    """Compose bounded parameter/analysis reports and optional replicate output.

    Simulation output defaults to the truth request's choice, or false for an
    observed-data analysis. An explicit boolean overrides it. Report bounds apply
    to the combined truth and analysis text. No generation, fitting or file IO.
    """
    request = analysis.request
    if write_simulations is None:
        write_simulations = request.write_simulations if request is not None else False
    if not isinstance(write_simulations, bool):
        raise ValueError("write_simulations must be boolean or None")
    prefix = ""
    if request is not None:
        _validate_request(request)
        expected = _truth_data(request, analysis.data.seen)
        if (
            not np.array_equal(expected.dna, analysis.data.dna)
            or not np.array_equal(expected.wells, analysis.data.wells)
            or expected.allele_sizes != analysis.data.allele_sizes
            or expected.progenitor != analysis.data.progenitor
        ):
            raise ValueError("Generating truth does not match analysis design")
        probability = (
            request.truth.probability
            if request.bootstrap_from_truth
            else analysis.data.seen / analysis.data.wells[:, None]
        )
        actual = analysis.bootstrap.samples.probability
        if not (
            np.array_equal(actual, probability)
            or np.array_equal(actual, probability.astype(np.float32))
        ):
            raise ValueError("Requested bootstrap model does not match sampled probabilities")
        prefix = format_sppcr_truth(
            request, precision=precision, max_characters=max_report_characters
        )
    report = format_sppcr_report(
        analysis.data,
        analysis.bootstrap,
        precision=precision,
        multiplier=multiplier,
        max_characters=max_report_characters - len(prefix),
    )
    simulations = (
        format_sppcr_simulations(
            analysis.data,
            analysis.bootstrap,
            precision=precision,
            max_characters=max_simulation_characters,
        )
        if write_simulations
        else None
    )
    return SPPCRReports(prefix + report, simulations)
