"""Bounded SPPCR analysis and replicate text reports with explicit diagnostics."""

import math

import numpy as np

from ._validation import FloatArray
from .cdflib_console import _positive
from .sppcr_bootstrap import SPPCRBootstrap
from .sppcr_data import SPPCRData, sppcr_data
from .sppcr_frequencies import sppcr_frequencies
from .sppcr_intervals import sppcr_bootstrap_intervals


class _Report:
    def __init__(self, precision: int, max_characters: int):
        self.precision = _positive(precision, "precision")
        self.limit = _positive(max_characters, "max_characters")
        if precision > 17:
            raise ValueError("precision must not exceed 17 significant digits")
        self.lines: list[str] = []
        self.size = 0

    def number(self, value: float | FloatArray) -> str:
        x = float(value)
        return "NA" if math.isnan(x) else format(x, f".{self.precision}g")

    def row(self, *values: str) -> None:
        line = "\t".join(values) + "\n"
        self.size += len(line)
        if self.size > self.limit:
            raise ValueError("SPPCR report exceeds max_characters")
        self.lines.append(line)

    def finish(self) -> str:
        return "".join(self.lines)


def _validate(data: SPPCRData, result: SPPCRBootstrap) -> None:
    canonical = sppcr_data(
        data.genome_dna,
        data.seen,
        data.wells,
        data.allele_sizes,
        data.progenitor_sizes,
        unseen_alleles="retain",
    )
    if not np.array_equal(data.dna, canonical.dna) or data.progenitor != canonical.progenitor:
        raise ValueError("SPPCR report data has inconsistent units or progenitor mapping")
    fit = result.observed_fit
    if fit.mu.ndim != 1 or result.fit.mu.ndim != 2:
        raise ValueError("SPPCR reports require one observed experiment")
    for a, b in [(data.dna, fit.dna), (data.seen, fit.original_seen), (data.wells, fit.wells)]:
        if not np.array_equal(a, b):
            raise ValueError("SPPCR report data does not match the observed fit")
    if data.progenitor != result.summary.progenitor or len(data.allele_sizes) != fit.mu.size:
        raise ValueError("SPPCR report allele identities do not match the bootstrap")


def format_sppcr_report(
    data: SPPCRData,
    result: SPPCRBootstrap,
    *,
    precision: int = 10,
    multiplier: float = 1.959964,
    max_characters: int = 1_000_000,
) -> str:
    """Format one observed experiment and its bootstrap as labeled TSV sections.

    Original counts and both DNA units are retained. NA denotes unavailable or
    uncomputed uncertainty; inf denotes an unbounded reciprocal interval. Solver
    brackets replace the native solver's initial-guess diagnostic. No files open.
    """
    out = _Report(precision, max_characters)
    _validate(data, result)
    fit, boot = result.observed_fit, result.summary
    limits = sppcr_bootstrap_intervals(result, multiplier=multiplier)
    observed = (
        sppcr_frequencies(fit.mu, fit.variance, progenitor=data.progenitor)
        if np.any(fit.mu > 0)
        else None
    )
    n = out.number
    out.row("SPPCR analysis")
    out.row("NA = unavailable/not computed; inf = unbounded upper limit")
    out.row("Normal interval multiplier", n(limits.multiplier))
    out.row("Bootstrap replicates", str(len(result.fit.mu)))
    out.row("Undefined frequency replicates", str(np.count_nonzero(~boot.defined)))
    out.row("Progenitor sizes", *(str(x) for x in data.progenitor_sizes))
    out.row("Omitted never-seen allele sizes", *(str(x) for x in data.omitted_allele_sizes))
    out.row("[DATA]")
    out.row(
        "run", "DNA_genome_input", "DNA_model", "wells", *(f"seen_{x}" for x in data.allele_sizes)
    )
    for i in range(len(data.dna)):
        out.row(
            str(i + 1),
            n(data.genome_dna[i]),
            n(data.dna[i]),
            str(int(data.wells[i])),
            *(str(int(x)) for x in data.seen[i]),
        )
    out.row("[SAMPLING PROBABILITIES]")
    out.row("run", *(str(x) for x in data.allele_sizes))
    for i, row in enumerate(result.samples.probability):
        out.row(str(i + 1), *(n(x) for x in row))
    out.row("[MEANS]")
    out.row(
        "allele",
        "role",
        "estimate",
        "bootstrap_mean",
        "asymptotic_sd",
        "bootstrap_sd",
        "solver_lower",
        "solver_upper",
        "adjusted",
        "boundary",
        "bootstrap_adjusted",
        "bootstrap_boundary",
    )
    for i, size in enumerate(data.allele_sizes):
        out.row(
            str(size),
            "P" if i in data.progenitor else "M",
            n(fit.mu[i]),
            n(boot.mu.mean[i]),
            n(np.sqrt(fit.variance[i])),
            n(boot.mu.standard_deviation[i]),
            n(fit.mu_lower[i]),
            n(fit.mu_upper[i]),
            str(int(fit.adjusted[i])),
            str(int(not fit.interior[i])),
            str(np.count_nonzero(result.fit.adjusted[:, i])),
            str(np.count_nonzero(~result.fit.interior[:, i])),
        )
    out.row("[SUMMARY]")
    out.row(
        "quantity",
        "estimate",
        "bootstrap_mean",
        "asymptotic_sd",
        "bootstrap_sd",
        "lower",
        "upper",
        "available",
        "lower_clipped",
        "upper_clipped",
    )
    for name, interval, series, asym in [
        (
            "calibration",
            limits.calibration,
            boot.calibration,
            observed.calibration.standard_error if observed else np.nan,
        ),
        ("inverse_calibration", limits.inverse_calibration, None, np.nan),
        (
            "mutant",
            limits.mutant,
            boot.mutant,
            observed.mutant.standard_error if observed else np.nan,
        ),
    ]:
        out.row(
            name,
            n(interval.estimate),
            n(series.mean) if series else "NA",
            n(asym),
            n(series.standard_deviation) if series else "NA",
            n(interval.lower),
            n(interval.upper),
            str(int(interval.available)),
            str(int(interval.lower_clipped)),
            str(int(interval.upper_clipped)),
        )
    out.row("[FREQUENCIES]")
    out.row(
        "allele",
        "role",
        "estimate",
        "bootstrap_mean",
        "asymptotic_sd",
        "bootstrap_sd",
        "lower",
        "upper",
        "available",
        "lower_clipped",
        "upper_clipped",
    )
    f = limits.frequency
    for i, size in enumerate(data.allele_sizes):
        out.row(
            str(size),
            "P" if i in data.progenitor else "M",
            n(f.estimate[i]),
            n(boot.frequency.mean[i]),
            n(observed.frequency.standard_error[i]) if observed else "NA",
            n(boot.frequency.standard_deviation[i]),
            n(f.lower[i]),
            n(f.upper[i]),
            str(int(f.available[i])),
            str(int(f.lower_clipped[i])),
            str(int(f.upper_clipped[i])),
        )
    out.row("[TRANSFORMED FREQUENCIES: 2*asin(sqrt(p))]")
    out.row("allele_or_group", "estimate", "bootstrap_mean", "asymptotic_sd", "bootstrap_sd")
    for i, size in enumerate(data.allele_sizes):
        out.row(
            str(size),
            n(observed.frequency.transformed.value[i]) if observed else "NA",
            n(boot.transformed_frequency.mean[i]),
            n(observed.frequency.transformed.standard_error[i]) if observed else "NA",
            n(boot.transformed_frequency.standard_deviation[i]),
        )
    out.row(
        "mutant",
        n(observed.mutant.transformed.value) if observed else "NA",
        n(boot.transformed_mutant.mean),
        n(observed.mutant.transformed.standard_error) if observed else "NA",
        n(boot.transformed_mutant.standard_deviation),
    )
    return out.finish()


def format_sppcr_simulations(
    data: SPPCRData,
    result: SPPCRBootstrap,
    *,
    precision: int = 10,
    max_characters: int = 10_000_000,
) -> str:
    """TSV replicate estimates with identities, undefined values and boundary counts.

    Every replicate is retained. This replaces the native Cal/Mu/Freq text blocks
    with one labeled row per replicate. Sampling counts remain in result.samples.
    """
    out = _Report(precision, max_characters)
    _validate(data, result)
    s = result.summary
    out.row(
        "replicate",
        "frequency_defined",
        "calibration",
        *(f"mu_{x}" for x in data.allele_sizes),
        *(f"frequency_{x}" for x in data.allele_sizes),
        "mutant",
        "adjusted_alleles",
        "boundary_alleles",
    )
    for i in range(len(result.fit.mu)):
        out.row(
            str(i + 1),
            str(int(s.defined[i])),
            out.number(s.calibration.values[i]),
            *(out.number(x) for x in s.mu.values[i]),
            *(out.number(x) for x in s.frequency.values[i]),
            out.number(s.mutant.values[i]),
            str(np.count_nonzero(result.fit.adjusted[i])),
            str(np.count_nonzero(~result.fit.interior[i])),
        )
    return out.finish()
