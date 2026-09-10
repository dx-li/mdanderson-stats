"""Generation-aligned SOGS distributions and Monte Carlo summaries."""

from dataclasses import dataclass

import numpy as np

from ._cdflib import _freeze
from ._validation import FloatArray
from .sogs_simulation import SOGSSimulation


@dataclass(frozen=True)
class SOGSSummary:
    stages: tuple[str, ...]
    metrics: tuple[str, ...]
    mean: FloatArray
    mcse: FloatArray
    apparent_dmt_cdf: FloatArray
    missed_dmt_cdf: FloatArray
    replicates: int


def sogs_summary(result: SOGSSimulation) -> SOGSSummary:
    """CDFs and means/Monte Carlo SE, with every quantity on the same generation."""
    apparent = result.apparent_dmt
    total = result.donor_length[0, 0]
    values = np.stack(
        (
            apparent,
            result.missed,
            result.donor_length,
            result.missed_length,
            100 * result.donor_length / total,
            100 * result.missed_length / total,
        ),
        axis=-1,
    )
    mean = values.mean(axis=0)
    n = values.shape[0]
    mcse = values.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(mean.shape, np.nan)
    cdfs = []
    for counts in (apparent, result.missed):
        pmf = (
            np.stack(
                [
                    np.bincount(counts[:, j], minlength=len(result.chromosomes) + 1)
                    for j in range(counts.shape[1])
                ]
            )
            / n
        )
        cdf = np.cumsum(pmf, axis=1)
        cdf[:, -1] = 1.0
        cdfs.append(_freeze(cdf))
    stages = ("F1",) + tuple(f"BC{i}" for i in range(1, len(result.offspring) + 1))
    metrics = (
        "apparent_dmt",
        "missed_dmt",
        "donor_cm",
        "missed_cm",
        "donor_percent",
        "missed_percent",
    )
    return SOGSSummary(stages, metrics, _freeze(mean), _freeze(mcse), cdfs[0], cdfs[1], n)


def format_sogs(result: SOGSSimulation) -> str:
    """Return the simulation configuration, generation means and count CDF tables."""
    summary = sogs_summary(result)
    lines = [
        "SOGS genotype-selection simulation",
        f"Replicates: {summary.replicates}; seed: {result.seed}; selection rule: {result.rule}",
        f"Simulated chromosomes: {', '.join(map(str, result.chromosomes))}",
        f"Eligible offspring per backcross: {', '.join(map(str, result.offspring))}",
        f"Screening error: {result.screening_error}; avoidance: {result.avoidance:g} cM",
        f"Avoidance placements accepted after nine failed attempts: {result.avoidance_failures}",
        "F1 is the initial state; BCk is the selected parent after backcross k.",
        "Monte Carlo SE is unavailable (nan) for a single replicate.",
    ]
    for j, metric in enumerate(summary.metrics):
        lines.extend(["", metric, "Stage          Mean         MCSE"])
        for stage, mean, se in zip(
            summary.stages, summary.mean[:, j], summary.mcse[:, j], strict=True
        ):
            lines.append(f"{stage:<8} {mean:12.6g} {se:12.6g}")
    for title, cdf in [
        ("Apparent DMT count", summary.apparent_dmt_cdf),
        ("Missed DMT count", summary.missed_dmt_cdf),
    ]:
        lines.extend(
            ["", f"{title}: P(count <= k)", "  k " + " ".join(f"{s:>9}" for s in summary.stages)]
        )
        for k in range(len(result.chromosomes) + 1):
            lines.append(f"{k:3} " + " ".join(f"{v:9.6f}" for v in cdf[:, k]))
    return "\n".join(lines) + "\n"
